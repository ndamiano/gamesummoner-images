"""Turn a JEV-27B-VL checkout and its NVFP4 base into what SGLang and contract/decide.py load.

SGLang's lm_head LoRA runs F.linear on the raw lm_head weight, which is FP8 in the NVFP4 base, so
the adapter is split:
  model/                 the base's weights and config with JEV's tokenizer and templates (symlinks)
  adapter_sglang/        the backbone LoRA only -> sglang --lora-paths
  readout.safetensors    lm_head rows for the verbalizer tokens (dequantized) and the lm_head LoRA
                         rows for them, scaling folded in: logits = h @ (W + B @ A)^T
  decision_head.json, calibration.json   copied for the head
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE_FILES = ("config.json", "model.safetensors", "model_mtp.safetensors",
              "model.safetensors.index.json")


def assemble_model(jev: Path, base: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for src in list(jev.iterdir()) + [base / f for f in BASE_FILES]:
        if src.is_file() and (src.parent == base or src.name not in BASE_FILES):
            link = out / src.name
            link.unlink(missing_ok=True)
            link.symlink_to(src.resolve())


def split_in_proj_qkv(weights: dict, base: Path) -> dict:
    """HF/vLLM layout has in_proj_qkv (q|k|v rows) + in_proj_z; SGLang loads in_proj_q/k/v/z and
    stacks them into in_proj_qkvz. Exact: each slice keeps the shared lora_A and takes its rows
    of lora_B."""
    text = json.loads((base / "config.json").read_text())["text_config"]
    key_dim = text["linear_key_head_dim"] * text["linear_num_key_heads"]
    value_dim = text["linear_value_head_dim"] * text["linear_num_value_heads"]
    sizes = {"q": key_dim, "k": key_dim, "v": value_dim}
    out = {}
    for k, t in weights.items():
        if ".in_proj_qkv." not in k:
            out[k] = t
            continue
        if "lora_A" in k:
            for s in sizes:
                out[k.replace(".in_proj_qkv.", f".in_proj_{s}.")] = t.clone()
        else:
            assert t.shape[0] == sum(sizes.values()), (k, t.shape)
            for s, part in zip(sizes, torch.split(t, list(sizes.values()), dim=0)):
                out[k.replace(".in_proj_qkv.", f".in_proj_{s}.")] = part.contiguous()
    return out


def prep(jev: Path, base: Path, out: Path) -> None:
    src = jev / "adapter_vllm"
    cfg = json.loads((src / "adapter_config.json").read_text())
    head = json.loads((src / "decision_head.json").read_text())
    ids = head["verbalizer_ids"]
    scaling = cfg["lora_alpha"] / cfg["r"]

    backbone, lm_a, lm_b = {}, None, None
    with safe_open(src / "adapter_model.safetensors", "pt") as f:
        for k in f.keys():
            if ".lm_head." in k:
                if "lora_A" in k:
                    lm_a = f.get_tensor(k)
                else:
                    lm_b = f.get_tensor(k)
            else:
                backbone[k] = f.get_tensor(k)
    assert lm_a is not None and lm_b is not None, "adapter has no lm_head LoRA"

    assemble_model(jev, base, out / "model")
    adapter = out / "adapter_sglang"
    adapter.mkdir(parents=True, exist_ok=True)
    save_file(split_in_proj_qkv(backbone, base), adapter / "adapter_model.safetensors",
              metadata={"format": "pt"})
    targets = [m for m in cfg["target_modules"] if m not in ("lm_head", "in_proj_qkv", "in_proj_z")]
    cfg["target_modules"] = targets + ["in_proj_qkvz"]
    (adapter / "adapter_config.json").write_text(json.dumps(cfg, indent=2))

    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(base / index["lm_head.weight"], "pt") as f:
        w = f.get_slice("lm_head.weight")[ids]
        scale = f.get_slice("lm_head.weight_scale")[ids] if "lm_head.weight_scale" in index else None
    w = w.float() * (scale.float() if scale is not None else 1.0)
    save_file({
        "ids": torch.tensor(ids, dtype=torch.int64),
        "weight": w.contiguous(),
        "lora_a": lm_a.float().contiguous(),
        "lora_b": (lm_b[ids].float() * scaling).contiguous(),
    }, out / "readout.safetensors")
    shutil.copyfile(src / "decision_head.json", out / "decision_head.json")
    shutil.copyfile(jev / "calibration.json", out / "calibration.json")
    print(f"prepped {out}: {len(backbone)} backbone tensors, {len(ids)} readout rows, "
          f"lora scaling {scaling}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jev", required=True, type=Path, help="autotrust/JEV-27B-VL checkout")
    parser.add_argument("--base", required=True, type=Path, help="the NVFP4 base checkout")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    prep(args.jev, args.base, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
