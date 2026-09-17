"""Sprite sheets rendered from a rigged mesh.

What is pinned here is what runs without a GPU: which silhouettes get a skeleton at all, that the
skeleton is the one a mocap clip can address, that a clip's dead head and tail do not eat the
sheet's cells, and that the manifest is the one `lib/sprites.js` already reads. The rig, the
retarget and the render themselves need kimodo and a browser, and run only on the mesh box.
"""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from engines.sprites.sprites3d import retarget, sheet, verbs  # noqa: E402
from engines.sprites.sprites3d.rig import humanoid, skeleton  # noqa: E402

# proportions measured off the five characters the pipeline was built on
BODIES = {
    "aldric": (1.00, 1.00, 0.33, 0.43),
    "brann": (1.00, 0.65, 0.58, 0.32),
    "sera": (1.00, 0.94, 0.30, 0.35),
    "shopkeeper": (0.97, 1.04, 0.24, 0.23),
    "horse and rider": (0.94, 0.89, 0.39, 0.94),
}


def measured(height, arm, hip, depth):
    return {"height": height, "floor": 0.0, "arm_span": arm * height,
            "hip_width": hip * height, "depth": depth * height,
            "centre_x": 0.0, "centre_z": 0.0}


@pytest.mark.parametrize("name", ["aldric", "brann", "sera", "shopkeeper"])
def test_a_person_gets_a_skeleton(name):
    assert humanoid(measured(*BODIES[name]))


def test_a_horse_and_rider_does_not():
    """Depth is what separates them: a rider's arms are still arms, so arm span passes the horse
    at 0.89 — but a horse is nearly as deep as it is tall, and a person never is."""
    assert not humanoid(measured(*BODIES["horse and rider"]))


def test_a_card_does_not():
    assert not humanoid(measured(1.0, 0.02, 0.9, 0.01))


def test_the_skeleton_is_the_one_a_clip_can_address():
    bones = skeleton(measured(*BODIES["aldric"]))
    names = {n for n, _, _ in bones}
    assert set(retarget.BONE_MAP.values()) <= names
    parents = {n: p for n, p, _ in bones}
    assert parents["Hips"] is None
    assert parents["LeftForeArm"] == "LeftArm"
    assert parents["RightToeBase"] == "RightFoot"


def test_the_sides_are_crossed_once():
    """A mocap skeleton's left arm points the opposite way to ours, so the map crosses them —
    and must cross every limb, or one arm swings up while the other swings down."""
    for a, b in (("LeftArm", "RightArm"), ("LeftLeg", "RightUpLeg"), ("RightFoot", "LeftFoot")):
        assert retarget.BONE_MAP[a] == b


def test_the_window_is_where_the_motion_is():
    """A generated clip is mostly held pose: an 8.3s attack happened in 0.27-1.87s."""
    times = np.linspace(0, 8.0, 81)
    speed = np.zeros(80)
    speed[10:25] = 5.0
    lo, hi = sheet.window_of(times, speed)
    assert 0.9 <= lo <= 1.1
    assert 2.4 <= hi <= 2.7


def test_a_still_clip_keeps_its_whole_length():
    times = np.linspace(0, 2.0, 21)
    assert sheet.window_of(times, np.zeros(20)) == (0.0, 2.0)


def _cell(w=30, h=60, at=(20, 10)):
    im = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    im.paste(Image.new("RGBA", (w, h), (255, 0, 0, 255)), at)
    return im


def test_the_manifest_is_the_one_the_player_reads():
    clips = {(a, d): [_cell()] * 4 for a in ("walk", "attack") for d in sheet.DIRS}
    png, manifest = sheet.pack(clips, ["walk", "attack"])
    assert set(manifest) == {"cell", "dirs", "anims", "pivot"}
    assert manifest["dirs"] == ["front", "right", "back", "left"]
    assert set(manifest["anims"]) == {"walk", "attack"}
    walk = manifest["anims"]["walk"]
    assert walk["frames"] == 4 and walk["fps"] == 12
    assert sorted(walk["rows"]) == ["back", "front", "left", "right"]
    rows = [r for a in manifest["anims"].values() for r in a["rows"].values()]
    assert sorted(rows) == list(range(8))
    assert png.size == (manifest["cell"]["w"] * 4, manifest["cell"]["h"] * 8)


def test_one_box_for_every_cell():
    """Cropping each frame to its own bounds makes the character jump around inside its sprite."""
    clips = {(("walk"), d): [_cell(at=(20, 10)), _cell(at=(50, 30))] for d in sheet.DIRS}
    _, manifest = sheet.pack(clips, ["walk"])
    # the union spans x 20..80 and y 10..90, so the cell is 60 wide by 80 tall, scaled to 128
    assert manifest["cell"]["w"] == round(60 * 128 / 80)


def test_a_sheet_of_nothing_is_an_error():
    empty = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    with pytest.raises(ValueError):
        sheet.pack({("walk", d): [empty] for d in sheet.DIRS}, ["walk"])


def test_a_walk_repeats_and_an_attack_does_not():
    """A shorter shift compares fewer, more similar frames, so raw error alone calls every clip a
    tiny loop — measured, every one-shot's best match sat on the smallest shift searched."""
    frames, joints = 90, 6
    stride = np.linspace(0, 2 * np.pi, 31)[:-1]
    walk = np.zeros((frames, joints, 4))
    for f in range(frames):
        walk[f, :, 0] = np.sin(stride[f % 30])
        walk[f, :, 3] = np.cos(stride[f % 30])
    assert retarget.cycle_length(walk)["loops"] is True
    assert retarget.cycle_length(walk)["cycle_frames"] == 30

    swing = np.zeros((frames, joints, 4))
    ramp = np.concatenate([np.linspace(0, 1, 40), np.linspace(1, 0.2, 50)])
    swing[:, :, 0] = ramp[:, None]
    swing[:, :, 3] = (1 - ramp)[:, None]
    assert retarget.cycle_length(swing)["loops"] is False


def test_a_one_shot_says_so_in_the_manifest():
    clips = {("attack", d): [_cell()] * 3 for d in sheet.DIRS}
    _, manifest = sheet.pack(clips, ["attack"], loops={"attack": False})
    assert manifest["anims"]["attack"]["loops"] is False


def test_the_library_answers_the_names_and_the_prose():
    table = verbs.Table()
    assert table.resolve("attack") == "attack"
    assert table.resolve(verbs.VERBS["attack"]) == "attack"
    # kimodo capitalises a prompt and appends a period before encoding
    assert table.resolve(verbs.VERBS["death"].capitalize() + ".") == "death"
    assert table.resolve("lunges forward with a two-handed sword slash") == ""


def test_every_verb_has_an_embedding():
    table = verbs.Table()
    assert set(table.names) == set(verbs.VERBS)
    assert table.feats.shape[0] == len(verbs.VERBS)
    assert table.llm_dim == 4096
