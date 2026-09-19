import json
from types import SimpleNamespace
import numpy as np
import pytest
from gs_object_extraction import cli
from gs_object_extraction import extract as ex
from gs_object_extraction.ply import load_ply
from gs_object_extraction.renderer import Lifted
from gs_object_extraction.scene import GaussianScene

# The 1 x 6 fixture of test_extract: Gaussian 3 is junk hidden by the floor Gaussian 4.
MASK = np.array([[0, 0, 1, 1, 1, 0]], bool)
FOOTPRINT = [{2: 1., 3: 1.}, {4: 1.}, {0: 1., 1: 1.}, {3: .5, 5: 1.}, {5: 1.}]
HIDDEN_BY = {(3, 5): 4}


class FakeRenderer:
    n = len(FOOTPRINT)

    def weights(self, active):
        active = np.ones(self.n, bool) if active is None else active
        w = np.zeros((self.n, MASK.shape[1]))
        for i, cover in enumerate(FOOTPRINT):
            for px, v in cover.items():
                blocker = HIDDEN_BY.get((i, px))
                if active[i] and not (blocker is not None and active[blocker]):
                    w[i, px] = v
        return w

    def lift(self, camera, labels, *, active=None):
        w, labels = self.weights(active), labels[0]
        return Lifted(w[:, labels == 1].sum(1), w[:, labels == 0].sum(1), w.sum(1))

    def render(self, camera, *, active=None):
        return SimpleNamespace(alpha=np.minimum(self.weights(active).sum(0), 1)[None])


@pytest.fixture
def fake_scene(monkeypatch):
    views = [(None, MASK)]
    scene = GaussianScene.from_colors(np.arange(15.).reshape(5, 3), np.full((5, 3), .1),
                                      np.full((5, 3), .5), np.full(5, .9))
    monkeypatch.setattr(cli, "usid_views", lambda scene_dir, model_dir: (views, views))
    monkeypatch.setattr(cli, "point_cloud", lambda model_dir: "point_cloud.ply")
    monkeypatch.setattr(cli, "load_ply", lambda path: scene)
    monkeypatch.setattr(cli, "preview", lambda *args: None)
    monkeypatch.setattr("gs_object_extraction.renderer.GsplatRenderer", lambda scene: FakeRenderer())
    return views


def test_band_reaches_pruning_and_scoring_and_is_recorded(fake_scene, tmp_path, capsys):
    out = tmp_path / "out"
    cli.main(["extract", "--scene-dir", "s", "--model-dir", "m", "--output", str(out), "--band", "0"])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["options"] == {"threshold": ex.THRESHOLD, "rounds": ex.ROUNDS, "band": 0,
                                  "off_threshold": ex.OFF_MASK}
    assert summary["scores"]["selected"] == {"gaussians": 3, "dirt": 1 / 3, "missing": 0.0}
    assert summary["scores"]["cleaned"] == {"gaussians": 2, "dirt": 0.0, "missing": 0.0}
    assert len(load_ply(out / "object.ply").means) == 2
    assert json.loads(capsys.readouterr().out) == summary["scores"]


def test_default_band_is_used_for_scoring_too(fake_scene, tmp_path):
    out = tmp_path / "out"
    cli.main(["extract", "--scene-dir", "s", "--model-dir", "m", "--output", str(out)])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["options"]["band"] == ex.BAND
    # Every pixel of the 1 x 6 mask lies within 2 px of the boundary, so nothing is pruned or scored.
    assert summary["scores"]["selected"] == {"gaussians": 3, "dirt": 0.0, "missing": 0.0}
    assert summary["scores"]["cleaned"]["gaussians"] == 3


def usid_scene(root, names):
    Image = pytest.importorskip("PIL.Image")
    from test_colmap_usid import write_cameras, write_images
    for folder in ("images", "object_masks", "unseen_masks", "test_images", "test_object_masks", "sparse/0"):
        (root / folder).mkdir(parents=True)
    mask = np.zeros((18, 32), np.uint8)
    mask[4:10, 8:20] = 255
    for name in names:
        Image.new("RGB", (32, 18)).save(root / "images" / f"{name}.jpg")
        Image.fromarray(mask).save(root / "object_masks" / f"{name}.jpg")
        Image.new("L", (32, 18)).save(root / "unseen_masks" / f"{name}.jpg")
    Image.new("RGB", (32, 18)).save(root / "test_images" / "99999.jpg")
    Image.new("L", (32, 18)).save(root / "test_object_masks" / "99999.png")
    write_cameras(root / "sparse/0/cameras.bin", [(1, 1, 64, 36, (40., 40., 32., 18.))])
    photos = [f"{n}.jpg" for n in names] + ["99999.jpg"]
    write_images(root / "sparse/0/images.bin", [(i + 1, (1, 0, 0, 0), (0, 0, i), 1, p, 0) for i, p in enumerate(photos)])


def graphdeco_model(folder, names):
    folder.mkdir()
    (folder / "cameras.json").write_text(json.dumps([{"img_name": n} for n in names]))
    (folder / "cfg_args").write_text("Namespace(eval=True)")


def test_usid_views_follow_the_models_training_split(tmp_path):
    names = [f"{i:05d}" for i in range(10)]
    usid_scene(tmp_path / "scene", names)
    held = ["00000", "00008"]
    graphdeco_model(tmp_path / "model", held + [n for n in names if n not in held])
    train, test = cli.usid_views(tmp_path / "scene", tmp_path / "model")
    assert len(train) == 8 and len(test) == 2  # the object-removed photo is never used
    camera, mask = test[0]
    assert mask.shape == (camera.height, camera.width) and mask[5, 10] and not mask[0, 0]


def test_usid_views_refuse_photos_the_model_never_saw(tmp_path):
    names = [f"{i:05d}" for i in range(10)]
    usid_scene(tmp_path / "scene", names)
    graphdeco_model(tmp_path / "model", names[:7])
    with pytest.raises(ValueError, match="missing"):
        cli.usid_views(tmp_path / "scene", tmp_path / "model")


def test_off_threshold_reaches_pruning_and_is_recorded(fake_scene, tmp_path):
    out = tmp_path / "out"
    cli.main(["extract", "--scene-dir", "s", "--model-dir", "m", "--output", str(out), "--band", "0",
              "--off-threshold", "0.9"])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["options"]["off_threshold"] == .9
    # the junk Gaussian spills two thirds off-mask, which this limit tolerates
    assert summary["scores"]["cleaned"]["gaussians"] == summary["scores"]["selected"]["gaussians"] == 3
