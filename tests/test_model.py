import json
import pytest
from gs_object_extraction.model import point_cloud, training_roles


def model(tmp_path, names, eval_flag="True"):
    (tmp_path / "cameras.json").write_text(json.dumps([{"img_name": n} for n in names]))
    (tmp_path / "cfg_args").write_text(f"Namespace(eval={eval_flag}, images='images')")
    return tmp_path


def test_eval_models_hold_out_every_eighth_sorted_photo_listed_first(tmp_path):
    names = [f"{i:05d}" for i in range(17)]
    held = ["00000", "00008", "00016"]
    roles = training_roles(model(tmp_path, held + [n for n in names if n not in held]))
    assert sorted(n for n, r in roles.items() if r == "test") == held and len(roles) == 17


def test_extensions_are_dropped_and_non_eval_models_train_on_everything(tmp_path):
    roles = training_roles(model(tmp_path, ["b.jpg", "a.jpg"], "False"))
    assert roles == {"b": "train", "a": "train"}


def test_unexpected_camera_order_is_refused(tmp_path):
    names = [f"{i:05d}" for i in range(9)]
    with pytest.raises(ValueError, match="held-out cameras first"):
        training_roles(model(tmp_path, names[1:] + names[:1]))


def test_point_cloud_defaults_to_the_latest_iteration(tmp_path):
    for it in (7000, 30000):
        (tmp_path / "point_cloud" / f"iteration_{it}").mkdir(parents=True)
        (tmp_path / "point_cloud" / f"iteration_{it}" / "point_cloud.ply").write_text("")
    assert point_cloud(tmp_path).parent.name == "iteration_30000"
    assert point_cloud(tmp_path, 7000).parent.name == "iteration_7000"
    with pytest.raises(FileNotFoundError):
        point_cloud(tmp_path, 1)
