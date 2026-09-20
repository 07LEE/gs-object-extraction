import os
import stat
import time

import numpy as np
import pytest

from gs_object_extraction import atomic
from gs_object_extraction.ply import load_ply, save_ply
from gs_object_extraction.scene import GaussianScene


def make_scene(sh_count=16):
    rng = np.random.default_rng(7)
    q = rng.normal(size=(5, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return GaussianScene(
        rng.normal(size=(5, 3)), np.exp(rng.normal(size=(5, 3))), q,
        np.array([0.0, 0.12, 0.5, 0.97, 1.0]), rng.normal(size=(5, sh_count, 3)),
        ids=np.array([13, 8, 47, 0, 999]),
        extras={"confidence": rng.random(5), "label": np.array([0, 2, 4, 8, 255], dtype=np.uint8),
                "nx": rng.normal(size=5).astype(np.float32)},
    )


@pytest.mark.parametrize("sh_count", [1, 4, 9, 16])
def test_binary_semantic_roundtrip_preserves_all_sh_ids_and_extras(tmp_path, sh_count):
    original = make_scene(sh_count)
    path = tmp_path / "scene.ply"
    save_ply(original, path)
    assert path.read_bytes().startswith(b"ply\nformat binary_little_endian 1.0\n")
    loaded = load_ply(path)
    for name in ("means", "scales", "quaternions", "opacities", "sh"):
        np.testing.assert_allclose(getattr(loaded, name), getattr(original, name), rtol=3e-7, atol=3e-7)
    # Covariances square scales and rotate them, amplifying float32 roundoff.
    np.testing.assert_allclose(loaded.covariances, original.covariances, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(loaded.ids, original.ids)
    assert loaded.extras.keys() == original.extras.keys()
    for name, expected in original.extras.items():
        np.testing.assert_array_equal(loaded.extras[name], expected)
        assert loaded.extras[name].dtype == expected.dtype


def write_ascii(path, extra_properties=(), rows=None, rest_count=24):
    properties = [("float", name) for name in
                  ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                   "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]]
    # Deliberately nonnumeric order: catches lexicographic SH sorting.
    rest = sorted([f"f_rest_{i}" for i in range(rest_count)])
    properties += [("float", name) for name in rest]
    properties += list(extra_properties)
    if rows is None:
        values = {name: 0 for _, name in properties}
        values.update({"x": 1, "y": 2, "z": 3, "rot_0": 2, "opacity": -1000,
                       "f_dc_0": 0.1, "f_dc_1": 0.2, "f_dc_2": 0.3})
        values.update({f"f_rest_{i}": i + 1 for i in range(rest_count)})
        values.update({name: 7 for _, name in extra_properties})
        rows = [[values[name] for _, name in properties]]
    text = ["ply", "format ascii 1.0", "comment example", f"element vertex {len(rows)}"]
    text += [f"property {kind} {name}" for kind, name in properties]
    text += ["end_header"] + [" ".join(map(str, row)) for row in rows]
    path.write_text("\n".join(text) + "\n")


def test_ascii_numeric_sh_order_channel_major_and_raw_activation(tmp_path):
    path = tmp_path / "ascii.ply"
    write_ascii(path, [("ushort", "category"), ("int", "gaussian_id")])
    loaded = load_ply(path)
    np.testing.assert_array_equal(loaded.means, [[1, 2, 3]])
    np.testing.assert_array_equal(loaded.scales, [[1, 1, 1]])
    np.testing.assert_array_equal(loaded.quaternions, [[1, 0, 0, 0]])
    assert loaded.opacities[0] == 0
    np.testing.assert_array_equal(loaded.sh[0, 1:, 0], np.arange(1, 9))
    np.testing.assert_array_equal(loaded.sh[0, 1:, 1], np.arange(9, 17))
    np.testing.assert_array_equal(loaded.sh[0, 1:, 2], np.arange(17, 25))
    np.testing.assert_array_equal(loaded.ids, [7])
    assert loaded.extras["category"].dtype == np.uint16
    assert "gaussian_id" not in loaded.extras


def test_reject_ordinary_pointcloud_and_mesh(tmp_path):
    path = tmp_path / "cloud.ply"
    path.write_text("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n")
    with pytest.raises(ValueError, match="not a Graphdeco Gaussian"):
        load_ply(path)
    path.write_text("ply\nformat ascii 1.0\nelement vertex 0\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n3 0 1 2\n")
    with pytest.raises(ValueError, match="meshes"):
        load_ply(path)


def test_reject_vertex_lists_and_unsupported_sh_without_silent_drop(tmp_path):
    path = tmp_path / "invalid.ply"
    path.write_text("ply\nformat ascii 1.0\nelement vertex 1\nproperty list uchar float samples\nend_header\n0\n")
    with pytest.raises(ValueError, match="list-valued"):
        load_ply(path)
    write_ascii(path, rest_count=3)
    with pytest.raises(ValueError, match="unsupported SH"):
        load_ply(path)
    write_ascii(path)
    path.write_text(path.read_text().replace("float f_rest_23", "float f_rest_99"))
    with pytest.raises(ValueError, match="contiguous"):
        load_ply(path)


def test_truncated_binary_and_bad_ids_rejected(tmp_path):
    path = tmp_path / "scene.ply"
    save_ply(make_scene(), path)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="truncated"):
        load_ply(path)
    write_ascii(path, [("float", "gaussian_id")])
    text = path.read_text().rsplit("7\n", 1)
    path.write_text("1.5\n".join(text))
    with pytest.raises(ValueError, match="integral"):
        load_ply(path)


def test_scene_subset_copy_and_rotated_covariance(tmp_path):
    scene = GaussianScene.from_colors([[0, 0, 0]], [[1, 2, 3]], [[0.2, 0.4, 0.6]],
                                     quaternions=[[np.sqrt(0.5), 0, 0, np.sqrt(0.5)]])
    np.testing.assert_allclose(scene.covariances[0], np.diag([4, 1, 9]), atol=1e-14)
    np.testing.assert_allclose(scene.colors, [[0.2, 0.4, 0.6]])
    original = make_scene()
    subset = original.subset(np.array([False, True, False, True, False]))
    np.testing.assert_array_equal(subset.ids, [8, 0])
    np.testing.assert_array_equal(subset.extras["label"], [2, 8])
    subset.means[:] = 500
    assert not np.any(original.means == 500)
    copied = original.copy()
    copied.extras["label"][:] = 99
    assert not np.any(original.extras["label"] == 99)
    empty = original.subset(np.zeros(len(original), dtype=bool))
    path = tmp_path / "empty.ply"
    save_ply(empty, path)
    loaded = load_ply(path)
    assert len(loaded) == 0
    assert loaded.sh.shape == (0, 16, 3)


@pytest.mark.parametrize("change, message", [
    ({"scales": np.zeros((5, 3))}, "strictly positive"),
    ({"quaternions": np.ones((5, 4))}, "unit length"),
    ({"opacities": np.full(5, 2.0)}, "in \\[0, 1\\]"),
    ({"sh": np.zeros((5, 2, 3))}, "sh must"),
    ({"ids": np.zeros(5, dtype=int)}, "unique"),
    ({"means": np.full((5, 3), np.nan)}, "finite"),
])
def test_scene_validation(change, message):
    scene = make_scene()
    parameters = {name: getattr(scene, name) for name in
                  ("means", "scales", "quaternions", "opacities", "sh", "ids", "extras")}
    parameters.update(change)
    with pytest.raises(ValueError, match=message):
        GaussianScene(**parameters)


def test_invalid_export_does_not_truncate_existing_file(tmp_path):
    scene = make_scene()
    scene.extras["f_rest_100"] = np.ones(len(scene))
    path = tmp_path / "existing.ply"
    path.write_bytes(b"original")
    with pytest.raises(ValueError, match="conflicts"):
        save_ply(scene, path)
    assert path.read_bytes() == b"original"


def test_a_failed_write_keeps_the_previous_file_and_leaves_no_temporary(tmp_path, monkeypatch):
    path = tmp_path / "object.ply"
    path.write_bytes(b"previous export")
    # A full disk usually surfaces on the flush, once the data is already written out.
    monkeypatch.setattr(atomic.os, "fsync", lambda descriptor: (_ for _ in ()).throw(OSError("no space left")))
    with pytest.raises(OSError, match="no space left"):
        save_ply(make_scene(), path)
    assert path.read_bytes() == b"previous export"
    assert list(tmp_path.iterdir()) == [path]


def test_save_flushes_the_data_and_the_directory_entry(tmp_path, monkeypatch):
    synced = []
    real = atomic.os.fsync
    def record(descriptor):
        synced.append(stat.S_ISDIR(os.fstat(descriptor).st_mode))
        return real(descriptor)
    monkeypatch.setattr(atomic.os, "fsync", record)
    save_ply(make_scene(), tmp_path / "object.ply")
    # The bytes first, then the rename that points at them.
    assert synced == [False, True]


def test_save_clears_a_killed_run_s_temporary_but_never_a_live_one(tmp_path):
    path = tmp_path / "object.ply"
    stale = tmp_path / f".{path.name}.abcdef.tmp"
    stale.write_bytes(b"killed mid-export")
    old = time.time() - atomic.STALE_SECONDS - 60
    os.utime(stale, (old, old))
    live = tmp_path / f".{path.name}.123456.tmp"  # another process, still writing
    live.write_bytes(b"in flight")
    unrelated = tmp_path / ".other.ply.abcdef.tmp"
    unrelated.write_bytes(b"another target")
    os.utime(unrelated, (old, old))

    save_ply(make_scene(), path)
    assert not stale.exists()
    assert live.read_bytes() == b"in flight"
    assert unrelated.read_bytes() == b"another target"
    assert len(load_ply(path)) == 5


def test_save_keeps_an_existing_targets_mode_and_is_private_until_complete(tmp_path):
    shared = tmp_path / "shared.ply"
    shared.write_bytes(b"previous export")
    shared.chmod(0o640)
    save_ply(make_scene(), shared)
    assert stat.S_IMODE(shared.stat().st_mode) == 0o640
    reference = tmp_path / "reference.txt"
    reference.write_text("x")  # an ordinary new file gets the umask default
    fresh = tmp_path / "fresh.ply"
    save_ply(make_scene(), fresh)
    assert stat.S_IMODE(fresh.stat().st_mode) == stat.S_IMODE(reference.stat().st_mode)
