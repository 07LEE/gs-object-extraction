import struct
import numpy as np
import pytest
from gs_object_extraction.colmap import (ColmapCamera, ColmapImage, camera_from_colmap, qvec_to_rotation,
                            read_cameras_binary, read_images_binary)


def write_cameras(path, cameras):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cid, model_id, w, h, params in cameras:
            f.write(struct.pack("<iiQQ", cid, model_id, w, h) + struct.pack(f"<{len(params)}d", *params))


def write_images(path, images):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for iid, q, t, cid, name, n2d in images:
            f.write(struct.pack("<i7di", iid, *q, *t, cid) + name.encode() + b"\0")
            f.write(struct.pack("<Q", n2d) + b"\1" * 24 * n2d)


def rotation_z(degrees):
    a = np.radians(degrees)
    return (np.cos(a / 2), 0, 0, np.sin(a / 2))


def test_binary_reader_keeps_poses_and_skips_keypoints(tmp_path):
    write_cameras(tmp_path / "cameras.bin", [(3, 1, 953, 536, (700., 700., 476.5, 268.)), (4, 0, 64, 48, (50., 32., 24.))])
    write_images(tmp_path / "images.bin", [(1, (1, 0, 0, 0), (1, 2, 3), 3, "00000.jpg", 2),
                                           (2, rotation_z(90), (0, 0, 1), 4, "00001.jpg", 0)])
    cams, imgs = read_cameras_binary(tmp_path / "cameras.bin"), read_images_binary(tmp_path / "images.bin")
    assert cams[3] == ColmapCamera(3, "PINHOLE", 953, 536, (700., 700., 476.5, 268.))
    assert cams[4].focal == (50., 50.) and cams[4].principal == (32., 24.)
    assert set(imgs) == {"00000.jpg", "00001.jpg"}
    assert imgs["00000.jpg"] == ColmapImage(1, "00000.jpg", (1, 0, 0, 0), (1, 2, 3), 3)
    assert imgs["00001.jpg"].camera_id == 4


def test_unsupported_camera_model_is_rejected(tmp_path):
    write_cameras(tmp_path / "cameras.bin", [(1, 2, 64, 48, (50., 32., 24., .1))])
    with pytest.raises(ValueError, match="undistorted pinhole"):
        read_cameras_binary(tmp_path / "cameras.bin")


def test_quaternion_is_world_to_camera_rotation():
    np.testing.assert_allclose(qvec_to_rotation((1, 0, 0, 0)), np.eye(3))
    np.testing.assert_allclose(qvec_to_rotation(rotation_z(90)) @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(qvec_to_rotation((2, 0, 0, 0)), np.eye(3))


def test_resized_camera_matches_colmap_projection_with_half_pixel_shift():
    cam = ColmapCamera(1, "PINHOLE", 953, 536, (700., 690., 480.2, 260.7))
    img = ColmapImage(1, "a.jpg", rotation_z(30), (.2, -.1, .5), 1)
    ours = camera_from_colmap(cam, img, 960, 540)
    point = np.array([.3, -.4, 4.])
    x = qvec_to_rotation(img.qvec) @ point + img.tvec
    colmap_uv = np.array([700 * x[0] / x[2] + 480.2, 690 * x[1] / x[2] + 260.7])  # continuous, pixel centre at .5
    ours_uv = (ours.world_to_camera[:3, :3] @ point + ours.world_to_camera[:3, 3])
    ours_uv = np.array([ours.fx * ours_uv[0] / ours_uv[2] + ours.cx, ours.fy * ours_uv[1] / ours_uv[2] + ours.cy])
    np.testing.assert_allclose(ours_uv, colmap_uv * [960 / 953, 540 / 536] - .5, atol=1e-9)


def test_usid_scene_indexes_views_masks_and_holdout(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    from gs_object_extraction.usid import UsidScene, load_mask
    root = tmp_path / "scene"
    for folder in ("images", "object_masks", "unseen_masks", "test_images", "test_object_masks", "sparse/0"):
        (root / folder).mkdir(parents=True)
    train, test = [f"{i:05d}.jpg" for i in range(10)], ["00010.jpg"]
    for name in train:
        Image.new("RGB", (32, 18)).save(root / "images" / name)
        for folder in ("object_masks", "unseen_masks"):
            Image.new("L", (32, 18)).save(root / folder / name)
    Image.new("RGB", (32, 18)).save(root / "test_images" / test[0])
    soft = np.zeros((18, 32), np.uint8)
    soft[4:8, 4:10], soft[4:8, 10:12] = 200, 100
    Image.fromarray(soft).save(root / "test_object_masks" / "00010.png")
    write_cameras(root / "sparse/0/cameras.bin", [(1, 1, 64, 36, (40., 40., 32., 18.))])
    write_images(root / "sparse/0/images.bin",
                 [(i + 1, (1, 0, 0, 0), (0, 0, i), 1, name, 0) for i, name in enumerate(train + test)])
    scene = UsidScene(root)
    assert [v.name for v in scene.split("test")] == ["00010"]
    assert len(scene.split("train")) == 10 and scene.split("test")[0].unseen_mask is None
    assert (scene.views[0].camera.width, scene.views[0].camera.fx, scene.views[0].camera.cx) == (32, 20., 15.5)
    assert scene.max_principal_offset == 0
    assert scene.graphdeco_holdout() == {"00000", "00008"}
    mask = load_mask(scene.split("test")[0].object_mask)
    assert mask.sum() == 24 and load_mask(scene.split("test")[0].object_mask, soft=True).max() == pytest.approx(200 / 255)
    assert UsidScene(root, width=16).views[0].camera.height == 9
    (root / "images" / "extra.jpg").write_bytes((root / "images" / train[0]).read_bytes())
    with pytest.raises(ValueError, match="without COLMAP poses"):
        UsidScene(root)
