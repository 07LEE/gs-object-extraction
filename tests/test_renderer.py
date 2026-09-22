import os
import numpy as np
import pytest
from gs_object_extraction.camera import Camera
from gs_object_extraction.renderer import Exclusive
from gs_object_extraction.scene import GaussianScene

cuda = pytest.mark.skipif(os.environ.get("GS_OBJECT_EXTRACTION_TEST_CUDA") != "1", reason="set GS_OBJECT_EXTRACTION_TEST_CUDA=1 in the GPU environment")


def test_a_held_renderer_turns_away_a_caller_that_will_not_wait():
    """The window asks without blocking, so its event loop never queues behind a lift."""
    import threading
    exclusive = Exclusive()
    taken, done, refused = threading.Event(), threading.Event(), []

    def hold():
        with exclusive.held() as obtained:
            assert obtained
            taken.set()
            assert done.wait(5)

    holder = threading.Thread(target=hold)
    holder.start()
    assert taken.wait(5)
    with exclusive.held(blocking=False) as obtained:
        refused.append(obtained)
    done.set()
    holder.join(5)
    with exclusive.held(blocking=False) as obtained:  # free again once the holder is gone
        refused.append(obtained)
    assert refused == [False, True]


def test_holding_a_renderer_nests_so_its_own_calls_still_work():
    exclusive = Exclusive()
    with exclusive.held() as outer, exclusive.held(blocking=False) as inner:
        assert outer and inner


def three_gaussian_scene():
    return GaussianScene.from_colors(np.array([[0, 0, 2.], [.05, .02, 2.6], [.4, 0, 3.]]),
                                     np.full((3, 3), .16), np.array([[.9, .1, .1], [.1, .8, .1], [.1, .1, .8]]),
                                     np.array([.9, .85, .8]))


def three_gaussians():
    from gs_object_extraction.renderer import GsplatRenderer
    return GsplatRenderer(three_gaussian_scene()), Camera.look_at((0, 0, 0), (0, 0, 2), width=24, height=24)


@cuda
def test_lift_matches_a_per_gaussian_basis_render():
    renderer, camera = three_gaussians()
    labels = np.full((24, 24), -1, dtype=np.int8)
    labels[8:14, 8:14] = 1
    labels[14:19, 10:17] = 0
    lifted = renderer.lift(camera, labels)
    basis = renderer.render_features(camera, np.eye(3))  # each channel tags one Gaussian
    np.testing.assert_allclose(lifted.inside, basis[labels == 1].sum(axis=0), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(lifted.outside, basis[labels == 0].sum(axis=0), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(lifted.total, basis.sum(axis=(0, 1)), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(renderer.render(camera).alpha, basis.sum(axis=2), rtol=2e-5, atol=2e-5)
    for tensor in (renderer.means, renderer.scales, renderer.rotations, renderer.opacities, renderer.sh):
        assert not tensor.requires_grad and tensor.grad is None


@cuda
def test_lift_with_active_renders_the_selection_on_its_own():
    renderer, camera = three_gaussians()
    labels = np.zeros((24, 24), dtype=np.int8)
    labels[8:14, 8:14] = 1
    active = np.array([False, True, True])  # the front Gaussian is left out, so the one behind shows through
    lifted = renderer.lift(camera, labels, active=active)
    basis = renderer.render_features(camera, np.eye(3), active=active)
    np.testing.assert_allclose(lifted.inside, basis[labels == 1].sum(axis=0), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(lifted.outside, basis[labels == 0].sum(axis=0), rtol=2e-5, atol=2e-5)
    assert lifted.total[0] == 0 and lifted.inside[1] > renderer.lift(camera, labels).inside[1]
    everyone = renderer.lift(camera, labels, active=np.ones(3, bool))
    np.testing.assert_allclose(everyone.inside, renderer.lift(camera, labels).inside, rtol=1e-6, atol=1e-7)


@cuda
def test_lift_stays_float32_but_accumulates_into_lifted_zeros_as_float64():
    """lift() itself is float32, straight off the GPU; summing views is where it widens."""
    renderer, camera = three_gaussians()
    labels = np.full((24, 24), -1, dtype=np.int8)
    labels[8:14, 8:14] = 1
    lifted = renderer.lift(camera, labels)
    assert lifted.inside.dtype == np.float32
    from gs_object_extraction.renderer import Lifted
    accumulated = Lifted.zeros(3)
    accumulated += lifted
    accumulated += lifted
    assert accumulated.inside.dtype == np.float64
    np.testing.assert_allclose(accumulated.inside, 2 * lifted.inside.astype(np.float64), rtol=1e-6, atol=1e-7)


def test_extract_select_and_off_share_tolerate_float32_lifted_arrays():
    """select() and off_share() run on whatever Lifted.zeros() accumulated into, float32 or float64."""
    from gs_object_extraction.extract import off_share, select
    from gs_object_extraction.renderer import Lifted
    inside32 = np.array([.7, .6, .04, 0.], dtype=np.float32)
    outside32 = np.array([.3, .4, 0., 0.], dtype=np.float32)
    lifted32 = Lifted(inside32, outside32, np.zeros(4, dtype=np.float32))
    lifted64 = Lifted(inside32.astype(np.float64), outside32.astype(np.float64), np.zeros(4))
    np.testing.assert_array_equal(select(lifted32), select(lifted64))
    np.testing.assert_allclose(off_share(inside32, outside32), off_share(inside32.astype(np.float64), outside32.astype(np.float64)),
                               rtol=1e-6, atol=1e-7)


@cuda
def test_concurrent_lifts_do_not_corrupt_each_others_pinned_buffer():
    """Two threads hammering lift() must each see only their own call's values, never a torn read.

    Needs enough Gaussians that a call takes long enough to actually overlap another thread's;
    with only a few, both threads finish inside the GIL's own scheduling slice and never race.
    """
    import threading
    from gs_object_extraction.camera import Camera
    from gs_object_extraction.renderer import GsplatRenderer
    rng = np.random.default_rng(0)
    n = 30000
    means = rng.normal(scale=1., size=(n, 3))
    means[:, 2] += 5
    scene = GaussianScene.from_colors(means, np.full((n, 3), .05), rng.random((n, 3)), np.full(n, .9))
    renderer = GsplatRenderer(scene)
    camera = Camera.look_at((0, 0, 0), (0, 0, 5), width=64, height=64)
    labels_a = np.full((64, 64), -1, dtype=np.int8)
    labels_a[10:30, 10:30] = 1
    labels_b = np.full((64, 64), -1, dtype=np.int8)
    labels_b[34:54, 34:54] = 1
    reference_a, reference_b = renderer.lift(camera, labels_a), renderer.lift(camera, labels_b)
    # gsplat's backward accumulates with atomics, so even repeated single-threaded calls on the
    # same inputs differ at this order; a torn read off another thread's labels is far larger.
    mismatches = []

    def hammer(labels, reference):
        for _ in range(40):
            lifted = renderer.lift(camera, labels)
            if not np.allclose(lifted.inside, reference.inside, rtol=1e-3, atol=1e-5):
                mismatches.append(lifted.inside)

    threads = [threading.Thread(target=hammer, args=(labels_a, reference_a)),
              threading.Thread(target=hammer, args=(labels_b, reference_b))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not mismatches


@cuda
def test_lift_batch_matches_summing_individual_lifts():
    renderer, camera = three_gaussians()
    masks = [np.zeros((24, 24), bool), np.zeros((24, 24), bool)]
    masks[0][8:14, 8:14] = True
    masks[1][14:19, 10:17] = True
    views = [(camera, m) for m in masks]
    from gs_object_extraction.masks import band_labels
    from gs_object_extraction.renderer import Lifted
    reference = Lifted.zeros(3)
    for m in masks:
        reference += renderer.lift(camera, band_labels(m, 0))
    batched = renderer.lift_batch(views)
    np.testing.assert_allclose(batched.inside, reference.inside, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(batched.outside, reference.outside, rtol=1e-4, atol=1e-6)
    calls = []
    renderer.lift_batch(views, on_view=lambda: calls.append(1))
    assert calls == [1, 1]  # once per view, not once per round



@cuda
def test_lift_batch_still_rejects_one_bad_view_even_though_the_round_sums():
    """A single bad view must fail lift_batch, not get averaged away by the rest of the round.

    If lift_batch only checked the final sum, a large positive first view could hide a
    corrupt negative second view. Poisoning only the second view's gradient in place, with
    a first view large enough to swamp it in the total, tells the two cases apart.
    """
    import torch
    renderer, camera = three_gaussians()
    good = np.ones((24, 24), bool)  # every pixel: the largest inside contribution three Gaussians can give
    small = np.zeros((24, 24), bool)
    small[8:9, 8:9] = True
    views = [(camera, good), (camera, small)]
    real_grad = torch.autograd.grad
    calls = []

    def poisoning_grad(*args, **kwargs):
        out = real_grad(*args, **kwargs)
        calls.append(1)
        if len(calls) == 2:  # the second view: corrupt its own gradient before lift_batch checks it
            grad = out[0].clone()
            grad[0, 0] = -1.
            out = (grad,)
        return out

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch.autograd, "grad", poisoning_grad)
        with pytest.raises(RuntimeError, match="invalid"):
            renderer.lift_batch(views)
    assert len(calls) == 2  # the failure was caught right after the second view, not the first
