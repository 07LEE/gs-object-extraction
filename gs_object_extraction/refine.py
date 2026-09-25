"""Refit the opacity, colour and size of an extracted object so it stands on its own.

In the scene, part of what makes the object look solid is drawn by Gaussians that are
not its own: the ground and the background behind and around it. Taken out alone it
shows through. The scene's own renders of the marked views say what the object looked
like there, so the object's Gaussians are refit to reproduce those pixels inside the
masks, fully opaque, and to leave the pixels outside the masks empty. Positions stay as
extraction left them; no source photo is read.
"""

import numpy as np
from .masks import band_labels

STEPS = 300
OPACITY_RATE, COLOUR_RATE, SIZE_RATE = .05, .01, .01  # Adam step sizes: the opacity logit, the base colour, the log of a size factor
SIZE_LIMITS = (.5, 2.)  # a size may shrink or grow by at most this factor
OUTSIDE = 10.  # weight of the empty-outside term against the filled-inside one; at this weight the refit fills the object without adding stray alpha outside the masks (360-USID, 16 views)
BAND = 2


def targets_from(scene_renderer, views, band=BAND):
    """``(camera, scene render as 0..1 RGB, labels)`` per marked view: the scene's own pixels inside the mask are the goal."""
    return [(camera, scene_renderer.render_image(camera).astype(np.float32) / 255, band_labels(mask, band))
            for camera, mask in views]


def refine(renderer, targets, *, steps=STEPS, on_step=None, seed=0, fit_scales=True):
    """Fit ``renderer``'s Gaussians to ``targets``; returns the new ``opacities``, ``sh`` and ``scale_factors`` and the mask coverage before and after.

    ``renderer`` is a ``GsplatRenderer`` of the object alone and is left as it was; the fitted
    values come back for the caller to keep. Coverage is the mean alpha inside the masks.
    Sizes are fitted within a factor of two either way: the Gaussians at the edge shrink to
    leave the outside empty and the ones with gaps between them grow to close them.
    """
    if not targets:
        raise ValueError("refining needs at least one marked view")
    torch = renderer.torch
    with renderer._lock:
        original = renderer.opacities, renderer.sh, renderer.scales
        first_scales = renderer.scales.detach().clone()
        rest = renderer.sh[:, 1:].detach()
        logit = torch.logit(renderer.opacities.detach().clamp(1e-4, 1 - 1e-4)).clone().requires_grad_(True)
        base = renderer.sh[:, :1].detach().clone().requires_grad_(True)
        grow = torch.zeros_like(first_scales, requires_grad=True)  # log of the factor each size is multiplied by
        prepared = [(camera, renderer._tensor(rgb), torch.as_tensor(labels == 1, device="cuda"),
                     torch.as_tensor(labels == 0, device="cuda")) for camera, rgb, labels in targets]

        def factors():
            return torch.exp(grow.clamp(*np.log(SIZE_LIMITS))) if fit_scales else torch.ones_like(grow)

        def losses(camera, rgb, inside, outside):
            renderer.opacities, renderer.sh = torch.sigmoid(logit), torch.cat([base, rest], dim=1)
            renderer.scales = first_scales * factors()
            image, alpha = renderer._rasterize(camera)
            n_inside, n_outside = inside.sum().clamp_min(1), outside.sum().clamp_min(1)
            colour = ((image - rgb).abs().mean(-1) * inside).sum() / n_inside
            filled = ((1 - alpha) * inside).sum() / n_inside
            stray = (alpha * outside).sum() / n_outside
            return colour + filled + OUTSIDE * stray, 1 - filled

        def coverage():
            with torch.no_grad():
                return float(np.mean([float(losses(*view)[1]) for view in prepared]))

        try:
            before = coverage()
            groups = [{"params": [logit], "lr": OPACITY_RATE}, {"params": [base], "lr": COLOUR_RATE}]
            if fit_scales:
                groups.append({"params": [grow], "lr": SIZE_RATE})
            optimiser = torch.optim.Adam(groups)
            order = np.random.default_rng(seed)
            for step in range(steps):
                with torch.enable_grad():
                    loss, _ = losses(*prepared[int(order.integers(len(prepared)))])
                    optimiser.zero_grad()
                    loss.backward()
                optimiser.step()
                if on_step is not None:
                    on_step(step + 1, steps)
            after = coverage()
            opacities = torch.sigmoid(logit).detach().cpu().numpy().astype(np.float64)
            sh = torch.cat([base, rest], dim=1).detach().cpu().numpy()
            scale_factors = factors().detach().cpu().numpy()
        finally:
            renderer.opacities, renderer.sh, renderer.scales = original
    if not (np.isfinite(opacities).all() and np.isfinite(sh).all() and np.isfinite(scale_factors).all()):
        raise RuntimeError("refining produced invalid values")
    return {"opacities": opacities, "sh": sh, "scale_factors": scale_factors, "before": before, "after": after}
