import torch

def rescale_to_unit_cube(
    points: torch.Tensor,
    mask: torch.Tensor | None = None,
    return_affine: bool = False,
    keep_aspect: bool = False
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    B, D, N = points.shape

    if mask is not None:
        mask_exp = mask.unsqueeze(1).expand(B, D, N)  # (B, D, N)
        min_candidates = points.masked_fill(~mask_exp, float('inf'))
        max_candidates = points.masked_fill(~mask_exp, float('-inf'))
        min_vals_per_axis = min_candidates.min(dim=2, keepdim=True).values  # (B, D, 1)
        max_vals_per_axis = max_candidates.max(dim=2, keepdim=True).values  # (B, D, 1)
    else:
        min_vals_per_axis = points.min(dim=2, keepdim=True).values
        max_vals_per_axis = points.max(dim=2, keepdim=True).values

    if keep_aspect:
        min_used = min_vals_per_axis.min(dim=1, keepdim=True).values  # (B, 1, 1)
        max_used = max_vals_per_axis.max(dim=1, keepdim=True).values  # (B, 1, 1)
    else:
        min_used = min_vals_per_axis                                 # (B, D, 1)
        max_used = max_vals_per_axis                                 # (B, D, 1)

    range_val = (max_used - min_used).clamp_min(1e-8)
    rescaled_points = (points - min_used) / range_val

    rescaled_points = rescaled_points.clamp(0.0, 1.0 - 1e-6)

    if return_affine:
        return rescaled_points, (min_used, max_used)
    return rescaled_points
