import torch


def _scalar_triple_product(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return torch.bmm(a.view(-1, 1, 3), torch.cross(b, c).view(-1, 3, 1)).squeeze(2).squeeze(1)


def _compute_tetrahedron_barycentric_coordinates(a: torch.Tensor,
                                                 b: torch.Tensor,
                                                 c: torch.Tensor,
                                                 d: torch.Tensor,
                                                 p: torch.Tensor) -> torch.Tensor:
    vap = p - a
    vbp = p - b
    vab = b - a
    vac = c - a
    vad = d - a
    vbc = c - b
    vbd = d - b

    va6 = _scalar_triple_product(vbp, vbd, vbc)
    vb6 = _scalar_triple_product(vap, vac, vad)
    vc6 = _scalar_triple_product(vap, vad, vab)
    vd6 = _scalar_triple_product(vap, vab, vac)

    v6 = 1 / _scalar_triple_product(vab, vac, vad)
    return torch.stack([va6 * v6, vb6 * v6, vc6 * v6, vd6 * v6], dim=-1).to(p)


def apply_scene_flow_colormap(offsets: torch.Tensor) -> torch.Tensor:
    """
    Colorizes the given offsets as follows:
     -  0, 0, 0 -> white
     -  1, 0, 0 / 0, 1, 0 / 0, 0, 1 -> red/green/blue
     - -1, 0, 0 / 0,-1, 0 / 0, 0,-1 -> cyan/purple/orange

    This is done by subdividing the 3D xyz offset space into 8 tetrahedrons (each contains the origin and two axis points)
    The corner points of the tetrahedrons are the above-mentioned colors and the color interpolation is found via
    barycentric coordinates of the offset point within its containing tetrahedron

    Args:
        offsets: The offsets to colorize

    Taken from: https://stackoverflow.com/questions/38545520/barycentric-coordinates-of-a-tetrahedron

    Returns:
        colors for the given offsets
    """

    points = torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]], dtype=torch.float32).to(offsets)

    colors = torch.tensor([
        [1, 1, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ], dtype=torch.float32).to(offsets)

    original_shape = offsets.shape
    offsets = offsets.view(-1, 3)  # Ensure we just have a list of points

    # TODO: Need better way to choose containing/nearest tetrahedron. It can happen that chosen 4 points are not a
    #   Tetrahedron because 2 of them are opposite to each other!
    distances = (points[None, ...] - offsets[:, None, :]).norm(dim=2)  # [B, 7]
    idx_sorted = torch.sort(distances, dim=1).indices  # [B, 7]
    closest_4_points = points[idx_sorted[:, :4]]

    barycentric_coordinates = _compute_tetrahedron_barycentric_coordinates(closest_4_points[:, 0],
                                                                           closest_4_points[:, 1],
                                                                           closest_4_points[:, 2],
                                                                           closest_4_points[:, 3],
                                                                           offsets)

    barycentric_coordinates_clipped = torch.clip(barycentric_coordinates, min=0, max=1)
    barycentric_coordinates_clipped /= barycentric_coordinates_clipped.sum(dim=1).view(-1, 1)

    interpolated_colors = torch.bmm(barycentric_coordinates_clipped.view(-1, 1, 4), colors[idx_sorted[:, :4]]).squeeze(
        1)
    interpolated_colors = torch.clip(interpolated_colors, min=0, max=1)

    # It can happen that the barycentric coordinates are nan. This is the case if an offset is exactly 0.
    # Then it is not possible to decide for the correct tetrahedron by just the 4 nearest vertices.
    # Since these are rare edge cases, just use white color.
    interpolated_colors[barycentric_coordinates.isnan().any(dim=1)] = colors[0]
    interpolated_colors = interpolated_colors.view(original_shape)

    return interpolated_colors
