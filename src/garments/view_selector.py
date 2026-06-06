def average_visibility(pose, keys):
    values = []

    for key in keys:
        if key in pose:
            values.append(pose[key]["vis"])

    if not values:
        return 0.0

    return sum(values) / len(values)


def select_view(pose, forced_view=None):
    """
    Returns one of:
    front, back, left, right

    V1 behavior:
    - If user forces a view, use it.
    - Otherwise default to front because photo booth testing is front-facing.
    - Back/side can be tested manually using keyboard controls.
    """

    if forced_view in ["front", "back", "left", "right"]:
        return forced_view

    return "front"