"""Node type authoring rules shared by the object panel and both translators.

`mw.node_type` names the NIF class an Empty exports as. Only the switch family has rules beyond the type test.
A switch-family node is an Empty whose `mw.node_type` is NiSwitchNode or NiLODNode. Its variants are its direct children, ordered by
the scene's collection-link sequence (never `Object.children`, which is alphabetical). An Empty child
flagged `mw.is_switch_null` stands for an intentional null link. The selection is an object reference,
so it follows its target through renames and reordering; the numeric index exists only in the file.
A LOD's distance ranges are rows on the LOD Empty, each owned by its child through an object reference.
"""

import bpy

NODE, SWITCH_NODE, LOD, COLLISION = "NiNode", "NiSwitchNode", "NiLODNode", "RootCollisionNode"

NULL_SLOT_NAME = "Empty Slot"

LOD_NO_LIMIT = 3.4028234663852886e38  # float32 max: the far of a level that has no limit
LOD_NO_LIMIT_THRESHOLD = 3.4e38  # a far at or above this is treated as no limit


class SwitchError(Exception):
    """An export precondition failed; nothing was written."""


def _node_type(ob):
    # pose bones share `mw` but have no `type`; skeletal switches are deferred
    return ob.mw.node_type if getattr(ob, "type", None) == "EMPTY" else None


def is_switch_node(ob):
    """An author-selected NiSwitchNode."""
    return _node_type(ob) == SWITCH_NODE


def is_lod(ob):
    """A distance-selected NiLODNode."""
    return _node_type(ob) == LOD


def is_switch(ob):
    """A switch in the NIF sense: NiSwitchNode or its subclass NiLODNode."""
    return _node_type(ob) in (SWITCH_NODE, LOD)


def is_collision(ob):
    """An author-selected RootCollisionNode."""
    return _node_type(ob) == COLLISION


def is_null_slot(ob):
    return ob.mw.is_switch_null


def scene_objects(scene=None):
    """The unfiltered scene sequence that defines switch order for the panel and the exporter."""
    return (scene or bpy.context.scene).collection.all_objects


def ordered_children(switch, objects=None):
    """Direct children of `switch` in collection-link order, including null-slot markers."""
    if objects is None:
        objects = scene_objects()
    seen = set()
    children = []
    for ob in objects:
        if ob.parent == switch and ob not in seen:
            seen.add(ob)
            children.append(ob)
    return children


def active_index(switch, children):
    """The index of the selected child in `children`, or -1 when nothing currently valid is selected."""
    target = switch.mw.switch_active_child
    if target is None:
        return -1
    try:
        return children.index(target)
    except ValueError:
        return -1  # the target was reparented away: inactive, not an error


def switch_node_problems(switch, children):
    """Why `switch` cannot be exported as-is. Empty when it can."""
    problems = []
    if switch.instance_type != "NONE":
        problems.append(f"switch '{switch.name}' instances a collection, which a switch cannot do")
    invalid = switch.mw.switch_invalid_index
    if switch.mw.switch_active_child is None and invalid != -1:
        problems.append(
            f"switch '{switch.name}' was imported with invalid selection {invalid} (of {len(children)} children);"
            " pick a child or None in the object panel"
        )
    return problems


def null_slot_problems(marker, objects=None):
    """Why `marker` cannot stand for a null link. Empty when it can."""
    if objects is None:
        objects = scene_objects()
    name = marker.name
    if marker.type != "EMPTY":
        return [f"empty slot '{name}' must be an Empty"]
    problems = []
    if not is_switch(marker.parent):
        problems.append(f"empty slot '{name}' is not a direct child of a NiSwitchNode or NiLODNode")
    if any(ob.parent == marker for ob in objects):
        problems.append(f"empty slot '{name}' has children, which a null link cannot keep")
    if marker.instance_type != "NONE":
        problems.append(f"empty slot '{name}' instances a collection")
    if marker.animation_data and marker.animation_data.action:
        problems.append(f"empty slot '{name}' is animated")
    return problems


def is_no_limit(far):
    return far >= LOD_NO_LIMIT_THRESHOLD


def _lod_rows(lod, children):
    """The first row per direct child in `children`, keyed by child."""
    rows = {}
    for row in lod.mw.lod_ranges:
        if row.child is not None and row.child in children:
            rows.setdefault(row.child, row)
    return rows


def ensure_lod_rows(lod, children):
    """Give every child in `children` a range row, with the default range where one is missing.

    Returns child -> row, looked up after adding: adding to a collection can invalidate held row references.
    """
    have = _lod_rows(lod, children)
    for child in children:
        if child not in have:
            lod.mw.lod_ranges.add().child = child
    return _lod_rows(lod, children)


def lod_ranges(lod, children):
    """One `(near, far)` per child in `children`; a child without a row gets `[0, no limit]`."""
    rows = _lod_rows(lod, children)
    ranges = []
    for child in children:
        row = rows.get(child)
        if row is None:
            ranges.append((0.0, LOD_NO_LIMIT))
        else:
            far = LOD_NO_LIMIT if is_no_limit(row.far) else row.far
            ranges.append((row.near, far))
    return ranges


def lod_problems(lod, children):
    """Why `lod` cannot be exported as-is. Empty when it can."""
    problems = []
    if lod.instance_type != "NONE":
        problems.append(f"NiLODNode '{lod.name}' instances a collection, which a NiLODNode cannot do")
    counts = {}
    for row in lod.mw.lod_ranges:
        if row.child is not None and row.child in children:
            counts[row.child] = counts.get(row.child, 0) + 1
    for child, count in counts.items():
        if count > 1:
            problems.append(f"NiLODNode '{lod.name}' has {count} ranges for '{child.name}'; remove the extras")
    return problems


def lod_warnings(lod, children):
    """Exportable but probably unintended ranges."""
    rows = _lod_rows(lod, children)
    return [
        f"NiLODNode '{lod.name}': '{child.name}' has near {row.near:g} >= far {row.far:g} and can never show"
        for child, row in rows.items()
        if row.near >= row.far  # near <= d < far is empty
    ]


def unused_lod_ranges(lod, children):
    """Indices of rows export does not use: the child is gone or no longer a direct child, or an earlier row
    already belongs to it. A duplicate is also an export error, so the panel must offer to remove it."""
    used = {row.as_pointer() for row in _lod_rows(lod, children).values()}
    return [index for index, row in enumerate(lod.mw.lod_ranges) if row.as_pointer() not in used]


def lod_neighbor(lod, row, offset):
    """The row of the child `offset` places from `row`'s child in export order, or None."""
    children = ordered_children(lod)
    rows = _lod_rows(lod, children)
    if row.child is None or row.child not in rows or rows[row.child].as_pointer() != row.as_pointer():
        return None  # unused rows have no neighbors
    index = children.index(row.child) + offset
    return rows.get(children[index]) if 0 <= index < len(children) else None


_syncing = False


def sync_lod_range(row, field):
    """Keep a range valid after `field` was edited: the edited end clamps its own pair, and with `lod_sync` on
    each moved end also moves the neighbor it meets (far -> the next near, near -> the previous far)."""
    global _syncing
    lod = row.id_data
    if _syncing or not is_lod(lod):
        return
    _syncing = True  # the writes below must not re-enter this through their own updates
    try:
        near_moved = field == "near"
        far_moved = field == "far"
        if far_moved and row.far < row.near:
            row.near, near_moved = row.far, True
        elif near_moved and row.near > row.far:
            row.far, far_moved = row.near, True
        if not lod.mw.lod_sync:
            return
        before = lod_neighbor(lod, row, -1) if near_moved else None
        if before is not None and before.far != row.near:
            before.far = row.near
        after = lod_neighbor(lod, row, 1) if far_moved and not is_no_limit(row.far) else None
        if after is not None and after.near != row.far:
            after.near = row.far  # a no-limit level has no successor boundary
    finally:
        _syncing = False


def levels_chained(levels):
    """Whether each level's far is the next level's near: the convention `lod_sync` maintains."""
    return all(levels[i][1] == levels[i + 1][0] for i in range(len(levels) - 1))
