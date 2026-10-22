from __future__ import annotations

import collections
from typing import Any, cast

import bpy


def get_fcurves(anim_data):
    """The F-curves of the assigned action, or none without animation data or an action (drivers alone)."""
    if bpy.app.version >= (5, 0, 0):
        from bpy_extras import anim_utils

        channelbag = cast(Any, anim_utils).animdata_get_channelbag_for_assigned_slot(anim_data)
        return () if channelbag is None else channelbag.fcurves
    action = anim_data and anim_data.action
    return () if action is None else action.fcurves


def fcurves_by_data_path(id_data):
    fcurves = collections.defaultdict(list)
    for fcurve in get_fcurves(id_data.animation_data):
        fcurves[fcurve.data_path].append(fcurve)
    return fcurves


def ensure_action(id_data):
    try:
        action = id_data.animation_data.action
    except AttributeError:
        action = bpy.data.actions.new(f"{id_data.name}Action")
        anim_data = id_data.animation_data_create()
        anim_data.action = action

        if bpy.app.version >= (4, 4, 0):
            anim_data.action_slot = action.slots.new(id_type=id_data.id_type, name=id_data.name)

    return action


def ensure_fcurve(action, data_path, index=0, group=""):
    if bpy.app.version >= (5, 0, 0):
        from bpy_extras import anim_utils

        channelbag = cast(Any, anim_utils).action_ensure_channelbag_for_slot(action, *action.slots)
        return channelbag.fcurves.ensure(data_path, index=index, group_name=group)
    return action.fcurves.new(data_path, index=index, action_group=group)


def action_fcurves(action):
    if bpy.app.version >= (5, 0, 0):
        from bpy_extras import anim_utils

        try:
            slot = action.slots[0]
        except IndexError:
            return ()
        channelbag = cast(Any, anim_utils).action_get_channelbag_for_slot(action, slot)
        return () if channelbag is None else channelbag.fcurves
    return action.fcurves


def set_fcurves_mute(fcurves, state):
    # Not foreach_set: it skips the per-curve RNA update, and Blender 3.6 then keeps ignoring curves after
    # they are unmuted, until the file is reopened (NOTES.md §6).
    for fcurve in fcurves:
        fcurve.mute = state
