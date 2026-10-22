import bpy

from .. import nif_node_preview, nif_switch
from ..nif_switch import LOD_NO_LIMIT


def poll_switch_child(self, ob):
    return ob.parent == self.id_data


def poll_lod_range_child(self, ob):
    return ob.parent == self.id_data  # self is the row; id_data is the owning Object


def update_node_type(self, context):
    # a new LOD starts with a range row per existing child, so every child is editable in the panel
    ob = self.id_data
    if nif_switch.is_lod(ob):
        nif_switch.ensure_lod_rows(ob, nif_switch.ordered_children(ob))


def update_lod_near(self, context):
    nif_switch.sync_lod_range(self, "near")


def update_lod_far(self, context):
    nif_switch.sync_lod_range(self, "far")


def update_switch_active_child(self, context):
    # an explicit pick supersedes an invalid selection kept from import
    if self.switch_invalid_index != -1:
        self.switch_invalid_index = -1
    nif_node_preview.refresh()  # a still view would otherwise wait for the full refresh


class LODRange(bpy.types.PropertyGroup):
    child: bpy.props.PointerProperty(
        name="Child",
        description="The direct child this range belongs to",
        type=bpy.types.Object,
        poll=poll_lod_range_child,
    )
    near: bpy.props.FloatProperty(
        name="Near",
        description="The child shows from this camera distance (inclusive)",
        min=0.0,
        max=LOD_NO_LIMIT,
        default=0.0,
        subtype="DISTANCE",
        update=update_lod_near,
    )
    far: bpy.props.FloatProperty(
        name="Far",
        description="The child shows up to this camera distance (exclusive). The maximum means no limit",
        min=0.0,
        max=LOD_NO_LIMIT,
        default=LOD_NO_LIMIT,
        subtype="DISTANCE",
        update=update_lod_far,
    )


class ObjectProperties(bpy.types.PropertyGroup):
    object_flags: bpy.props.IntProperty(name="Flags", min=0, max=65535, default=2)

    node_type: bpy.props.EnumProperty(
        name="Node Type",
        description="The NIF node type this Empty exports as",
        items=[
            ("NiNode", "NiNode", "An ordinary node", 0),
            ("NiSwitchNode", "NiSwitchNode", "At most one direct child is active; the author picks it", 1),
            ("NiLODNode", "NiLODNode", "At most one direct child is active, chosen by camera distance", 2),
            ("RootCollisionNode", "RootCollisionNode", "Its subtree is the collision shape and is never rendered", 3),
        ],
        default="NiNode",
        update=update_node_type,
    )

    switch_active_child: bpy.props.PointerProperty(
        name="Active Child",
        description="The switch's active child. None exports as -1, which shows no child",
        type=bpy.types.Object,
        poll=poll_switch_child,
        update=update_switch_active_child,
    )

    is_switch_null: bpy.props.BoolProperty(
        name="Empty Slot",
        description="This Empty exports as a null link in its parent switch's children",
        default=False,
    )

    switch_invalid_index: bpy.props.IntProperty(
        name="Invalid Imported Selection",
        description="An out-of-range selection kept from import for diagnosis. -1 when there is none",
        default=-1,
    )

    lod_center: bpy.props.FloatVectorProperty(
        name="LOD Center",
        description="The point distances are measured from, in this Empty's local space",
        size=3,
        default=(0.0, 0.0, 0.0),
        subtype="TRANSLATION",
    )

    lod_ranges: bpy.props.CollectionProperty(type=LODRange)

    lod_sync: bpy.props.BoolProperty(
        name="Sync Ranges",
        description="Editing a far also sets the next child's near, and editing a near sets the previous child's far",
        default=True,
    )

    @staticmethod
    def register() -> None:
        bpy.types.Object.mw = bpy.props.PointerProperty(type=ObjectProperties)
        bpy.types.PoseBone.mw = bpy.props.PointerProperty(type=ObjectProperties)

    @staticmethod
    def unregister() -> None:
        pass
