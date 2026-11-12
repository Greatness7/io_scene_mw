import bpy

from .. import nif_node_preview, nif_switch
from ..nif_switch import LOD_NO_LIMIT


def poll_switch_child(self, ob):
    return ob.parent == self.id_data


def poll_lod_range_child(self, ob):
    return ob.parent == self.id_data  # self is the row; id_data is the owning Object


def update_node_type(self, context):
    # the type-specific bits are reinterpreted by the new type, so they restart from its default
    default, _views = nif_switch.NODE_TYPE_FLAGS[self.node_type]
    self.object_flags = (self.object_flags & ~nif_switch.TYPE_FLAGS_MASK) | default
    # a new LOD starts with a range row per existing child, so every child is editable in the panel
    ob = self.id_data
    if nif_switch.is_lod(ob):
        nif_switch.ensure_lod_rows(ob, nif_switch.ordered_children(ob))


def flag_bit(mask, name, description, forced=None):
    """A checkbox view of one bit of `object_flags`; `forced(self)` true means export sets the bit regardless."""

    def get(self):
        return (self.object_flags & mask) != 0 or (forced is not None and forced(self))

    def set_(self, value):
        self.object_flags = (self.object_flags | mask) if value else (self.object_flags & ~mask)

    return bpy.props.BoolProperty(name=name, description=description, get=get, set=set_, options=set())


def flag_field(mask, shift, name, description, items):
    """A dropdown view of a multi-bit field of `object_flags`; item numbers are the field's values."""

    def get(self):
        return (self.object_flags & mask) >> shift

    def set_(self, value):
        self.object_flags = (self.object_flags & ~mask) | (value << shift)

    return bpy.props.EnumProperty(name=name, description=description, items=items, get=get, set=set_, options=set())


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
    node_type: bpy.props.EnumProperty(
        name="Node Type",
        description="The NIF node type this Empty exports as",
        items=[
            ("AvoidNode", "AvoidNode", "Actors steer around its subtree instead of stopping at it", 5),
            ("NiBillboardNode", "NiBillboardNode", "Turns toward the camera", 6),
            ("NiBSAnimationNode", "NiBSAnimationNode", "The engine updates the animations below it", 7),
            ("NiBSParticleNode", "NiBSParticleNode", "An animation node for particle systems", 8),
            ("NiCollisionSwitch", "NiCollisionSwitch", "Its Collision flag lets collision and picking through", 9),
            ("NiFltAnimationNode", "NiFltAnimationNode", "Shows each direct child in turn, over its period", 4),
            ("NiLODNode", "NiLODNode", "At most one direct child is active, chosen by camera distance", 2),
            ("NiNode", "NiNode", "An ordinary node", 0),
            ("NiSortAdjustNode", "NiSortAdjustNode", "Changes how the alpha geometry below it is sorted", 10),
            ("NiSwitchNode", "NiSwitchNode", "At most one direct child is active; the author picks it", 1),
            ("RootCollisionNode", "RootCollisionNode", "Its subtree is the collision shape and is never rendered", 3),
        ],
        default="NiNode",
        update=update_node_type,
    )

    mesh_node_type: bpy.props.EnumProperty(
        name="Node Type",
        description="A mesh always exports as NiTriShape",
        items=[("NiTriShape", "NiTriShape", "A mesh always exports as NiTriShape", 0)],
        get=lambda self: 0,
        options=set(),
    )

    object_flags: bpy.props.IntProperty(
        name="Flags",
        description="The raw NIF flags. The labeled flags above edit bits of this value",
        min=0,
        max=65535,
        default=2,
    )

    flag_app_culled: flag_bit(
        0x0001,
        "App Culled",
        "Never drawn. Collision still applies. Collision roots and shadows are always app culled",
        forced=lambda self: self.node_type == nif_switch.COLLISION,
    )

    flag_collision_mode: flag_field(
        0x0006,
        1,
        "Collision Mode",
        "How NetImmerse collision tests descend here. Morrowind's actor and projectile collision ignores it",
        [
            ("NONE", "None", "Stop here", 0),
            ("TRIANGLES", "Triangles", "Test children down to their triangles", 1),
            ("OBBS", "Bounding Boxes", "Test the bounding volume first, children only on a hit", 2),
            ("CONTINUE", "Continue", "Descend to child nodes without testing geometry here", 3),
        ],
    )

    # NiSwitchNode
    flag_update_only_active: flag_bit(
        0x0020, "Update Only Active", "Only the active child's controllers are updated and bounded"
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

    # NiFltAnimationNode
    flag_bounce: flag_bit(0x0040, "Bounce", "Cycle back and forth through the children instead of looping")

    flt_period: bpy.props.FloatProperty(
        name="Period",
        description="Frames to show every child once",
        min=0.0,
        default=0.0,
    )

    # NiLODNode
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

    # NiBSAnimationNode
    flag_animated: flag_bit(
        0x0020, "Animated", "The engine updates the controllers below. Without it animations and particles stay still"
    )

    flag_not_random: flag_bit(0x0040, "Not Random", "Controllers start in phase instead of at a random time")

    # NiBSParticleNode
    flag_follow: flag_bit(
        0x0080, "Follow Parent", "The subtree follows its parent. Off anchors it in world space where it spawned"
    )

    # NiCollisionSwitch
    flag_collision: flag_bit(0x0020, "Collision", "Collision and picking reach the children. Off blocks both")

    # NiBillboardNode
    billboard_mode: flag_field(
        0x0060,
        5,
        "Billboard Mode",
        "How the node turns toward the camera",
        [
            ("ALWAYS_FACE_CAMERA", "Always Face Camera", "Turn fully toward the camera", 0),
            ("ROTATE_ABOUT_UP", "Rotate About Up", "Turn only about the local up axis", 1),
            ("RIGID_FACE_CAMERA", "Rigid Face Camera", "Face the camera without rolling", 2),
            # Morrowind has no mode 3: the value matches no branch and the node does not turn
        ],
    )

    # NiSortAdjustNode
    sorting_mode: bpy.props.EnumProperty(
        name="Sorting",
        description="How the alpha geometry below is sorted",
        items=[
            ("INHERIT", "Inherit", "Sort as the parent does", 0),
            ("OFF", "Off", "No sorting; draw in scene order", 1),
            ("SUBSORT", "Subsort", "Sort this subtree on its own", 2),
            ("GROUPED", "Grouped (MWSE)", "Sort the subtree as one object. Without MWSE this acts as Inherit", 64),
        ],
        default="OFF",
    )

    @staticmethod
    def register() -> None:
        bpy.types.Object.mw = bpy.props.PointerProperty(type=ObjectProperties)
        bpy.types.PoseBone.mw = bpy.props.PointerProperty(type=ObjectProperties)

    @staticmethod
    def unregister() -> None:
        pass
