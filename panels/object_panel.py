import bpy

from .. import nif_node_preview, nif_switch


def get_switch_node(context):
    ob = context.active_object
    return ob if nif_switch.is_switch_node(ob) else None


def get_switch(context):
    ob = context.active_object
    return ob if nif_switch.is_switch(ob) else None


def get_lod(context):
    ob = context.active_object
    return ob if nif_switch.is_lod(ob) else None


class SwitchNodeSetActive(bpy.types.Operator):
    """Make this child the switch's active child"""

    bl_idname = "object.mw_switch_node_set_active"
    bl_label = "Set Active Child"
    bl_options = {"REGISTER", "UNDO", "INTERNAL"}

    child: bpy.props.StringProperty(name="Child", description="Name of the child to select; empty selects None")

    @classmethod
    def poll(cls, context):
        return get_switch_node(context) is not None

    def execute(self, context):
        switch = get_switch_node(context)
        if self.child:
            child = bpy.data.objects.get(self.child)
            if child is None or child.parent != switch:
                self.report({"ERROR"}, f"'{self.child}' is not a direct child of '{switch.name}'")
                return {"CANCELLED"}
        else:
            child = None
        switch.mw.switch_active_child = child
        switch.mw.switch_invalid_index = -1
        return {"FINISHED"}


class SwitchAddNullSlot(bpy.types.Operator):
    """Append an empty slot, which exports as a null link"""

    bl_idname = "object.mw_switch_add_null_slot"
    bl_label = "Add Empty Slot"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and get_switch(context) is not None

    def execute(self, context):
        switch = get_switch(context)
        # link beside the last variant so the slot appends in collection order
        children = nif_switch.ordered_children(switch, nif_switch.scene_objects(context.scene))
        collection = (children[-1] if children else switch).users_collection[0]

        marker = bpy.data.objects.new(nif_switch.NULL_SLOT_NAME, None)
        marker.mw.is_switch_null = True
        marker.empty_display_size = switch.empty_display_size * 0.5
        collection.objects.link(marker)
        marker.parent = switch
        if nif_switch.is_lod(switch):
            nif_switch.ensure_lod_rows(switch, [marker])
        return {"FINISHED"}


class SwitchRemoveNullSlot(bpy.types.Operator):
    """Delete this empty slot"""

    bl_idname = "object.mw_switch_remove_null_slot"
    bl_label = "Remove Empty Slot"
    bl_options = {"REGISTER", "UNDO", "INTERNAL"}

    child: bpy.props.StringProperty(name="Child")

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and get_switch(context) is not None

    def execute(self, context):
        switch = get_switch(context)
        marker = bpy.data.objects.get(self.child)
        if marker is None or marker.parent != switch or not nif_switch.is_null_slot(marker):
            self.report({"ERROR"}, f"'{self.child}' is not an empty slot of '{switch.name}'")
            return {"CANCELLED"}
        bpy.data.objects.remove(marker)
        return {"FINISHED"}


class LODAddRange(bpy.types.Operator):
    """Give this child its own distance range"""

    bl_idname = "object.mw_lod_add_range"
    bl_label = "Add Range"
    bl_options = {"REGISTER", "UNDO", "INTERNAL"}

    child: bpy.props.StringProperty(name="Child")

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and get_lod(context) is not None

    def execute(self, context):
        lod = get_lod(context)
        child = bpy.data.objects.get(self.child)
        if child is None or child.parent != lod:
            self.report({"ERROR"}, f"'{self.child}' is not a direct child of '{lod.name}'")
            return {"CANCELLED"}
        lod.mw.lod_ranges.add().child = child
        return {"FINISHED"}


class LODRemoveRange(bpy.types.Operator):
    """Remove this range"""

    bl_idname = "object.mw_lod_remove_range"
    bl_label = "Remove Range"
    bl_options = {"REGISTER", "UNDO", "INTERNAL"}

    index: bpy.props.IntProperty(name="Index")

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and get_lod(context) is not None

    def execute(self, context):
        lod = get_lod(context)
        if not 0 <= self.index < len(lod.mw.lod_ranges):
            self.report({"ERROR"}, f"'{lod.name}' has no range {self.index}")
            return {"CANCELLED"}
        lod.mw.lod_ranges.remove(self.index)
        return {"FINISHED"}


def get_lod_range(context, index):
    lod = get_lod(context)
    if 0 <= index < len(lod.mw.lod_ranges):
        return lod.mw.lod_ranges[index]
    return None


class LODToggleNoLimit(bpy.types.Operator):
    """Toggle whether this child shows at any distance beyond near (far = FLT_MAX)"""

    bl_idname = "object.mw_lod_toggle_no_limit"
    bl_label = "No Limit (FLT_MAX)"
    bl_options = {"REGISTER", "UNDO", "INTERNAL"}

    index: bpy.props.IntProperty(name="Index")

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and get_lod(context) is not None

    def execute(self, context):
        row = get_lod_range(context, self.index)
        if row is None:
            self.report({"ERROR"}, f"'{context.active_object.name}' has no range {self.index}")
            return {"CANCELLED"}
        if not nif_switch.is_no_limit(row.far):
            row.far = nif_switch.LOD_NO_LIMIT
            return {"FINISHED"}
        # a finite far meets the next level when synced; otherwise it starts at near, an empty range the
        # author then widens, flagged until they do
        lod = get_lod(context)
        after = nif_switch.lod_neighbor(lod, row, 1) if lod.mw.lod_sync else None
        row.far = after.near if after is not None and after.near > row.near else row.near
        return {"FINISHED"}


class LODSortByDistance(bpy.types.Operator):
    """Reorder the children by near, then far, which changes the export order"""

    bl_idname = "object.mw_lod_sort_by_distance"
    bl_label = "Sort by Distance"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and get_lod(context) is not None

    def execute(self, context):
        lod = get_lod(context)
        objects = nif_switch.scene_objects(context.scene)
        children = nif_switch.ordered_children(lod, objects)
        ranges = dict(zip(children, nif_switch.lod_ranges(lod, children)))
        wanted = sorted(children, key=ranges.__getitem__)  # stable: equal ranges keep their order
        # relinking appends to each collection, so relinking in sorted order sorts within every collection
        for child in wanted:
            for collection in child.users_collection:
                collection.objects.unlink(child)
                collection.objects.link(child)
        if nif_switch.ordered_children(lod, objects) != wanted:
            self.report({"WARNING"}, f"'{lod.name}' has children in several collections; each collection is sorted on its own")
        return {"FINISHED"}


class OBJECT_MT_mw_lod_range(bpy.types.Menu):
    bl_label = "LOD Range"

    def draw(self, context):
        row = getattr(context, "mw_lod_range", None)
        lod = get_lod(context)
        if lod is None:
            return
        if row is None:
            child = getattr(context, "mw_lod_child", None)
            if child is not None:
                self.layout.operator(LODAddRange.bl_idname, text="Add Range", icon="ADD").child = child.name
            return
        index = next(i for i, r in enumerate(lod.mw.lod_ranges) if r.as_pointer() == row.as_pointer())
        layout = self.layout
        unlimited = nif_switch.is_no_limit(row.far)
        icon = "CHECKBOX_HLT" if unlimited else "CHECKBOX_DEHLT"
        layout.operator(LODToggleNoLimit.bl_idname, icon=icon).index = index


class ObjectPanel(bpy.types.Panel):
    bl_idname = "OBJECT_PT_MW_ObjectPanel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "object"
    bl_label = "Morrowind"

    def draw(self, context):
        ob = context.active_object
        if ob is None:
            return
        node_type = ob.mw.node_type if ob.type == "EMPTY" else nif_switch.NODE
        if ob.type == "EMPTY":
            self.layout.prop(ob.mw, "node_type")
        elif ob.type == "MESH":
            # the same row as an Empty's, locked: a mesh always exports as NiTriShape
            row = self.layout.row()
            row.enabled = False
            row.prop(ob.mw, "mesh_node_type")
        self.draw_flags(ob, node_type)

        if node_type == nif_node_preview.BILLBOARD:
            self.draw_options(context, self.layout)
        if nif_switch.is_switch_node(ob):
            self.draw_switch_node(context, ob)
        elif nif_switch.is_lod(ob):
            self.draw_lod(context, ob)
        elif nif_switch.is_null_slot(ob):
            for problem in nif_switch.null_slot_problems(ob, nif_switch.scene_objects(context.scene)):
                self.layout.label(text=problem, icon="ERROR")

    def draw_flags(self, ob, node_type):
        # label-left property split, as in the Transform panel
        column = self.layout.column(heading="Flags")
        column.use_property_split = True
        column.use_property_decorate = False
        row = column.row()
        row.enabled = node_type != nif_switch.COLLISION  # export always hides collision roots
        row.prop(ob.mw, "flag_hidden")
        column.prop(ob.mw, "flag_collision_mode")
        for name in nif_switch.NODE_TYPE_FLAGS[node_type][1]:
            column.prop(ob.mw, name)
        if node_type == nif_switch.FLT_ANIMATION:
            column.prop(ob.mw, "flt_period")
        elif node_type == "NiSortAdjustNode":
            column.prop(ob.mw, "sorting_mode")

    def draw_switch_node(self, context, switch):
        children = nif_switch.ordered_children(switch, nif_switch.scene_objects(context.scene))
        active = nif_switch.active_index(switch, children)

        box = self.layout.box()
        box.label(text="Switch Children")
        self.draw_options(context, box)
        for problem in nif_switch.switch_node_problems(switch, children):
            box.label(text=problem, icon="ERROR")

        column = box.column(align=True)
        for index, child in enumerate(children):
            is_null = nif_switch.is_null_slot(child)
            row = column.row(align=True)
            op = row.operator(
                SwitchNodeSetActive.bl_idname,
                text=f"{index}: {child.name}",
                icon="RADIOBUT_ON" if index == active else "RADIOBUT_OFF",
                depress=index == active,
            )
            op.child = child.name
            if is_null:
                row.label(text="", icon="GHOST_DISABLED")
                row.operator(SwitchRemoveNullSlot.bl_idname, text="", icon="X").child = child.name
        row = column.row(align=True)
        op = row.operator(
            SwitchNodeSetActive.bl_idname,
            text="None (-1)",
            icon="RADIOBUT_ON" if active == -1 else "RADIOBUT_OFF",
            depress=active == -1,
        )
        op.child = ""

        target = switch.mw.switch_active_child
        if target is not None and active == -1:
            box.label(text=f"'{target.name}' is no longer a child; exports as None", icon="INFO")

        box.operator(SwitchAddNullSlot.bl_idname, icon="ADD")

    @staticmethod
    def draw_options(context, box):
        """The Options block both switch types open with, so changing type keeps the top of the box stable."""
        # label-left property split, as in the Transform panel
        options = box.column(heading="Options")
        options.use_property_split = True
        options.use_property_decorate = False
        options.prop(context.window_manager, "mw_node_preview")
        return options

    def draw_lod(self, context, lod):
        children = nif_switch.ordered_children(lod, nif_switch.scene_objects(context.scene))
        rows = {}
        for row in lod.mw.lod_ranges:
            if row.child is not None and row.child in children:
                rows.setdefault(row.child, row)

        box = self.layout.box()
        header = box.row()
        header.label(text="LOD Levels")
        header.operator(LODSortByDistance.bl_idname, text="", icon="SORTSIZE")
        self.draw_options(context, box).prop(lod.mw, "lod_sync")
        for problem in nif_switch.lod_problems(lod, children):
            box.label(text=problem, icon="ERROR")
        for warning in nif_switch.lod_warnings(lod, children):
            box.label(text=warning, icon="INFO")

        # label-left property split, as in the Transform panel
        column = box.column()
        column.use_property_split = True
        column.use_property_decorate = False
        column.prop(lod.mw, "lod_center")
        wm = context.window_manager
        shown = nif_node_preview.active_child_name(lod) if wm.mw_node_preview else None

        for index, child in enumerate(children):
            split = box.split(factor=0.25)
            left = split.column(align=True)
            left.label(text=f"Child {index}", icon="HIDE_OFF" if child.name == shown else "BLANK1")
            right = split.column(align=True)

            line = right.row(align=True)
            name = line.row(align=True)
            name.enabled = False  # read-only: ranges follow their child, so the child is not picked here
            name.prop(child, "name", text="", icon_value=name.icon(child))
            if nif_switch.is_null_slot(child):
                line.label(text="", icon="GHOST_DISABLED")
                line.operator(SwitchRemoveNullSlot.bl_idname, text="", icon="X").child = child.name

            line = right.row(align=True)
            line.context_pointer_set("mw_lod_child", child)
            row = rows.get(child)
            if row is None:
                # parented after the LOD was set up: the default range, editable once the menu adds a row
                fallback = line.row(align=True)
                fallback.enabled = False
                fallback.operator(LODAddRange.bl_idname, text="Near: 0").child = child.name
                fallback.operator(LODAddRange.bl_idname, text="FLT_MAX").child = child.name
            else:
                line.context_pointer_set("mw_lod_range", row)
                line.prop(row, "near")
                if nif_switch.is_no_limit(row.far):
                    far = line.row(align=True)
                    far.enabled = False  # set and cleared only through the menu
                    far.operator(LODToggleNoLimit.bl_idname, text="FLT_MAX").index = -1
                else:
                    line.prop(row, "far")
            line.menu(OBJECT_MT_mw_lod_range.__name__, text="", icon="DOWNARROW_HLT")

        unused = nif_switch.unused_lod_ranges(lod, children)
        if unused:
            box.label(text="Unused ranges")
            column = box.column(align=True)
            for index in unused:
                row = lod.mw.lod_ranges[index]
                line = column.row(align=True)
                line.label(text=row.child.name if row.child else "(no child)")
                line.label(text=f"{row.near:g} to {row.far:g}")
                line.operator(LODRemoveRange.bl_idname, text="", icon="X").index = index

        box.operator(SwitchAddNullSlot.bl_idname, icon="ADD")


class ObjectRawFlagsPanel(bpy.types.Panel):
    bl_idname = "OBJECT_PT_MW_ObjectRawFlagsPanel"
    bl_parent_id = ObjectPanel.bl_idname
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "object"
    bl_label = "Raw Flags"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        ob = context.active_object
        if ob is not None:
            self.layout.prop(ob.mw, "object_flags")
