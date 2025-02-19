import bpy


class MarkersListCopy(bpy.types.Operator):
    bl_idname = "marker.mw_markers_copy"
    bl_options = {"REGISTER", "UNDO"}

    bl_label = "Copy Markers"
    bl_description = "Copy pose markers to the system clipboard"

    def execute(self, context):
        try:
            markers = context.active_object.animation_data.action.pose_markers
        except AttributeError:
            return {"CANCELLED"}

        bpy.context.window_manager.clipboard = "\n".join(f"{marker.frame} {marker.name}" for marker in markers)

        return {"FINISHED"}


class MarkersListPaste(bpy.types.Operator):
    bl_idname = "marker.mw_markers_paste"
    bl_options = {"REGISTER", "UNDO"}

    bl_label = "Paste Markers"
    bl_description = "Paste pose markers from the system clipboard"

    def execute(self, context):
        try:
            action = context.active_object.animation_data.action
        except AttributeError:
            action = context.active_object.animation_data_create().action = bpy.data.actions.new(
                f"{context.active_object.name}Action"
            )

        markers = action.pose_markers

        for marker in markers:
            markers.remove(marker)

        for line in bpy.context.window_manager.clipboard.split("\n"):
            try:
                frame, _, name = line.partition(" ")
                frame = int(frame)
            except (AttributeError, ValueError):
                print(f"Invalid marker format: {line}")
            else:
                markers.new(name).frame = frame

        return {"FINISHED"}


class MarkersListSort(bpy.types.Operator):
    bl_idname = "marker.mw_markers_sort"
    bl_options = {"REGISTER", "UNDO"}

    bl_label = "Sort Markers"
    bl_description = "Sort markers by their timings"

    def execute(self, context):
        try:
            markers = context.active_object.animation_data.action.pose_markers
        except AttributeError:
            pass
        else:
            temp = [(m.frame, m.name) for m in markers]
            temp.sort()
            for m, t in zip(markers, temp):
                m.frame, m.name = t
        return {"FINISHED"}


class MarkersListMenu(bpy.types.Menu):
    bl_idname = "DOPESHEET_MT_markers_menu"
    bl_label = "Markers Specials"

    def draw(self, _context):
        layout = self.layout
        layout.operator("marker.mw_markers_copy", icon="COPYDOWN")
        layout.operator("marker.mw_markers_paste", icon="PASTEDOWN")
        layout.separator()
        layout.operator("marker.mw_markers_sort", icon="SORTTIME")


class MarkersList(bpy.types.UIList):
    bl_idname = "DOPESHEET_UL_MW_MarkersList"

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.alignment = "LEFT"
        row.prop(item, "frame", text="", emboss=False)

        pose_marker_icon = "PMARKER_SEL" if item.select else "PMARKER"
        layout.prop(item, "name", text="", icon=pose_marker_icon, emboss=False)

    @staticmethod
    def get_selected(self):
        return self.pose_markers.active_index

    @staticmethod
    def set_selected(self, index):
        for m in self.pose_markers:
            m.select = False
        self.pose_markers.active_index = index
        self.pose_markers.active.select = True


class MarkersPanel(bpy.types.Panel):
    bl_idname = "DOPESHEET_PT_MW_MarkersPanel"
    bl_space_type = "DOPESHEET_EDITOR"
    bl_region_type = "UI"
    bl_label = "Morrowind"

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        sd = context.space_data
        return (ob is not None) and (sd.type == "DOPESHEET_EDITOR") and (sd.ui_mode == "ACTION")

    def draw(self, context):
        space_data = context.space_data
        try:
            action = context.active_object.animation_data.action
        except AttributeError:
            return

        self.layout.template_ID(space_data, "action", new="action.new", unlink="action.unlink")
        if action is None:
            return

        self.layout.prop(space_data, "show_pose_markers", text="Show Text Keys")

        # Markers List
        row = self.layout.row()
        row.template_list("DOPESHEET_UL_MW_MarkersList", "", action, "pose_markers", action, "active_pose_marker_index")
        row.enabled = space_data.show_pose_markers

        # Markers Operators
        col = row.column(align=True)
        col.operator("marker.add", icon="ADD", text="")
        col.operator("marker.delete", icon="REMOVE", text="")
        col.separator()
        col.menu("DOPESHEET_MT_markers_menu", icon="DOWNARROW_HLT", text="")

        col.enabled = space_data.show_pose_markers

    @staticmethod
    def register():
        bpy.types.Action.active_pose_marker_index = bpy.props.IntProperty(
            name="Active Pose Marker",
            get=MarkersList.get_selected,
            set=MarkersList.set_selected,
            options={"HIDDEN", "SKIP_SAVE"},
        )

    def unregister():
        del bpy.types.Action.active_pose_marker_index
