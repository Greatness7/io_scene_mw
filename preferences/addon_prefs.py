import bpy

from .texture_path import TexturePath


class Preferences(bpy.types.AddonPreferences):
    bl_idname = "io_scene_mw"

    update_status = "UPDATE_UNCHECKED"
    update_url = ""

    scale_correction: bpy.props.FloatProperty(
        name="Scale Correction",
        description="Adjust the scale of imported/exported objects by a specified multiplier",
        min=0.01,
        max=10.0,
        default=0.01,
    )

    texture_paths: bpy.props.CollectionProperty(type=TexturePath)

    texture_paths_active_index: bpy.props.IntProperty(name="Select Item", options={"SKIP_SAVE"})

    def draw(self, context):
        layout = self.layout

        layout.operator(self.get_update_operator(), icon="URL")
        layout.separator()

        layout.label(text="Import/Export:")
        layout.prop(self, "scale_correction")
        layout.separator()

        layout.label(text="Texture Paths:")
        row = layout.row(align=True)
        row.template_list(
            "PREFERENCES_UL_MW_TexturePathList", "", self, "texture_paths", self, "texture_paths_active_index"
        )

        col = row.column(align=True)
        col.operator("preferences.mw_texture_paths_add", icon="ADD", text="")
        col.operator("preferences.mw_texture_paths_remove", icon="REMOVE", text="")
        col.separator()
        col.operator("preferences.mw_texture_paths_move", icon="TRIA_UP", text="").direction = -1
        col.operator("preferences.mw_texture_paths_move", icon="TRIA_DOWN", text="").direction = 1

    @classmethod
    def get_update_operator(cls):
        if cls.update_status == "UPDATE_UNCHECKED":
            return "preferences.mw_update_check"
        if cls.update_status == "UPDATE_AVAILABLE":
            return "preferences.mw_update_apply"
        if cls.update_status == "UPDATE_INSTALLED":
            return "preferences.mw_update_notes"
        if cls.update_status == "UPDATE_FORBIDDEN":
            return "preferences.mw_update_limit"

    @classmethod
    def from_context(cls, context):
        return context.preferences.addons[cls.bl_idname].preferences
