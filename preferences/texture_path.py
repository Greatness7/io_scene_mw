import bpy


def addon_preferences(context):
    return context.preferences.addons[__package__].preferences


class TexturePath(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(
        subtype="DIR_PATH",
        name="",
        description="Texture Path",
        default="C:\\Morrowind\\Data Files\\Textures\\",
    )


class TexturePathList(bpy.types.UIList):
    bl_idname = "PREFERENCES_UL_MW_TexturePathList"

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        layout.prop(item, "name", emboss=False)


class TexturePathAdd(bpy.types.Operator):
    bl_idname = "preferences.mw_texture_paths_add"
    bl_options = {"REGISTER", "UNDO"}

    bl_label = "Add Texture Path"
    bl_description = "Add a new texture path"

    @classmethod
    def poll(cls, context):
        return context.space_data.type == "PREFERENCES"

    def execute(self, context):
        p = addon_preferences(context)
        p.texture_paths.add()
        p.texture_paths_active_index += 1
        p.texture_paths.move(len(p.texture_paths) - 1, p.texture_paths_active_index)
        return {"FINISHED"}


class TexturePathRemove(bpy.types.Operator):
    bl_idname = "preferences.mw_texture_paths_remove"
    bl_options = {"REGISTER", "UNDO"}

    bl_label = "Remove Texture Path"
    bl_description = "Remove the selected texture path"

    @classmethod
    def poll(cls, context):
        return context.space_data.type == "PREFERENCES"

    def execute(self, context):
        p = addon_preferences(context)
        p.texture_paths.remove(p.texture_paths_active_index)
        p.texture_paths_active_index = min(p.texture_paths_active_index, len(p.texture_paths) - 1)
        return {"FINISHED"}


class TexturePathMove(bpy.types.Operator):
    bl_idname = "preferences.mw_texture_paths_move"
    bl_options = {"REGISTER", "UNDO"}

    bl_label = "Move Texture Path"
    bl_description = "Move the selected texture path up or down"

    direction: bpy.props.IntProperty(default=1)

    @classmethod
    def poll(cls, context):
        return context.space_data.type == "PREFERENCES"

    def execute(self, context):
        p = addon_preferences(context)
        old_index = p.texture_paths_active_index
        new_index = old_index + self.direction
        if len(p.texture_paths) > new_index > -1:
            p.texture_paths.move(old_index, new_index)
            p.texture_paths_active_index = new_index
        return {"FINISHED"}
