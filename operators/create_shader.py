import bpy


class CreateShader(bpy.types.Operator):
    bl_idname = "material.mw_create_shader"
    bl_options = {"REGISTER", "UNDO"}
    bl_label = "Create Morrowind Shader"
    bl_description = "Convert to (or create) a Morrowind shader"

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return (ob is not None) and (ob.type == "MESH")

    def execute(self, context):
        from .. import nif_shader

        nif_shader.execute(context.active_object)
        return {"FINISHED"}
