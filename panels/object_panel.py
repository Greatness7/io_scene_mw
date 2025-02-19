import bpy


class ObjectPanel(bpy.types.Panel):
    bl_idname = "OBJECT_PT_MW_ObjectPanel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "object"
    bl_label = "Morrowind"

    def draw(self, context):
        ob = context.active_object
        self.layout.prop(ob.mw, "object_flags")
