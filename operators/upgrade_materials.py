import bpy


class UpgradeMaterials(bpy.types.Operator):
    bl_idname = "material.mw_upgrade_materials"
    bl_options = {"REGISTER", "UNDO"}
    bl_label = "Upgrade Morrowind Materials"
    bl_description = "Update materials made by an older addon version. Opening a file does this automatically"

    def execute(self, context):
        from ..properties.material_properties import upgrade_materials

        upgrade_materials()
        return {"FINISHED"}
