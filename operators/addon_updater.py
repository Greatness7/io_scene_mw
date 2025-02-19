from pathlib import Path

import bpy

from ..preferences import Preferences

PATH = Path(__file__).parent.parent
assert PATH.name == "io_scene_mw"


class UpdateCheck(bpy.types.Operator):
    """Check if a new version is available on the plugin repository."""

    bl_idname = "preferences.mw_update_check"
    bl_options = {"REGISTER"}
    bl_label = "Check for updates"
    bl_description = "Requires internet connection"

    @classmethod
    def poll(cls, context):
        return context.space_data.type == "PREFERENCES"

    def execute(self, context):
        version, zipball_url = self.get_latest_version_info()

        if version > self.current_version:
            # require manual install on non-patch releases
            # cannot replace native libraries while loaded
            if version[:2] > self.current_version[:2]:
                Preferences.update_status = "UPDATE_FORBIDDEN"
                Preferences.update_url = zipball_url
            else:
                Preferences.update_status = "UPDATE_AVAILABLE"
                Preferences.update_url = zipball_url
        else:
            Preferences.update_status = "UPDATE_INSTALLED"
            Preferences.update_url = ""

        return {"FINISHED"}

    @staticmethod
    def get_latest_version_info():
        import json
        import ssl
        import urllib.request

        tags_url = "https://api.github.com/repos/Greatness7/io_scene_mw/tags"

        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(tags_url, context=ctx) as response:
            data = response.read()

        latest, *_ = json.loads(data)
        latest_zipball_url = latest["zipball_url"]
        latest_version_tag = latest["name"].split(".")
        latest_version = tuple(map(int, latest_version_tag))

        return latest_version, latest_zipball_url

    @property
    def current_version(self):
        from sys import modules

        return modules["io_scene_mw"].bl_info["version"]


class UpdateApply(bpy.types.Operator):
    """Download and install the latest version from the plugin repository."""

    bl_idname = "preferences.mw_update_apply"
    bl_options = {"REGISTER"}

    bl_label = "An update is available! Click to install"
    bl_description = "Requires internet connection"

    @classmethod
    def poll(cls, context):
        return context.space_data.type == "PREFERENCES"

    def execute(self, context):
        self.create_backup()
        self.install_files(Preferences.update_url)
        Preferences.update_status = "UPDATE_INSTALLED"
        Preferences.update_url = ""
        return {"FINISHED"}

    def create_backup(self):
        """Make a backup archive of our addon in the parent directory.
        e.g. scripts/addons/io_scene_mw.zip
        """
        import shutil

        shutil.make_archive(PATH, "zip", root_dir=PATH.parent, base_dir=PATH.name)

    def install_files(self, zipball_url):
        import io
        import ssl
        import urllib.request
        import zipfile as zf

        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(zipball_url, context=ctx) as response:
            zipball = zf.ZipFile(io.BytesIO(response.read()))

        root = Path(PATH.name)

        for info in zipball.infolist()[1:]:
            parts = Path(info.filename).parts[1:]
            relative_path = root.joinpath(*parts)
            if not relative_path.suffix:
                continue  # directories

            try:
                info.filename = str(relative_path)
                zipball.extract(info, PATH.parent)
            except PermissionError:
                print(f"PermissionError: cannot replace {info.filename}")


class UpdateNotes(bpy.types.Operator):
    bl_idname = "preferences.mw_update_notes"
    bl_options = {"REGISTER"}

    bl_label = "You are up to date"
    bl_description = "Click to open changelog"

    @classmethod
    def poll(cls, context):
        return context.space_data.type == "PREFERENCES"

    def execute(self, context):
        Preferences.update_status = "UPDATE_UNCHECKED"
        Preferences.update_url = ""
        bpy.ops.wm.url_open(url="https://github.com/Greatness7/io_scene_mw/releases")
        return {"FINISHED"}


class UpdateLimit(bpy.types.Operator):
    bl_idname = "preferences.mw_update_limit"
    bl_options = {"REGISTER"}

    bl_label = "An update is available, but requires manual installation"
    bl_description = "Click to open download page"

    @classmethod
    def poll(cls, context):
        return context.space_data.type == "PREFERENCES"

    def execute(self, context):
        bpy.ops.wm.url_open(url="https://blender-morrowind.readthedocs.io/en/latest/getting-started/downloading.html")
        return {"FINISHED"}
