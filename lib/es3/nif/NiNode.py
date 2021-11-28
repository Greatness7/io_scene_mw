from __future__ import annotations

from es3.utils.math import ID44, la, np
from .NiAVObject import NiAVObject


def _sort_children_key(child):
    name = getattr(child, "name", "").lower()
    if name.startswith("tri "):
        prefix, suffix = name.rsplit(" ", 1)
        if suffix.isnumeric():
            return prefix, True, int(suffix)
    return name, False, 0


class NiNode(NiAVObject):
    children: list[NiAVObject | None] = []
    effects: list[NiDynamicEffect | None] = []

    _refs = (*NiAVObject._refs, "children", "effects")

    def load(self, stream):
        super().load(stream)
        self.children = stream.read_links()
        self.effects = stream.read_links()

    def save(self, stream):
        super().save(stream)
        stream.write_links(self.children)
        stream.write_links(self.effects)

    def sort(self, key=_sort_children_key):
        """Sort the NiNode's children list.

        If no key is provided children will be sorted by their name, with
        special handling for the "Tri (...) N" naming convention, where N
        is interpreted as an integer rather than a string.
        """
        super().sort()
        self.children.sort(key=key)

    def skinned_meshes(self):
        for mesh in self.descendants():
            if getattr(mesh, "skin", None):
                yield mesh

    def apply_skins(self, keep_skins=False):
        for mesh in self.skinned_meshes():
            mesh.apply_skin(keep_skins)

    def calc_bone_bind_poses(self):
        """ TODO
            handle bones without defined bind poses
            e.g. any Bip01 stuffs unused in rigging
        """
        temps = {}
        binds = {}

        # collect bind positions
        for mesh in self.skinned_meshes():
            matrix = mesh.matrix_relative_to(self)
            temps[mesh] = {bone: matrix @ la.inv(data.matrix) for bone, data in mesh.bone_influences}

        # bind position priority
        def sorter(items):
            node, data = items
            # give shadow meshes preference as they tend to encompass the entire skeleton
            return int(node.is_shadow), len(data)  # else bones with more users gets prio

        # process bind positions
        for mesh, data in sorted(temps.items(), key=sorter, reverse=True):

            # compare with previously seen bones
            for bone in data.keys() & binds.keys():
                diff = la.solve(data[bone], binds[bone])

                # do the bind positions coincide
                if not np.allclose(diff, ID44, rtol=0, atol=0.001):
                    print(f"MISMATCHED BINDPOSES: {bone} vs {mesh}")
                    # print(diff)
                    break
            else:
                binds.update(data)

        # use hierarchical order
        return {bone: binds[bone] for bone in self.descendants() if bone in binds}

    def apply_bone_bind_poses(self, lock_children=False):
        bind_poses = self.calc_bone_bind_poses()

        for bone, bind in bind_poses.items():
            pose = bone.matrix_relative_to(self)

            # calculate difference
            diff = la.solve(pose, bind)

            if not np.allclose(diff, ID44, rtol=0, atol=1e-6):

                # update bind pose
                bone.matrix = bone.matrix @ diff

                # correct children
                if lock_children:
                    diff_inv = la.inv(diff)
                    for child in bone.children:
                        child.matrix = child.matrix @ diff_inv


if __name__ == "__main__":
    from es3.nif import NiDynamicEffect
    from es3.utils.typing import *
