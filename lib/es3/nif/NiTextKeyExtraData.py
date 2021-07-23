from __future__ import annotations

from es3.utils.math import np, zeros
from .NiExtraData import NiExtraData

_dtype = np.dtype("<f, O")


class NiTextKeyExtraData(NiExtraData):
    keys: ndarray = zeros(0, dtype=_dtype)

    def load(self, stream):
        super().load(stream)
        num_text_keys = stream.read_uint()
        if num_text_keys:
            self.keys = zeros(num_text_keys, dtype=_dtype)
            for i in range(num_text_keys):
                self.keys[i] = stream.read_float(), stream.read_str()

    def save(self, stream):
        super().save(stream)
        stream.write_uint(len(self.keys))
        for time, value in self.keys.tolist():
            stream.write_float(time)
            stream.write_str(value)

    @property
    def times(self) -> ndarray:
        return self.keys["f0"]

    @times.setter
    def times(self, array: ndarray):
        self.keys["f0"] = array

    @property
    def values(self) -> ndarray:
        return self.keys["f1"]

    @values.setter
    def values(self, array: ndarray):
        self.keys["f1"] = array

    @staticmethod
    def _get_stop_text(start_text_lower):
        group_name = start_text_lower[:-6]  # trim " start"
        if group_name.endswith(("chop", "slash", "thrust")):
            return f"{group_name} small follow stop"
        elif group_name.endswith("shoot"):
            return f"{group_name} follow stop"
        else:
            return f"{group_name} stop"

    def get_animation_groups(self):
        group_stop = ""
        group_name = ""
        start_index = 0
        for i, text in enumerate(self.values.tolist()):
            for line in filter(None, text.splitlines()):
                line_lower = line.lower()

                if line_lower == group_stop:
                    yield (group_name, (start_index, i))
                    group_stop = ""
                    continue

                if group_stop != "":
                    # keep searching until we hit the group stop
                    continue

                if line_lower.endswith(" start"):
                    group_stop = self._get_stop_text(line_lower)
                    group_name = line[:-6]
                    start_index = i

    def expand_groups(self):
        temp = []
        seen = {}

        for time, text in self.keys.tolist():
            for line in filter(None, text.splitlines()):
                lowercased = line.lower()
                if (lowercased in seen) and ("sound" not in lowercased):
                    print(f"Skipped duplicate text key '{line}' at {time:.3f}. Previous at {seen[lowercased]:.3f}.")
                    continue

                seen[lowercased] = time
                temp.append((time, line))

        self.keys = np.array(temp, dtype=_dtype)

    def collapse_groups(self):
        uniques, inverse = np.unique(self.times, return_inverse=True)
        if len(uniques) == len(self.keys):
            return

        new_keys = np.empty(len(uniques), _dtype)

        for i, time in enumerate(uniques.tolist()):
            # list of all the strings for this timing
            # TODO: use hash map here for performance
            strings = self.values[inverse == i].tolist()
            # split strings to clean up extraneous newlines
            cleaned = [s for s in strings for s in s.splitlines() if s]
            # re-join the strings and update the keys array
            new_keys[i] = time, "\r\n".join(cleaned)

        self.keys = new_keys


if __name__ == "__main__":
    from es3.utils.typing import *
