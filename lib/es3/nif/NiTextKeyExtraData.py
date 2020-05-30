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
    def times(self):
        return self.keys["f0"]

    @times.setter
    def times(self, array):
        self.keys["f0"] = array

    @property
    def values(self):
        return self.keys["f1"]

    @values.setter
    def values(self, array):
        self.keys["f1"] = array

    @staticmethod
    def _get_stop_text(start_text):
        group_name = start_text[:-6]  # trim " start" prefix
        if group_name.endswith(("chop", "slash", "thrust")):
            return f"{group_name} hit"
        elif group_name.endswith("shoot"):
            return f"{group_name} release"
        else:
            return f"{group_name} stop"

    def get_action_groups(self):
        start_index = 0
        end_text = None
        for i, text in enumerate(self.keys["f1"]):
            for line in text.lower().splitlines():

                if (end_text is None) and line.endswith(" start"):
                    start_index = i
                    end_text = self._get_stop_text(line)
                    continue

                if line == end_text:
                    yield (start_index, i)
                    end_text = None

    def expand_groups(self):
        temp = []
        seen = {}

        for time, text in self.keys:
            for line in filter(None, text.lower().splitlines()):
                if (line in seen) and ("sound" not in line):
                    print(f"Skipped duplicate text key '{line}' at {time:.3f}. Previous at {seen[line]:.3f}.")
                    continue

                seen[line] = time
                temp.append((time, line))

        self.keys = np.array(temp, dtype=_dtype)

    def collapse_groups(self):
        uniques, inverse = np.unique(self.keys["f0"], return_inverse=True)
        if len(uniques) == len(self.keys):
            return

        new_keys = self.keys.copy()
        new_keys.resize(len(uniques))

        for i, time in enumerate(uniques):
            # list of all the strings for this timing
            strings = self.keys[inverse == i]["f1"].tolist()  # TODO: use hash map here for performance!
            # split strings to clean up extraneous newlines
            sanitized = [s for s in strings for s in s.splitlines() if s]
            # re-join the strings and update the keys array
            new_keys[i] = time, "\r\n".join(sanitized)

        self.keys = new_keys


if __name__ == "__main__":
    from es3.utils.typing import *
