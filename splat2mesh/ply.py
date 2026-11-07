"""Minimal PLY vertex I/O for 3D Gaussian splats (numpy only).

Reads the `vertex` element of binary (little or big endian) or ASCII PLY files with scalar properties, which is the
layout every 3DGS trainer and Idol write; other elements after `vertex` (faces) are ignored. Writes binary little
endian. Replaces the GPL-licensed `plyfile` package so the project depends only on permissive code.

    v = read_vertex("splat.ply")          # numpy structured array: v["x"], v["opacity"], v["f_dc_0"], ...
    write_vertex("out.ply", v)
"""
import numpy as np

_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2",
          "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
          "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}
_NAMES = {"i1": "char", "u1": "uchar", "i2": "short", "u2": "ushort", "i4": "int", "u4": "uint",
          "f4": "float", "f8": "double"}


def read_vertex(path) -> np.ndarray:
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path}: not a PLY file")
        fmt, count, props, element = None, 0, [], None
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: header has no end_header")
            w = line.decode("ascii", "replace").split()
            if not w or w[0] in ("comment", "obj_info"):
                continue
            if w[0] == "format":
                fmt = w[1]
            elif w[0] == "element":
                element = w[1]
                if element == "vertex":
                    count = int(w[2])
            elif w[0] == "property" and element == "vertex":
                if w[1] == "list":
                    raise ValueError(f"{path}: list property in vertex element is not supported")
                props.append((w[2], _TYPES[w[1]]))
            elif w[0] == "end_header":
                break
        if fmt == "ascii":
            rows = [f.readline().split() for _ in range(count)]
            out = np.empty(count, dtype=[(n, t) for n, t in props])
            for i, (n, t) in enumerate(props):
                out[n] = np.array([r[i] for r in rows], dtype=t)
            return out
        endian = "<" if fmt == "binary_little_endian" else ">"
        dt = np.dtype([(n, endian + t) for n, t in props])
        data = np.frombuffer(f.read(dt.itemsize * count), dtype=dt, count=count)
        return data.astype(dt.newbyteorder("=")) if endian == ">" else data.copy()


def write_vertex(path, v: np.ndarray) -> None:
    v = np.ascontiguousarray(v.astype([(n, "<" + v.dtype[n].str[1:]) for n in v.dtype.names]))
    head = ["ply", "format binary_little_endian 1.0", f"element vertex {len(v)}"]
    head += [f"property {_NAMES[v.dtype[n].str[1:]]} {n}" for n in v.dtype.names]
    with open(path, "wb") as f:
        f.write(("\n".join(head + ["end_header"]) + "\n").encode("ascii"))
        f.write(v.tobytes())
