"""NumPy-only Graphdeco Gaussian PLY input/output.

Reads ASCII and binary little-endian scalar vertex properties. Writes binary
little-endian PLY with float32 Gaussian parameters, preserving scalar extras
and stable ``gaussian_id`` values. This is a semantic, not byte-exact roundtrip:
raw rotations are normalized and activated opacity endpoints are clipped to
float64 epsilon before conversion to finite logits.
"""

from pathlib import Path

import numpy as np

from .scene import GaussianScene, SUPPORTED_SH_COUNTS


_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
    "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
}
_BASE_NAMES = ["x", "y", "z", "opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)] + [f"f_dc_{i}" for i in range(3)]


def _read_header(stream):
    if stream.readline().strip() != b"ply":
        raise ValueError("not a PLY file: expected 'ply' header")
    fmt = None
    vertex_count = None
    current_element = None
    properties = []
    while True:
        line = stream.readline()
        if not line:
            raise ValueError("incomplete PLY header")
        try:
            parts = line.decode("ascii").strip().split()
        except UnicodeDecodeError as exc:
            raise ValueError("PLY header must be ASCII") from exc
        if not parts or parts[0] in ("comment", "obj_info"):
            continue
        if parts[0] == "end_header":
            if len(parts) != 1:
                raise ValueError("invalid end_header declaration")
            break
        if parts[0] == "format":
            if len(parts) != 3 or parts[2] != "1.0" or fmt is not None:
                raise ValueError("invalid PLY format declaration")
            fmt = parts[1]
            if fmt not in ("ascii", "binary_little_endian"):
                raise ValueError("only ASCII and binary little-endian PLY are supported")
        elif parts[0] == "element":
            if len(parts) != 3:
                raise ValueError("invalid PLY element declaration")
            try:
                count = int(parts[2])
            except ValueError as exc:
                raise ValueError("invalid PLY element count") from exc
            if count < 0:
                raise ValueError("PLY element counts cannot be negative")
            current_element = parts[1]
            if current_element == "vertex":
                if vertex_count is not None:
                    raise ValueError("duplicate vertex element")
                vertex_count = count
            elif count:
                raise ValueError(f"unsupported nonempty PLY element {current_element!r}; meshes are not Gaussian scenes")
        elif parts[0] == "property":
            if current_element is None:
                raise ValueError("PLY property declared before element")
            if current_element != "vertex":
                continue
            if len(parts) > 1 and parts[1] == "list":
                raise ValueError("list-valued vertex properties are unsupported")
            if len(parts) != 3 or parts[1] not in _PLY_TYPES:
                raise ValueError("unsupported scalar PLY vertex property")
            if parts[2] in {name for name, _ in properties}:
                raise ValueError(f"duplicate vertex property {parts[2]!r}")
            properties.append((parts[2], _PLY_TYPES[parts[1]]))
        else:
            raise ValueError(f"unsupported PLY header directive {parts[0]!r}")
    if fmt is None or vertex_count is None:
        raise ValueError("PLY requires format and vertex declarations")
    return fmt, vertex_count, properties


def _read_vertices(stream, fmt, count, properties):
    dtype = np.dtype(properties)
    if fmt == "binary_little_endian":
        size = count * dtype.itemsize
        data = stream.read(size)
        if len(data) != size:
            raise ValueError("truncated binary PLY vertex data")
        vertices = np.frombuffer(data, dtype=dtype).copy()
        if stream.read().strip():
            raise ValueError("unexpected data after PLY vertices")
        return vertices
    try:
        tokens = stream.read().decode("ascii").split()
    except UnicodeDecodeError as exc:
        raise ValueError("ASCII PLY contains non-ASCII data") from exc
    if len(tokens) != count * len(properties):
        raise ValueError("ASCII PLY vertex data does not match declared size")
    vertices = np.empty(count, dtype=dtype)
    width = len(properties)
    for column, (name, type_name) in enumerate(properties):
        scalar_dtype = np.dtype(type_name)
        values = tokens[column::width]
        try:
            if scalar_dtype.kind in "iu":
                numbers = [int(value) for value in values]
                limits = np.iinfo(scalar_dtype)
                if any(value < limits.min or value > limits.max for value in numbers):
                    raise ValueError("integer value out of declared type range")
                vertices[name] = numbers
            else:
                with np.errstate(over="raise", invalid="raise"):
                    vertices[name] = np.asarray(values, dtype=scalar_dtype)
        except (ValueError, OverflowError, FloatingPointError) as exc:
            raise ValueError(f"invalid ASCII PLY values for property {name!r}") from exc
    return vertices


def _sigmoid(values):
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_negative = np.exp(values[~positive])
    result[~positive] = exp_negative / (1.0 + exp_negative)
    return result


def load_ply(path) -> GaussianScene:
    """Load activated Gaussian parameters, rejecting ordinary point clouds."""
    with Path(path).open("rb") as stream:
        fmt, count, properties = _read_header(stream)
        names = {name for name, _ in properties}
        missing = sorted(set(_BASE_NAMES) - names)
        if missing:
            raise ValueError("not a Graphdeco Gaussian PLY; missing properties: " + ", ".join(missing))
        rest_names = [name for name in names if name.startswith("f_rest_")]
        if any(not name[7:].isdigit() for name in rest_names):
            raise ValueError("SH f_rest properties require numeric suffixes")
        rest_names.sort(key=lambda name: int(name[7:]))
        if rest_names != [f"f_rest_{i}" for i in range(len(rest_names))]:
            raise ValueError("SH f_rest properties must be contiguous from f_rest_0")
        if len(rest_names) % 3 or 1 + len(rest_names) // 3 not in SUPPORTED_SH_COUNTS:
            raise ValueError("unsupported SH coefficient count; expected degrees 0 through 3")
        reserved = set(_BASE_NAMES) | set(rest_names) | {"gaussian_id"}
        malformed = [name for name in names - reserved
                     if name.startswith(("f_dc_", "scale_", "rot_"))]
        if malformed:
            raise ValueError(f"unexpected Gaussian parameter fields: {sorted(malformed)}")
        vertices = _read_vertices(stream, fmt, count, properties)

    def columns(prefix, size):
        return np.column_stack([vertices[f"{prefix}{i}"] for i in range(size)]).astype(np.float64)

    means = np.column_stack([vertices[name] for name in ("x", "y", "z")]).astype(np.float64)
    with np.errstate(over="ignore", under="ignore"):
        scales = np.exp(columns("scale_", 3))
    quaternions = columns("rot_", 4)
    # Scaling before norm also handles otherwise overflowing, finite raw values.
    largest = np.max(np.abs(quaternions), axis=1)
    if np.any(~np.isfinite(largest)) or np.any(largest == 0):
        raise ValueError("raw PLY rotations must be finite nonzero quaternions")
    quaternions = quaternions / largest[:, None]
    quaternions /= np.linalg.norm(quaternions, axis=1)[:, None]
    opacities = _sigmoid(vertices["opacity"].astype(np.float64))
    sh_count = 1 + len(rest_names) // 3
    sh = np.empty((count, sh_count, 3), dtype=np.float64)
    sh[:, 0, :] = columns("f_dc_", 3)
    if rest_names:
        # Graphdeco stores all red coefficients, then green, then blue.
        raw_rest = np.column_stack([vertices[name] for name in rest_names])
        sh[:, 1:, :] = raw_rest.reshape(count, 3, sh_count - 1).transpose(0, 2, 1)
    ids = None
    if "gaussian_id" in names:
        values = vertices["gaussian_id"]
        if values.dtype.kind not in "iu":
            if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
                raise ValueError("gaussian_id must contain integral values")
            if np.any(values < -(2.0**63)) or np.any(values >= 2.0**63):
                raise ValueError("gaussian_id values exceed signed 64-bit range")
        ids = values.astype(np.int64)
    extras = {name: vertices[name].copy() for name in names - reserved}
    return GaussianScene(means, scales, quaternions, opacities, sh, ids, extras)


def _storage_type(array):
    kind, size = array.dtype.kind, array.dtype.itemsize
    if kind == "b":
        return "uchar", np.dtype("u1")
    if kind in "iu":
        if size <= 1:
            return ("char", np.dtype("i1")) if kind == "i" else ("uchar", np.dtype("u1"))
        if size <= 2:
            return ("short", np.dtype("<i2")) if kind == "i" else ("ushort", np.dtype("<u2"))
        minimum = int(array.min()) if array.size else 0
        maximum = int(array.max()) if array.size else 0
        if kind == "u" and 0 <= minimum and maximum <= 2**32 - 1:
            return "uint", np.dtype("<u4")
        if -(2**31) <= minimum and maximum <= 2**31 - 1:
            return "int", np.dtype("<i4")
        if 0 <= minimum and maximum <= 2**32 - 1:
            return "uint", np.dtype("<u4")
        raise ValueError("portable PLY integer attributes must fit int32 or uint32")
    if kind == "f":
        if size > 8:
            raise ValueError("portable PLY floating attributes support at most float64")
        return ("float", np.dtype("<f4")) if size <= 4 else ("double", np.dtype("<f8"))
    raise ValueError(f"unsupported PLY attribute dtype {array.dtype}")


def save_ply(scene: GaussianScene, path) -> None:
    """Save a binary Graphdeco PLY, including all SH and scalar extra fields.

    Gaussian arrays use float32 on disk; extra floating arrays retain float64
    when applicable. Stable IDs use standard int32/uint32 PLY scalar storage.
    """
    # Revalidate mutable arrays before opening/truncating an existing output.
    scene = scene.copy()
    fields = {name: scene.means[:, i] for i, name in enumerate(("x", "y", "z"))}
    fields.update({f"f_dc_{i}": scene.sh[:, 0, i] for i in range(3)})
    raw_rest = scene.sh[:, 1:, :].transpose(0, 2, 1).reshape(len(scene), 3 * (scene.sh.shape[1] - 1))
    fields.update({f"f_rest_{i}": raw_rest[:, i] for i in range(raw_rest.shape[1])})
    eps = np.finfo(np.float64).eps
    opacity = np.clip(scene.opacities, eps, 1.0 - eps)
    fields["opacity"] = np.log(opacity) - np.log1p(-opacity)
    fields.update({f"scale_{i}": np.log(scene.scales[:, i]) for i in range(3)})
    fields.update({f"rot_{i}": scene.quaternions[:, i] for i in range(4)})
    properties = [(name, "float", np.dtype("<f4")) for name in fields]
    fields["gaussian_id"] = scene.ids
    type_name, dtype = _storage_type(scene.ids)
    properties.append(("gaussian_id", type_name, dtype))
    for name, values in scene.extras.items():
        if name in fields or name.startswith(("f_rest_", "f_dc_", "scale_", "rot_")):
            raise ValueError(f"extra attribute {name!r} conflicts with a Gaussian parameter")
        fields[name] = values
        type_name, dtype = _storage_type(values)
        properties.append((name, type_name, dtype))
    vertices = np.empty(len(scene), dtype=[(name, dtype) for name, _, dtype in properties])
    for name, _, _ in properties:
        with np.errstate(over="ignore", invalid="ignore"):
            vertices[name] = fields[name]
        if vertices[name].dtype.kind == "f" and not np.all(np.isfinite(vertices[name])):
            raise ValueError(f"property {name!r} is outside its PLY storage range")
    header = ["ply", "format binary_little_endian 1.0",
              "comment gscutter Gaussian scene; quaternion order wxyz",
              f"element vertex {len(scene)}"]
    header.extend(f"property {type_name} {name}" for name, type_name, _ in properties)
    header.append("end_header")
    with Path(path).open("wb") as stream:
        stream.write(("\n".join(header) + "\n").encode("ascii"))
        stream.write(vertices.tobytes())
