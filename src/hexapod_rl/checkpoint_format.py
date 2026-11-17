"""Version 3 wire format: JSON metadata and a separate torch tensor table.

The codec itself is dependency-independent. The caller supplies torch.Tensor
as tensor_type; tests can exercise metadata without importing torch or Gym.
No metadata tag imports a class or invokes a function supplied by a file.
"""

from collections import OrderedDict
import json


FORMAT_VERSION = 3


def encode_checkpoint(state, tensor_type):
    tensors = []

    def encode(value):
        if isinstance(value, tensor_type):
            tensors.append(value)
            return ["tensor", len(tensors) - 1]
        if value is None or type(value) in (bool, int, str):
            return ["literal", value]
        if type(value) is float:
            # Exact finite float round trips, including signed zero; also inf/nan.
            return ["float", value.hex()]
        if type(value) in (dict, OrderedDict):
            pairs = [[encode(k), encode(v)] for k, v in value.items()]
            if type(value) is OrderedDict:
                if set(vars(value)) - {"_metadata"}:
                    raise TypeError("Unsupported checkpoint OrderedDict attributes")
                metadata = [encode(value._metadata)] if hasattr(value, "_metadata") else []
                return ["ordered_dict", [pairs, metadata]]
            return ["dict", pairs]
        if type(value) in (tuple, list):
            return ["tuple" if type(value) is tuple else "list", [encode(v) for v in value]]
        raise TypeError(f"Unsupported checkpoint metadata type: {type(value).__name__}")

    return {"format_version": FORMAT_VERSION,
            "metadata": json.dumps(encode(state), ensure_ascii=True, allow_nan=False),
            "tensors": tensors}


def decode_checkpoint(wire, tensor_type):
    if (type(wire) is not dict or set(wire) != {"format_version", "metadata", "tensors"}
            or type(wire["format_version"]) is not int or wire["format_version"] != FORMAT_VERSION
            or type(wire["metadata"]) is not str or type(wire["tensors"]) is not list
            or any(not isinstance(t, tensor_type) for t in wire["tensors"])):
        raise ValueError("Invalid version 3 checkpoint envelope")

    def decode(node):
        if type(node) is not list or len(node) != 2 or type(node[0]) is not str:
            raise ValueError("Invalid checkpoint metadata node")
        kind, value = node
        if kind == "literal" and (value is None or type(value) in (bool, int, str)):
            return value
        if kind == "float" and type(value) is str:
            return float.fromhex(value)
        if kind == "tensor" and type(value) is int and 0 <= value < len(wire["tensors"]):
            return wire["tensors"][value]
        if kind in ("tuple", "list") and type(value) is list:
            items = [decode(v) for v in value]
            return tuple(items) if kind == "tuple" else items
        if kind in ("dict", "ordered_dict") and type(value) is list:
            metadata = []
            if kind == "ordered_dict":
                if len(value) != 2 or type(value[1]) is not list or len(value[1]) > 1:
                    raise ValueError("Invalid OrderedDict checkpoint metadata")
                value, metadata = value
            if type(value) is not list:
                raise ValueError("Invalid checkpoint dictionary")
            result = OrderedDict() if kind == "ordered_dict" else {}
            for pair in value:
                if type(pair) is not list or len(pair) != 2:
                    raise ValueError("Invalid checkpoint dictionary entry")
                key, item = decode(pair[0]), decode(pair[1])
                if key in result:
                    raise ValueError("Duplicate checkpoint dictionary key")
                result[key] = item
            if metadata:
                result._metadata = decode(metadata[0])
            return result
        raise ValueError(f"Invalid checkpoint metadata tag or value: {kind}")

    try:
        state = decode(json.loads(wire["metadata"]))
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise ValueError("Invalid version 3 checkpoint metadata") from error
    if type(state) is not dict or type(state.get("format_version")) is not int or state["format_version"] != FORMAT_VERSION:
        raise ValueError("Invalid decoded checkpoint version")
    return state
