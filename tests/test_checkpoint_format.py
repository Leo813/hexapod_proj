"""Metadata format tests without PyTorch, CUDA or Isaac Gym."""

from collections import OrderedDict
import json
import math
from pathlib import Path
import pickle
import pickletools
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hexapod_rl.checkpoint_format import FORMAT_VERSION, decode_checkpoint, encode_checkpoint


class OpaqueTensor:
    """Only tests the codec's external tensor references, not tensor I/O."""


class CheckpointFormatTests(unittest.TestCase):
    def state(self, value):
        return {"format_version": FORMAT_VERSION, "value": value}

    def roundtrip(self, value):
        return decode_checkpoint(encode_checkpoint(self.state(value), OpaqueTensor), OpaqueTensor)["value"]

    def test_all_metadata_scalar_types_and_float_precision(self):
        value = [None, True, False, "unicode \u516d\u8db3", 0, -1, 2 ** 100,
                 0.1, -0., 5e-324, float("inf"), float("-inf"), float("nan")]
        restored = self.roundtrip(value)
        for old, new in zip(value, restored):
            self.assertIs(type(new), type(old))
            if type(old) is float:
                self.assertEqual(new.hex(), old.hex())
            else:
                self.assertEqual(new, old)
        self.assertEqual(math.copysign(1., restored[8]), -1.)

    def test_container_types_integer_keys_and_state_dict_metadata(self):
        value = OrderedDict([(7, ("a", [1., False])), ("tuple", ()), ("mapping", {})])
        value._metadata = OrderedDict([("", {"version": 1})])
        restored = self.roundtrip(value)
        self.assertIs(type(restored), OrderedDict)
        self.assertEqual(list(restored), list(value))
        self.assertEqual(restored, value)
        self.assertIs(type(restored[7]), tuple)
        self.assertIs(type(restored[7][1]), list)
        self.assertIs(type(restored._metadata), OrderedDict)
        self.assertEqual(restored._metadata, value._metadata)

    def test_tensor_references_are_separate_from_json_metadata(self):
        tensor = OpaqueTensor()
        wire = encode_checkpoint(self.state({"weight": tensor, "again": tensor}), OpaqueTensor)
        self.assertEqual(set(wire), {"format_version", "metadata", "tensors"})
        restored = decode_checkpoint(wire, OpaqueTensor)["value"]
        self.assertIs(restored["weight"], tensor)
        self.assertIs(restored["again"], tensor)
        self.assertNotIn("OpaqueTensor", wire["metadata"])

    def test_metadata_pickle_excludes_float_and_object_reconstruction_opcodes(self):
        # Demonstrate the original cause independently of an installed torch.
        raw_ops = {op.name for op, _, _ in pickletools.genops(pickle.dumps({"lr": .0003}, protocol=2))}
        self.assertIn("BINFLOAT", raw_ops)
        wire = encode_checkpoint(self.state({"lr": .0003, "limits": (float("inf"), -0.),
                                             "large_integer": 2 ** 100}), OpaqueTensor)
        ops = {op.name for op, _, _ in pickletools.genops(pickle.dumps(wire, protocol=2))}
        self.assertFalse(ops & {"BINFLOAT", "FLOAT", "GLOBAL", "REDUCE", "BUILD", "FRAME", "MEMOIZE"})
        self.assertEqual(decode_checkpoint(wire, OpaqueTensor)["value"]["lr"], .0003)

    def test_malformed_envelopes_tags_and_references_are_rejected(self):
        wire = encode_checkpoint(self.state(None), OpaqueTensor)
        for key, bad in (("format_version", 2), ("metadata", b"{}"), ("tensors", [object()])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                decode_checkpoint(dict(wire, **{key: bad}), OpaqueTensor)
        for node in (["global", "os.system"], ["tensor", -1], ["tensor", True], ["tensor", 0],
                     ["literal", 1.5], ["float", "invalid"], ["ordered_dict", [[], [1, 2]]],
                     ["dict", [[["literal", "x"], ["literal", 1]],
                               [["literal", "x"], ["literal", 2]]]],
                     ["literal", None]):
            with self.subTest(node=node), self.assertRaises(ValueError):
                decode_checkpoint(dict(wire, metadata=json.dumps(node)), OpaqueTensor)
        with self.assertRaises(ValueError):
            decode_checkpoint(dict(wire, metadata="not json"), OpaqueTensor)

    def test_unsupported_python_objects_are_rejected_at_encoding(self):
        custom = OrderedDict()
        custom.extra = "not a state_dict attribute"
        for value in (object(), b"bytes", {1, 2}, Path("file"), custom):
            with self.subTest(value=type(value).__name__), self.assertRaises(TypeError):
                encode_checkpoint(self.state(value), OpaqueTensor)
