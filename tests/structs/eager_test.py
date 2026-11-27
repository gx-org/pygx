"""Tests composite struct values with the eager TensorFlow runtime.

Unlike structs_test.py (which uses the Go graph runtime), this test uses the
eager TensorFlow runtime to exercise structs with constant-initialized, nested,
and slice fields as return values, method receivers, and arguments.
"""

import numpy as np
import tensorflow as tf

from google3.testing.pybase import googletest
from google3.third_party.gxlang.pygx import cgx
from google3.third_party.gxlang.pygx import dyngx
from google3.third_party.gxlang.pygx.backends.tensorflow import gxtf


class EagerStructsTest(googletest.TestCase):

  PKG_PATH = "github.com/gx-org/pygx/tests/structs"

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._runtime = gxtf.new_runtime()
    self._dev = self._runtime.device(0)
    self._structs = dyngx.build_package(self._dev, self.PKG_PATH)
    self.assertIsNotNone(self._structs)

  def _array_value(self, values, dtype=np.int32):
    return cgx.Value.from_ndarray(self._dev, np.array(values, dtype=dtype))

  def _assert_type_error(self, fn, *args):
    with self.assertRaisesRegex(RuntimeError, "cannot use"):
      fn(*args)

  def testCompositeConstantsAndInputs(self):
    composite = self._structs.NewComposite(np.int32(5))
    self.assertIsInstance(composite, dyngx.Instance)
    np.testing.assert_array_equal(
        composite.Constant, np.array([1, 2], np.int32)
    )
    np.testing.assert_array_equal(composite.Dynamic, np.array([8, 9], np.int32))
    np.testing.assert_array_equal(
        composite.Nested.B, np.array([5, 6, 7], np.float32)
    )

    combined, nested = composite.Combine(np.int32(1))
    np.testing.assert_array_equal(combined, np.array([25, 27], np.int32))
    np.testing.assert_array_equal(nested, np.array([6, 7, 8], np.float32))

    total, nested_b, items_sum = self._structs.ReadComposite(composite)
    np.testing.assert_array_equal(total, np.array([9, 11], np.int32))
    np.testing.assert_array_equal(nested_b, np.array([5, 6, 7], np.float32))
    np.testing.assert_array_equal(items_sum, np.int32(15))

    composite.value.set_field("Dynamic", self._array_value([20, 30]))
    combined_updated, nested_updated = composite.Combine(np.int32(2))
    np.testing.assert_array_equal(
        combined_updated, np.array([38, 49], np.int32)
    )
    np.testing.assert_array_equal(
        nested_updated, np.array([7, 8, 9], np.float32)
    )
    total_updated, _, _ = self._structs.ReadComposite(composite)
    np.testing.assert_array_equal(total_updated, np.array([21, 32], np.int32))

    vec = self._structs.NewVec(np.int32(5))
    self.assertIsInstance(self._structs.PassVec(vec), dyngx.Instance)
    np.testing.assert_array_equal(
        self._structs.ReadItems(composite.value.get_field("Items")),
        np.int32(15),
    )
    b2 = self._structs.NewBox2()
    np.testing.assert_array_equal(
        self._structs.ReadBox(2, b2), np.array([6, 7], np.int32)
    )

  def testCompositeInputValidation(self):
    sa_bad = self._structs.New(np.int32(5))
    b2 = self._structs.NewBox2()
    sa_bad.value.set_field("B", b2.value)
    self._assert_type_error(self._structs.ReadA, sa_bad)
    self._assert_type_error(sa_bad.GetA)

    self._assert_type_error(self._structs.ReadComposite, b2)
    self._assert_type_error(
        self._structs.ReadComposite, tf.constant([1, 2], dtype=tf.int32)
    )
    self._assert_type_error(self._structs.PassVec, b2)

    composite = self._structs.NewComposite(np.int32(5))
    self._assert_type_error(
        self._structs.ReadFloats, composite.value.get_field("Items")
    )

    for bad_val in (
        self._array_value([20]),
        self._array_value([[20, 30]]),
        self._array_value([20, 30], np.float32),
    ):
      bad_composite = self._structs.NewComposite(np.int32(5))
      bad_composite.value.set_field("Dynamic", bad_val)
      self._assert_type_error(self._structs.ReadComposite, bad_composite)
      self._assert_type_error(bad_composite.Combine, np.int32(1))

    for bad_nested_b in (
        self._array_value([1.0, 2.0], np.float32),
        self._array_value([1, 2, 3]),
    ):
      bad_nested = self._structs.NewComposite(np.int32(5))
      bad_nested.Nested.value.set_field("B", bad_nested_b)
      self._assert_type_error(self._structs.ReadComposite, bad_nested)

    b2_bad = self._structs.NewBox2()
    b2_bad.value.set_field("A", self._array_value([1, 2, 3]))
    self._assert_type_error(self._structs.ReadBox, 2, b2_bad)

    for bad_items_val in (
        self._array_value([1, 2]),
        cgx.Value.from_object(tf.constant([20, 30], dtype=tf.float32)),
    ):
      bad_items = self._structs.NewComposite(np.int32(5))
      bad_items.value.set_field("Items", bad_items_val)
      self._assert_type_error(self._structs.ReadComposite, bad_items)


if __name__ == "__main__":
  googletest.main()
