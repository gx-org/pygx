"""Tests showing how to evaluate GX code from Python."""

import numpy as np

from google3.testing.pybase import googletest
from google3.third_party.gxlang.pygx import dyngx
from google3.third_party.gxlang.pygx.testing import pygxtesting


class PyTraceTest(googletest.TestCase):

  PKG_PATH = "github.com/gx-org/pygx/tests/trace"

  def setUp(self):
    super().setUp()
    self._runtime = pygxtesting.new_graph_runtime()
    self._dev = self._runtime.device(0)
    self._trace = dyngx.build_package(self._dev, self.PKG_PATH)
    self.assertIsNotNone(self._trace)

  def testTraceArrayInt32(self):
    traced_from_callback = None

    def trace(*args):
      nonlocal traced_from_callback
      traced_from_callback = args[0].to_host()

    dyngx.set_trace_callback(trace)
    traced_array = np.array([1, 2], dtype=np.int32)
    returned_array = np.array([3, 1], dtype=np.int32)
    res = self._trace.TraceInt32(traced_array, returned_array)
    np.testing.assert_equal(
        traced_from_callback, traced_array, "array from trace does not match"
    )
    np.testing.assert_equal(
        res.to_host(), returned_array, "returned array does not match"
    )

  def testTraceArrayFloat64(self):
    traced_from_callback = None

    def trace(*args):
      nonlocal traced_from_callback
      traced_from_callback = args[0].to_host()

    dyngx.set_trace_callback(trace)
    traced_array = np.array([1.64, -2.56], dtype=np.float64)
    returned_array = np.array([3.1, -1.6], dtype=np.float64)
    res = self._trace.TraceFloat64(traced_array, returned_array)
    np.testing.assert_equal(
        traced_from_callback, traced_array, "array from trace does not match"
    )
    np.testing.assert_equal(
        res.to_host(), returned_array, "returned array does not match"
    )

  def testTraceStringInt(self):
    message_from_trace = None
    value_from_trace = None

    def trace(*args):
      nonlocal message_from_trace
      message_from_trace = args[0]
      nonlocal value_from_trace
      value_from_trace = args[1].to_host()

    dyngx.set_trace_callback(trace)
    traced_value = 42
    self._trace.TraceInt(traced_value)
    np.testing.assert_equal(message_from_trace, "hello")
    np.testing.assert_equal(value_from_trace, traced_value)


if __name__ == "__main__":
  googletest.main()
