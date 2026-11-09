"""Test GX dynamic bindings."""

import threading
import traceback
from unittest import mock

from google3.testing.pybase import googletest
from google3.third_party.gxlang.pygx import dyngx
from google3.third_party.gxlang.pygx.testing import pygxtesting
from google3.third_party.gxlang.pygx.tests.holder import holder

_ = pygxtesting  # Guarantee that GX test modules are in the binary.


class _FailingHolder:

  def set(self, val):
    raise ValueError(f"cannot set {val.to_host()}")


class PygxTest(googletest.TestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._runtime = pygxtesting.new_graph_runtime()
    self._dev = self._runtime.device(0)
    self._pyobjects = dyngx.build_package(self._dev, "pyobjects")
    self.assertIsNotNone(self._pyobjects)

  def testSame(self):
    ret = self._pyobjects.Same(self)
    self.assertEqual(self, ret)

  def testSetValue(self):
    hld = holder.Holder()
    self._pyobjects.SetValue(hld, 5)
    self.assertEqual(hld.value, 5)

  def testCallWithEmptyBytes(self):
    received = []
    self._pyobjects.CallWithEmptyBytes(received.append)
    self._pyobjects.CallWithNilBytes(received.append)
    self.assertEqual(received, [b"", b""])

  def testSetValueErrorContainsTraceback(self):
    with self.assertRaises(RuntimeError) as ctx:
      self._pyobjects.SetValue(_FailingHolder(), 5)
    msg = str(ctx.exception)
    self.assertIn("Traceback (most recent call last):", msg)
    self.assertIn("ValueError: cannot set 5", msg)

  def testSetValueErrorRecursionGuard(self):
    with mock.patch.object(
        traceback, "format_exception", side_effect=RuntimeError("format boom")
    ):
      with self.assertRaises(RuntimeError) as ctx:
        self._pyobjects.SetValue(_FailingHolder(), 5)
    msg = str(ctx.exception)
    self.assertIn("cannot set 5", msg)
    self.assertIn("exception processing error:", msg)
    self.assertIn("format boom", msg)

    dyngx.clear_all_cache()
    with self.assertRaises(RuntimeError) as ctx:
      self._pyobjects.SetValue(_FailingHolder(), 5)
    self.assertIn("Traceback (most recent call last):", str(ctx.exception))

  def testSetValueErrorPerThreadGuard(self):
    real_format_exception = traceback.format_exception
    other_thread_errors = []

    def format_with_other_thread(*args, **kwargs):
      if threading.current_thread() is threading.main_thread():

        def run_in_thread():
          dyngx.clear_all_cache()
          try:
            self._pyobjects.SetValue(_FailingHolder(), 7)
          except RuntimeError as e:
            other_thread_errors.append(str(e))

        t = threading.Thread(target=run_in_thread)
        t.start()
        t.join()
      return real_format_exception(*args, **kwargs)

    with mock.patch.object(
        traceback, "format_exception", side_effect=format_with_other_thread
    ):
      with self.assertRaises(RuntimeError) as ctx:
        self._pyobjects.SetValue(_FailingHolder(), 5)

    self.assertIn("Traceback (most recent call last):", str(ctx.exception))
    self.assertIn("ValueError: cannot set 5", str(ctx.exception))
    self.assertLen(other_thread_errors, 1)
    self.assertIn("Traceback (most recent call last):", other_thread_errors[0])
    self.assertIn("ValueError: cannot set 7", other_thread_errors[0])


if __name__ == "__main__":
  googletest.main()
