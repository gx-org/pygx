"""Test GX dynamic bindings."""

from google3.testing.pybase import googletest
from google3.third_party.gxlang.pygx import dyngx
from google3.third_party.gxlang.pygx.testing import pygxtesting
from google3.third_party.gxlang.pygx.tests.holder import holder


class PyImporterTest(googletest.TestCase):

  PKG_PATH = "github.com/gx-org/pygx/tests/pyimport"

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._runtime = pygxtesting.new_graph_runtime()
    self._dev = self._runtime.device(0)
    self._pyimport = dyngx.build_package(self._dev, self.PKG_PATH)
    self.assertIsNotNone(self._pyimport)

  def testSetValue(self):
    hld = holder.Holder()
    self.assertIsNone(hld.value)
    self._pyimport.SetValue(hld, 5)
    self.assertEqual(hld.value, 5)


if __name__ == "__main__":
  googletest.main()
