"""Holder provides a class with a value that can be set."""

from google3.third_party.gxlang.pygx import cgx


class Holder:
  """Holder holds a value."""

  def __init__(self):
    self.value = None

  def set(self, val):
    self.value = val.to_host()


def set_value(h: Holder, val: cgx.Int):
  h.set(val)
