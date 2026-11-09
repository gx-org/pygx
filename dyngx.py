"""GX Python API for dynamic bindings."""

from __future__ import annotations

from google3.third_party.gxlang.pygx import cgx

Runtime = cgx.Runtime
static_builder = cgx.static_builder
set_trace_callback = cgx.set_trace_callback
Error = cgx.Error

super_hot = False
_device_to_cache = {}


def default_runtime() -> Runtime:
  rtm = cgx.default_runtime
  if rtm is None:
    raise RuntimeError("GX default runtime not initialized")
  return rtm


class _Cache:
  """GX Device and its cache."""

  def __init__(self, dev: cgx.Device):
    self.__device = dev
    self.__path_to_package = {}
    self.__cache = {}

  def reset_for(self, instance):
    if instance not in self.__cache:
      return
    del self.__cache[instance]

  def get(self, callee):
    instance = callee.__self__
    callee_result = self.__cache.get(instance, None)
    if callee_result is not None:
      return callee_result
    callee_result = callee()
    self.__cache[instance] = callee_result
    return callee_result

  def clear(self):
    self.__cache = {}

  def package(self, path: str) -> Package:
    pkg = self.__path_to_package.get(path, None)
    if pkg is not None:
      return pkg
    pkg = Package(self, path)
    self.__path_to_package[path] = pkg
    return pkg

  @property
  def device(self) -> cgx.Device:
    return self.__device


class Package:
  """GX Package."""

  def __init__(self, device_cache: _Cache, path: str):
    self.__cache = device_cache
    self.__path = path
    self.__names_to_ir = {}
    self.__statics = set()

  @property
  def path(self) -> str:
    return self.__path

  @property
  def cache(self) -> _Cache:
    return self.__cache

  @property
  def device(self) -> cgx.Device:
    return self.__cache.device

  def update_static_variable(self, vr: StaticVar):
    self.__cache.reset_for(self)
    if vr.value is None:
      self.__statics.remove(vr)
    else:
      self.__statics.add(vr)

  def __build_names_to_ir(self, pkg: cgx.Package):
    """Build a dyngx class for all GX elements of the package."""
    names_to_ir = {}

    for static_var in pkg.list_static_vars():
      name = static_var.name()
      dynvar = self.__names_to_ir.get(name, None)
      names_to_ir[name] = dynvar if dynvar else StaticVar(self, static_var)

    for func in pkg.list_functions():
      name = func.name()
      dynfunc = self.__names_to_ir.get(name, None)
      names_to_ir[name] = dynfunc if dynfunc else Function(self, func)

    for iface in pkg.list_interfaces():
      name = iface.name()
      dyniface = self.__names_to_ir.get(name, None)
      names_to_ir[name] = dyniface if dyniface else Type(self, iface)

    return names_to_ir

  def _no_cache_build_package(self) -> cgx.Package:
    return self.device.build(self.__path)

  def __no_cache_build(self) -> cgx.Package:
    pkg = self._no_cache_build_package()
    for vr in self.__statics:
      static_var = pkg.find_static_var(vr.name)
      static_var.set_value(vr.value)
    self.__names_to_ir = self.__build_names_to_ir(pkg)
    return pkg

  def find_interface(self, name: str):
    self.build()
    iface = self.__names_to_ir.get(name, None)
    if iface is None:
      raise RuntimeError(f"package {self.__path} has no type {name}")
    return iface

  def build(self) -> cgx.Package:
    return self.__cache.get(self.__no_cache_build)

  def __dir__(self):
    try:
      self.build()
    except cgx.Error:
      return ["compile_error"]
    names = list(
        name
        for name, element in self.__names_to_ir.items()
        if not isinstance(element, Type)
    )
    names.sort()
    return names

  def __getattr__(self, name: str):
    self.build()
    ir = self.__names_to_ir.get(name, None)
    if ir is None:
      raise AttributeError(name)
    return ir


class StaticVar:
  """GX static variable."""

  def __init__(self, pkg: Package, vr: cgx.StaticVar):
    self.__pkg = pkg
    self.__name = vr.name()
    self.__value = None

  def set(self, value: int):
    self.__value = value
    self.__pkg.update_static_variable(self)

  @property
  def value(self) -> int | None:
    return self.__value

  @property
  def has_value(self) -> bool:
    return self.__value is not None

  @property
  def name(self) -> str:
    return self.__name


def _process_args(device: cgx.Device, func: cgx.Function, args):
  """Transforms python arguments into GX arguments."""
  if len(args) != func.num_params():
    raise TypeError(
        f"got {len(args)} arguments but {func.name} want"
        f" {func.num_params()} arguments"
    )
  largs = []
  for i, arg in enumerate(args):
    param_dtype = func.param_dtype(i)
    largs.append(cgx.value_from_pyobj(device, arg, param_dtype))
  return largs


def _process_return_values(device: cgx.Device, pkg: cgx.Package, values):
  if not values:
    return None
  vals = tuple(
      cgx.python_value_from_gx_value(device, pkg, val) for val in values
  )
  return vals if len(vals) > 1 else vals[0]


class Function:
  """GX Function."""

  def __init__(self, pkg: Package, func: cgx.Function):
    self.__pkg = pkg
    self.__name = func.name()

  def __no_cache_build(self):
    pkg = self.__pkg.build()
    return pkg, pkg.find_function(self.__name)

  def __build(self) -> tuple[cgx.Package, cgx.Function]:
    return self.__pkg.cache.get(self.__no_cache_build)

  def __call__(self, *args):
    if super_hot:
      clear_all_cache()
    pkg, func = self.__build()
    gx_args = _process_args(self.__pkg.device, func, args)
    gx_ret_values = func.run(gx_args).return_values()
    return _process_return_values(self.__pkg.device, pkg, gx_ret_values)

  def __str__(self):
    _, func = self.__build()
    return str(func)

  def pkg(self) -> Package:
    return self.__pkg

  def name(self) -> str:
    return self.__name


class Method:
  """GX  method of an interface."""

  def __init__(self, iface: Type, method: cgx.Function):
    self.__iface = iface
    self.__name = method.name()

  def __no_cache_build(self):
    pkg, iface = self.__iface.build()
    return pkg, iface.find_method(self.__name)

  @property
  def interface(self) -> Type:
    return self.__iface

  @property
  def device(self) -> cgx.Device:
    return self.__iface.device

  def build(self) -> tuple[cgx.Package, cgx.Function]:
    return self.__iface.cache.get(self.__no_cache_build)


class _BoundMethod:
  """GX method bound to a receiver."""

  def __init__(self, receiver: Instance, method: Method):
    self.__receiver = receiver
    self.__method = method

  def __call__(self, *args):
    if super_hot:
      clear_all_cache()
    pkg, func = self.__method.build()
    device = self.__method.device
    largs = _process_args(device, func, args)
    ret_values = func.run_method(self.__receiver.value, largs).return_values()
    return _process_return_values(device, pkg, ret_values)


class Type:
  """GX Type."""

  def __init__(self, pkg: Package, iface: cgx.Type):
    self.__package = pkg
    self.__type_name = iface.name()
    self.__names_to_ir: dict[str, Method] = {}

  @property
  def package(self) -> Package:
    return self.__package

  @property
  def device(self) -> cgx.Device:
    return self.__package.device

  @property
  def cache(self) -> _Cache:
    return self.__package.cache

  def __build_names_to_ir(self, iface: cgx.Type):
    names_to_ir = {}

    for method in iface.list_methods():
      name = method.name()
      dynfunc = self.__names_to_ir.get(name, None)
      dynfunc = Method(self, method) if dynfunc is None else dynfunc
      names_to_ir[name] = dynfunc

    return names_to_ir

  def __no_cache_build(self) -> tuple[cgx.Package, cgx.Type]:
    pkg = self.__package.build()
    iface = pkg.find_interface(self.__type_name)
    self.__names_to_ir = self.__build_names_to_ir(iface)
    return pkg, iface

  def build(self) -> tuple[cgx.Package, cgx.Type]:
    return self.__package.cache.get(self.__no_cache_build)

  def list_fields(self):
    _, iface = self.build()
    return iface.list_fields()

  def list_methods(self):
    _, iface = self.build()
    return [method.name() for method in iface.list_methods()]

  def find_method(self, name: str):
    self.build()
    return self.__names_to_ir.get(name, None)

  def __repr__(self):
    return f"{self.__package.path}.{self.__type_name}{{}}"

  def __str__(self):
    return self.__repr__()


class Instance:
  """Instance is a dyngx Type attached to an instance."""

  def __init__(self, iface: Type, receiver):
    self.__iface = iface
    self.__receiver = receiver

  def to_gx_value(self):
    return self.__receiver

  @property
  def value(self):
    return self.__receiver

  def __dir__(self):
    try:
      methods = self.__iface.list_methods()
      fields = self.__iface.list_fields()
      return methods + fields
    except cgx.Error:
      return ["compile_error"]

  def __getattr__(self, name):
    method = self.__iface.find_method(name)
    if method is not None:
      return _BoundMethod(self, method)
    pkg, _ = self.__iface.build()
    field_value = self.__receiver.get_field(name)
    if field_value is not None:
      return cgx.python_value_from_gx_value(
          self.__iface.device, pkg, field_value
      )
    raise AttributeError(name)

  def __repr__(self):
    return str(self.__receiver.__repr__())

  def __str__(self):
    return self.__repr__()


def cache(dev: cgx.Device) -> _Cache:
  device_cache = _device_to_cache.get(dev, None)
  if device_cache is None:
    device_cache = _Cache(dev)
    _device_to_cache[dev] = device_cache
  return device_cache


def _build_interface(iface: cgx.Interface) -> Instance:
  device_cache = cache(iface.device)
  pkg = device_cache.package(iface.interface.package_name())
  dyniface = pkg.find_interface(iface.interface.name())
  return Instance(dyniface, iface.receiver)


cgx.interface_wrapper = _build_interface  # pyrefly: ignore[bad-assignment]


def build_package(dev: cgx.Device, path: str) -> Package:
  """Build a dynamic package."""
  return cache(dev).package(path)


def clear_all_cache():
  for device_cache in _device_to_cache.values():
    device_cache.clear()
