"""Helper classes for writing reports"""
import contextlib
import importlib
from typing import Dict, ContextManager


class _ReportMakerMeta(type):
  """Metaclass for report makers. It implements the report maker factory"""
  _report_types: Dict[str, type] = {}

  def __new__(mcs, *args, **kwargs):
    """When creating a new class, register it to
    the dictionary of subclasses"""
    cls = super().__new__(mcs, *args, **kwargs)
    type_key = getattr(cls, "_type_key", None)
    if type_key is not None:
      mcs._report_types[type_key] = cls
    return cls

  def __call__(cls, *args, **kwargs):
    """Instantiating a new class, if a type argument is specified,
    instantiate the corresponding class"""
    type_key = kwargs.pop("type", None)
    c = cls if type_key is None else cls._report_types[type_key]
    obj = c.__new__(c, *args, **kwargs)
    obj.__init__(*args, **kwargs)
    return obj


class ReportMaker(metaclass=_ReportMakerMeta):
  """Report maker class.

  Args:
    type (str): The type of output to generate. Available types are:
      :data:`"plain-text"`, :data:`"latex"`"""

  def __init__(self, *args, **kwargs):  # pylint: disable=W0613
    super().__init__()
    self._q = []

  def clear(self):
    """Clear the internal queue"""
    self._q.clear()

  def get(self) -> str:
    """Get the text output"""
    return "".join(self._q)

  def append(self, *s: str, **kwargs):  # pylint: disable=W0613
    """Enqueue some text"""
    for si in s:
      self._q.append(si)

  def newline(self):
    """Enqueue a new line"""
    self._q.append("\n")

  def _itemizer_kwargs(self):
    """Get the argument dictionary for the itemizer"""
    kw = {
        "type": "_plain-text-itemize",
        "level": getattr(self, "level", 0) + 1,
    }
    if hasattr(self, "indent_size"):
      kw["indent_size"] = getattr(self, "indent_size")
    return kw

  def display(self):
    """Display in IPython"""
    return importlib.import_module("IPython.display").Pretty(self.get())

  @contextlib.contextmanager
  def itemize(self) -> ContextManager["_ItemizeReportMaker"]:
    """Open an itemization maker"""
    it = ReportMaker(**self._itemizer_kwargs())
    yield it
    self.append(it.get())


class _ReportMakerPlain(ReportMaker):
  """Plain-text report maker (default)"""
  _type_key: str = "plain-text"


class _ItemizeReportMaker(ReportMaker):
  """Itemization handler for plain-text report maker

  Args:
    level (int): Indentation level (incremented for nested itemizations)
    indent_size (int): Indentation width for each level"""
  _type_key: str = "_plain-text-itemize"

  def __init__(self, *args, level: int = 0, indent_size: int = 2, **kwargs):
    super().__init__(*args, **kwargs)
    self.level = level
    self.indent_size = indent_size

  def item(self, *s: str, **kwargs):
    """Append an item to the itemization"""
    super().append(f"\n{' ' * self.level * self.indent_size}", *s, **kwargs)


class _ReportMakerLatex(ReportMaker):
  """Latex report maker (default)"""
  _type_key: str = "latex"
  _escape_list = [("%", r"\%")]

  def append(self, *s: str, escape: bool = False):
    if escape:
      for s_old, s_new in self._escape_list:
        s = [si.replace(s_old, s_new) for si in s]
    super().append(*s)

  def newline(self):
    self._q.append(r"\newline")

  def _itemizer_kwargs(self):
    return {
        "type": "_latex-itemize",
    }

  @contextlib.contextmanager
  def itemize(self) -> ContextManager["_ItemizeReportMakerLatex"]:
    self.append(r"\begin{itemize}")
    with super().itemize() as it:
      yield it
    self.append(r"\end{itemize}")

  def display(self):
    return importlib.import_module("IPython.display").Latex(self.get())


class _ItemizeReportMakerLatex(_ReportMakerLatex, _ItemizeReportMaker):
  """Itemization handler for latex report maker"""
  _type_key: str = "_latex-itemize"

  def item(self, *s: str, **kwargs):
    super().append("\n", r"\item ", *s, **kwargs)
