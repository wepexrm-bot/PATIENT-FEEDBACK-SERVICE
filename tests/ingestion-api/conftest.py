import importlib.util
import sys
from pathlib import Path

_SERVICE = Path(__file__).resolve().parent.parent.parent / "ingestion-api"
for _name in [m for m in list(sys.modules) if m == "app" or m.startswith("app.")]:
    del sys.modules[_name]
_spec = importlib.util.spec_from_file_location(
    "app", _SERVICE / "app" / "__init__.py", submodule_search_locations=[str(_SERVICE / "app")]
)
_pkg = importlib.util.module_from_spec(_spec)
sys.modules["app"] = _pkg
_spec.loader.exec_module(_pkg)
