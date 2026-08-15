import importlib.util
import os
import sys
import tempfile
from pathlib import Path

_SERVICE = Path(__file__).resolve().parent.parent.parent / "governance-service"
for _name in [m for m in list(sys.modules) if m == "app" or m.startswith("app.")]:
    del sys.modules[_name]
_spec = importlib.util.spec_from_file_location(
    "app", _SERVICE / "app" / "__init__.py", submodule_search_locations=[str(_SERVICE / "app")]
)
_pkg = importlib.util.module_from_spec(_spec)
sys.modules["app"] = _pkg
_spec.loader.exec_module(_pkg)

# Point governance at a disposable SQLite DB before app.db is used.
_TMP = tempfile.mkdtemp(prefix="gov-test-")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TMP}/governance.db")

from app.db import engine
from app.models import Base

Base.metadata.create_all(bind=engine)

import pytest


@pytest.fixture(autouse=True)
def _clean_db():
    """Wipe all rows before each test so suites are order-independent."""
    with engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())
    yield
