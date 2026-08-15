"""Shared test bootstrap.

Makes the repo root importable (so `shared.*` works) and helpers to activate a
specific service's `app` package deterministically.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def activate_app(service_dir: Path) -> None:
    """Register a service's `app` package under the module name 'app'.

    Every service ships code under a package literally named `app`; importing
    them all into one interpreter session collides in sys.modules. We load the
    requested package by absolute spec and bind `sys.modules['app']` to it so
    `import app.*` deterministically targets THIS service, independent of
    sys.path order or when conftests were loaded.
    """
    for name in [m for m in list(sys.modules) if m == "app" or m.startswith("app.")]:
        del sys.modules[name]
    package_init = service_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "app",
        package_init,
        submodule_search_locations=[str(service_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create spec for {service_dir}")
    package = importlib.util.module_from_spec(spec)
    sys.modules["app"] = package
    spec.loader.exec_module(package)