from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import DATABASE_URL

_connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    # allow use from worker threads (e.g. TestClient's portal thread)
    _connect_args["check_same_thread"] = False

_engine_kwargs = {"connect_args": _connect_args} if _connect_args else {}
engine = create_engine(DATABASE_URL, **_engine_kwargs)
SessionLocal = sessionmaker(bind=engine)