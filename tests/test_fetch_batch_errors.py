"""Pruebas unitarias de manejo de errores del adaptador ``fetch_batch`` (Tarea 2.6).

Valida los caminos de error de :func:`fetch_batch.fetch_batch` con un cliente HTTP
mockeado: tiempo de espera agotado (timeout), errores HTTP 4xx/5xx y respuesta
vacía. Cubre los criterios de aceptación:

- **Validates: Requirements 1.7** (timeout/HTTP error ⇒ falla preservando metadatos
  y registrando el error).
- **Validates: Requirements 12.1** (la condición de error/respuesta vacía/fin de
  datos se maneja sin corromper el estado del proceso).
- **Validates: Requirements 12.2** (las fallas transitorias relanzan la excepción
  para permitir el reintento configurado del DAG).

Estrategia de aislamiento
-------------------------
El adaptador vive en ``airflow/dags/tasks/fetch_batch.py`` e importa ``requests``,
``sqlalchemy`` y ``config`` (dependencias que solo existen dentro del contenedor de
Airflow). Para poder importarlo en este entorno de desarrollo:

1. Se inyectan módulos falsos ligeros para ``requests`` y ``sqlalchemy`` en
   ``sys.modules`` ANTES de importar el adaptador (solo si no están instalados).
2. Se agrega ``airflow/dags/tasks`` a ``sys.path`` para resolver ``from config import``.
3. ``mlops_core`` (lógica pura real) ya es importable gracias a ``conftest.py``.

Cada prueba reemplaza ``create_engine`` por una factoría que devuelve un motor
falso controlable y parchea ``requests.get`` con un doble configurable, de modo que
las ramas de manejo de errores del adaptador se ejecutan y verifican de verdad.
Todo lo inyectado se limpia en :func:`teardown_module`.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

# --------------------------------------------------------------------------- #
# Localización de rutas
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = REPO_ROOT / "airflow" / "dags" / "tasks"

# Registro de lo que esta prueba inyecta en el proceso, para limpiarlo después.
_INJECTED_MODULES: list[str] = []
_INJECTED_PATHS: list[str] = []


# --------------------------------------------------------------------------- #
# Dobles de prueba para ``requests``
# --------------------------------------------------------------------------- #
class _FakeTimeout(Exception):
    """Equivalente al ``requests.exceptions.Timeout`` real."""


class _FakeHTTPError(Exception):
    """Equivalente al ``requests.exceptions.HTTPError`` real."""


class FakeResponse:
    """Respuesta HTTP mínima compatible con el uso que hace ``fetch_batch``."""

    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self):
        if self._json_data is None:
            raise ValueError("respuesta sin cuerpo JSON")
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise _FakeHTTPError(f"HTTP {self.status_code}")


def _install_fake_requests() -> None:
    """Inyecta un módulo ``requests`` falso si el real no está disponible."""
    try:
        import requests  # noqa: F401
        return
    except ImportError:
        pass

    fake = types.ModuleType("requests")
    exceptions = types.ModuleType("requests.exceptions")
    exceptions.Timeout = _FakeTimeout
    exceptions.HTTPError = _FakeHTTPError
    exceptions.RequestException = Exception
    fake.exceptions = exceptions
    fake.Timeout = _FakeTimeout
    fake.HTTPError = _FakeHTTPError

    def _default_get(*_args, **_kwargs):  # pragma: no cover - se parchea por prueba
        raise AssertionError("requests.get debe parchearse en cada prueba")

    fake.get = _default_get

    sys.modules["requests"] = fake
    sys.modules["requests.exceptions"] = exceptions
    _INJECTED_MODULES.extend(["requests", "requests.exceptions"])


# --------------------------------------------------------------------------- #
# Dobles de prueba para ``sqlalchemy`` (motor / conexión)
# --------------------------------------------------------------------------- #
class FakeResult:
    """Resultado de ``execute`` que devuelve una fila fija en ``fetchone``."""

    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class FakeConnection:
    """Conexión que registra cada sentencia ejecutada y simula el COUNT."""

    def __init__(self, engine):
        self._engine = engine

    def execute(self, statement, params=None):
        self._engine.executed.append({"sql": str(statement), "params": params})
        # Solo ``SELECT COUNT(*)`` invoca ``fetchone``; devolver el índice de lote.
        return FakeResult([self._engine.batch_count])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Begin:
    """Gestor de contexto devuelto por ``engine.begin()``."""

    def __init__(self, engine):
        self._engine = engine

    def __enter__(self):
        return FakeConnection(self._engine)

    def __exit__(self, *exc):
        return False


class FakeEngine:
    """Motor SQLAlchemy falso controlable y observable."""

    def __init__(self, conn_str="fake://", batch_count=0):
        self.conn_str = conn_str
        self.batch_count = batch_count
        self.executed: list[dict] = []
        self.disposed = False

    def begin(self):
        return _Begin(self)

    def dispose(self):
        self.disposed = True


def _install_fake_sqlalchemy() -> None:
    """Inyecta un módulo ``sqlalchemy`` falso si el real no está disponible."""
    try:
        import sqlalchemy  # noqa: F401
        return
    except ImportError:
        pass

    fake = types.ModuleType("sqlalchemy")

    def _create_engine(conn_str, *args, **kwargs):  # pragma: no cover - se parchea
        return FakeEngine(conn_str)

    def _text(sql):
        return sql

    fake.create_engine = _create_engine
    fake.text = _text

    sys.modules["sqlalchemy"] = fake
    _INJECTED_MODULES.append("sqlalchemy")


# --------------------------------------------------------------------------- #
# Importación del adaptador bajo prueba
# --------------------------------------------------------------------------- #
def _import_fetch_batch():
    """Importa el módulo ``fetch_batch`` tras preparar las dependencias falsas."""
    _install_fake_requests()
    _install_fake_sqlalchemy()

    tasks_str = str(TASKS_DIR)
    if tasks_str not in sys.path:
        sys.path.insert(0, tasks_str)
        _INJECTED_PATHS.append(tasks_str)

    repo_str = str(REPO_ROOT)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
        _INJECTED_PATHS.append(repo_str)

    return importlib.import_module("fetch_batch")


fb = _import_fetch_batch()


def teardown_module(_module):
    """Limpia los módulos y rutas inyectados para no contaminar otras pruebas."""
    for name in ("fetch_batch", "config"):
        sys.modules.pop(name, None)
    for name in _INJECTED_MODULES:
        sys.modules.pop(name, None)
    for path in _INJECTED_PATHS:
        try:
            sys.path.remove(path)
        except ValueError:
            pass


# --------------------------------------------------------------------------- #
# Utilidades de prueba
# --------------------------------------------------------------------------- #
class FakeTI:
    """Doble de ``TaskInstance`` que captura los ``xcom_push``."""

    def __init__(self):
        self.xcoms: dict = {}

    def xcom_push(self, key, value):
        self.xcoms[key] = value


@pytest.fixture
def engine(monkeypatch):
    """Motor falso instalado en el adaptador; se devuelve para hacer aserciones."""
    eng = FakeEngine(batch_count=0)
    monkeypatch.setattr(fb, "create_engine", lambda *a, **k: eng)
    return eng


def _set_get(monkeypatch, fake_get):
    monkeypatch.setattr(fb.requests, "get", fake_get)


def _inserts(engine):
    """Devuelve las sentencias INSERT registradas en el motor falso."""
    return [e for e in engine.executed if "INSERT INTO batch_control" in e["sql"]]


# --------------------------------------------------------------------------- #
# (a) Timeout ⇒ la tarea falla relanzando la excepción
# --------------------------------------------------------------------------- #
def test_fetch_batch_timeout_raises_and_disposes(engine, monkeypatch):
    """Un timeout del cliente HTTP relanza la excepción y libera el motor.

    Validates: Requirements 1.7, 12.2
    """
    def fake_get(*_a, **_k):
        raise _FakeTimeout("tiempo de espera agotado tras 120s")

    _set_get(monkeypatch, fake_get)
    ti = FakeTI()

    with pytest.raises(Exception):
        fb.fetch_batch(ti=ti)

    # No se insertan metadatos de lote y no se publican XComs de éxito.
    assert _inserts(engine) == []
    assert "batch_number" not in ti.xcoms
    assert "records" not in ti.xcoms
    # El motor se libera incluso en la ruta de error (sin fugas de recursos).
    assert engine.disposed is True


# --------------------------------------------------------------------------- #
# (b) HTTP 400 (agotamiento) ⇒ se maneja sin error, marca data_exhausted
# --------------------------------------------------------------------------- #
def test_fetch_batch_http_400_marks_exhausted_without_raising(engine, monkeypatch):
    """El HTTP 400 de fin de datos se maneja sin fallar la tarea.

    Validates: Requirements 12.1
    """
    response = FakeResponse(status_code=400, json_data={"detail": "no more data"})
    _set_get(monkeypatch, lambda *a, **k: response)
    ti = FakeTI()

    # No debe lanzar excepción.
    fb.fetch_batch(ti=ti)

    assert ti.xcoms.get("data_exhausted") is True
    # Se registra el metadato con estado 'exhausted' y 0 registros (bn=-1).
    inserts = _inserts(engine)
    assert len(inserts) == 1
    assert "'exhausted'" in inserts[0]["sql"]
    assert inserts[0]["params"]["bn"] == -1
    assert engine.disposed is True


# --------------------------------------------------------------------------- #
# (c) HTTP 4xx distinto de 400 ⇒ la tarea falla relanzando la excepción
# --------------------------------------------------------------------------- #
def test_fetch_batch_http_404_raises(engine, monkeypatch):
    """Un 4xx que no es agotamiento (p. ej. 404) hace fallar la tarea.

    Validates: Requirements 1.7, 12.2
    """
    response = FakeResponse(status_code=404, json_data={"detail": "not found"})
    _set_get(monkeypatch, lambda *a, **k: response)
    ti = FakeTI()

    with pytest.raises(Exception):
        fb.fetch_batch(ti=ti)

    # No se registra metadato de lote 'fetched' ni se publican XComs de éxito.
    assert _inserts(engine) == []
    assert "batch_number" not in ti.xcoms
    assert engine.disposed is True


# --------------------------------------------------------------------------- #
# (d) HTTP 5xx ⇒ la tarea falla relanzando la excepción
# --------------------------------------------------------------------------- #
def test_fetch_batch_http_500_raises(engine, monkeypatch):
    """Un error 5xx del servidor hace fallar la tarea (apto para reintento).

    Validates: Requirements 1.7, 12.2
    """
    response = FakeResponse(status_code=500, json_data={"detail": "server error"})
    _set_get(monkeypatch, lambda *a, **k: response)
    ti = FakeTI()

    with pytest.raises(Exception):
        fb.fetch_batch(ti=ti)

    assert _inserts(engine) == []
    assert "batch_number" not in ti.xcoms
    assert engine.disposed is True


# --------------------------------------------------------------------------- #
# (e) Respuesta exitosa con lista de datos vacía ⇒ persiste metadatos y XComs
# --------------------------------------------------------------------------- #
def test_fetch_batch_empty_data_stores_metadata_and_pushes(engine, monkeypatch):
    """Una respuesta 200 con ``data: []`` se maneja sin error y publica XComs.

    Validates: Requirements 12.1
    """
    engine.batch_count = 2  # índice de lote 2 ⇒ día "Wednesday"
    response = FakeResponse(
        status_code=200,
        json_data={"batch_number": 5, "data": []},
    )
    _set_get(monkeypatch, lambda *a, **k: response)
    ti = FakeTI()

    fb.fetch_batch(ti=ti)

    # XComs de éxito con lista de registros vacía.
    assert ti.xcoms.get("batch_number") == 5
    assert ti.xcoms.get("records") == []
    assert ti.xcoms.get("day") == "Wednesday"
    assert ti.xcoms.get("data_exhausted") is False

    # Metadato persistido con estado 'fetched' y 0 registros.
    inserts = _inserts(engine)
    assert len(inserts) == 1
    assert "'fetched'" in inserts[0]["sql"]
    assert inserts[0]["params"]["bn"] == 5
    assert inserts[0]["params"]["cnt"] == 0
    assert engine.disposed is True
