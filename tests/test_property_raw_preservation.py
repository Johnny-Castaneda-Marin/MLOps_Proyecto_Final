"""Prueba basada en propiedades de la preservación de datos crudos (tarea 2.8).

Implementa la Property 4 de la sección "Correctness Properties" del diseño,
sobre la transformación que aplica el adaptador ``store_raw`` al ensamblar la
fila a persistir en ``raw_properties``.

El adaptador real mezcla I/O con PostgreSQL; para mantener esta prueba como una
PBT PURA (sin levantar PostgreSQL ni SQLAlchemy) se ejercita el invariante de
preservación de campos de la transformación que ``store_raw`` aplica:

1. ``mlops_core.ingest.deduplicate`` devuelve los MISMOS objetos de registro sin
   mutarlos ni transformar sus campos.
2. La fila almacenada se construye como ``{**record, **metadata}``: solo se
   AÑADEN columnas de metadatos (``row_hash``, ``loaded_at``, ``processed``,
   ``batch_number``, ``day_used``); ningún campo original se modifica.

La propiedad asegura el round-trip de los campos crudos: el subconjunto de la
fila almacenada restringido a las claves originales es idéntico al registro
recibido de la Data_API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from hypothesis import given, settings
from hypothesis import strategies as st

from mlops_core.ingest import deduplicate, row_hash

# Columnas de metadatos que añade el adaptador ``store_raw`` (no forman parte
# del registro crudo recibido de la Data_API). Reflejan ``_METADATA_COLUMNS`` del
# adaptador real.
METADATA_COLUMNS = ("row_hash", "loaded_at", "processed", "batch_number", "day_used")


def assemble_stored_row(
    record: Dict[str, Any],
    batch_number: int,
    day_used: str,
) -> Dict[str, Any]:
    """Réplica pura del ensamblado de fila del adaptador ``store_raw``.

    Imita la lógica del adaptador (``df["row_hash"] = ...; df["batch_number"] =
    ...; etc.``) sin tocar pandas ni SQLAlchemy: parte del registro crudo y solo
    AÑADE columnas de metadatos, sin transformar ningún campo original.
    """
    return {
        **record,
        "row_hash": row_hash(record),
        "batch_number": batch_number,
        "day_used": day_used,
        "loaded_at": datetime.utcnow(),
        "processed": False,
    }


# Estrategia para un registro inmobiliario crudo "estilo Data_API": claves del
# esquema esperado con tipos realistas (numéricos / texto, con nulos inyectados),
# excluyendo deliberadamente los nombres de columnas de metadatos para que el
# registro represente lo recibido de la fuente externa.
_numbers = st.one_of(
    st.none(),
    st.integers(min_value=-10_000, max_value=10_000_000),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
)
_strings = st.one_of(st.none(), st.text(max_size=20))

property_record = st.fixed_dictionaries(
    {
        "brokered_by": _numbers,
        "status": _strings,
        "price": _numbers,
        "bed": _numbers,
        "bath": _numbers,
        "acre_lot": _numbers,
        "street": _numbers,
        "city": _strings,
        "state": _strings,
        "zip_code": _numbers,
        "house_size": _numbers,
        "prev_sold_date": _strings,
    }
)


# Feature: mlops-real-estate-platform, Property 4: Preservación de datos crudos
@settings(max_examples=100)
@given(
    record=property_record,
    batch_number=st.integers(min_value=0, max_value=10_000),
    day_used=st.sampled_from(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    ),
)
def test_raw_fields_preserved_after_store_assembly(
    record: Dict[str, Any], batch_number: int, day_used: str
) -> None:
    """Los campos crudos se conservan idénticos tras el ensamblado de ``store_raw``.

    *Para todo* registro válido entrante, los campos originales conservados en
    ``raw_properties`` (excluyendo metadatos añadidos como ``row_hash``,
    ``loaded_at``, ``processed``, ``batch_number``, ``day_used``) son idénticos
    a los del registro recibido de la Data_API: round-trip de campos crudos, sin
    transformación de negocio.

    **Validates: Requirements 2.1**
    """
    original_keys = set(record.keys())
    # Un registro de la Data_API no contiene columnas de metadatos.
    assert original_keys.isdisjoint(METADATA_COLUMNS)

    # 1) deduplicate no muta ni transforma los registros: devuelve el MISMO objeto.
    new_records, _ = deduplicate([record], existing_hashes=set())
    assert len(new_records) == 1
    assert new_records[0] is record  # identidad: ningún campo fue transformado
    assert new_records[0] == record

    # 2) El ensamblado de la fila solo AÑADE metadatos.
    stored_row = assemble_stored_row(new_records[0], batch_number, day_used)

    # Restringir la fila almacenada a las claves originales debe reproducir
    # exactamente el registro recibido (round-trip de campos crudos).
    preserved = {k: stored_row[k] for k in original_keys}
    assert preserved == record

    # Cada metadato esperado fue añadido, y no se omitió ni renombró ninguna
    # clave original.
    assert set(stored_row.keys()) == original_keys | set(METADATA_COLUMNS)
