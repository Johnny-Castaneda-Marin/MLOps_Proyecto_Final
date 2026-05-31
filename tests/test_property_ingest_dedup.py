"""Prueba basada en propiedades de la ingesta: deduplicación, idempotencia y
estabilidad del hash (tarea 2.4).

Implementa una única prueba que verifica las tres sub-propiedades de la
Property 3 del diseño sobre las funciones puras ``mlops_core.ingest.deduplicate``
y ``mlops_core.ingest.row_hash``.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from mlops_core.ingest import deduplicate, row_hash

# Estrategia de registros de propiedad inmobiliaria con claves variadas y
# valores de tipos mixtos (números, cadenas, nulos). El espacio se mantiene
# pequeño para forzar colisiones de hash y duplicados con frecuencia.
record_strategy = st.dictionaries(
    keys=st.sampled_from(
        [
            "brokered_by",
            "status",
            "price",
            "bed",
            "bath",
            "acre_lot",
            "street",
            "city",
            "state",
            "zip_code",
            "house_size",
            "prev_sold_date",
        ]
    ),
    values=st.one_of(
        st.none(),
        st.integers(min_value=-5, max_value=5),
        st.floats(allow_nan=False, allow_infinity=False, min_value=-100.0, max_value=100.0),
        st.sampled_from(["for_sale", "sold", "Austin", "Texas", "Florida", ""]),
    ),
    min_size=0,
    max_size=4,
)


# Feature: mlops-real-estate-platform, Property 3: Deduplicación, idempotencia y estabilidad del hash
@settings(max_examples=100)
@given(
    records=st.lists(record_strategy, min_size=0, max_size=20),
    existing_hashes=st.sets(st.text(min_size=0, max_size=8), max_size=10),
)
def test_property_deduplicacion_idempotencia_y_estabilidad_del_hash(records, existing_hashes):
    """Property 3 (Validates: Requirements 1.4).

    1. Deduplicación: ``deduplicate`` no devuelve ningún registro cuyo
       ``row_hash`` ya esté presente en ``existing_hashes``.
    2. Idempotencia: reaplicar ``deduplicate`` sobre su propia salida (con los
       hashes existentes ya extendidos) no elimina más registros.
    3. Estabilidad del hash: ``row_hash`` produce el mismo valor para un
       registro independientemente del orden de sus claves.
    """
    new_records, new_hashes = deduplicate(records, existing_hashes)

    # --- Sub-propiedad 1: deduplicación ---
    # Ningún hash devuelto puede estar ya presente en los hashes existentes.
    for h in new_hashes:
        assert h not in existing_hashes
    # Cada registro devuelto coincide con su hash reportado y no estaba presente.
    for record, h in zip(new_records, new_hashes):
        assert row_hash(record) == h
        assert h not in existing_hashes
    # Los hashes devueltos son únicos (sin duplicados dentro del lote nuevo).
    assert len(new_hashes) == len(set(new_hashes))

    # --- Sub-propiedad 2: idempotencia ---
    # Reaplicar deduplicate sobre la salida, con los hashes existentes
    # extendidos por los nuevos, no debe eliminar ningún registro adicional.
    extended_hashes = set(existing_hashes) | set(new_hashes)
    second_records, second_hashes = deduplicate(new_records, extended_hashes)
    assert second_records == []
    assert second_hashes == []

    # Reaplicar deduplicate sobre la salida con SOLO los hashes existentes
    # originales devuelve exactamente la misma salida (estabilidad/idempotencia).
    again_records, again_hashes = deduplicate(new_records, existing_hashes)
    assert again_records == new_records
    assert again_hashes == new_hashes

    # --- Sub-propiedad 3: estabilidad del hash frente al orden de claves ---
    for record in records:
        shuffled = dict(reversed(list(record.items())))
        assert row_hash(shuffled) == row_hash(record)
