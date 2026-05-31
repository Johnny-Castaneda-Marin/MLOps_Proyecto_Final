"""Prueba basada en propiedades de ``validate_schema`` (tarea 3.2).

Property 6 (sección "Correctness Properties" del diseño): la validación de
esquema marca el lote como válido si y solo si no hay columnas faltantes, ni
columnas adicionales, ni cambios de tipo respecto a ``EXPECTED_SCHEMA``;
introducir cualquiera de esas discrepancias lo marca inválido.

La prueba ejercita ambas direcciones del "si y solo si":
1. Registros que conforman ``EXPECTED_SCHEMA`` (tipos correctos por
   "number"/"string") => ``valid=True``.
2. Una mutación que introduce una discrepancia (eliminar una columna, añadir una
   columna extra, o invertir el tipo de una columna) => ``valid=False``.
"""

from __future__ import annotations

from typing import Any, Dict, List

import string

from hypothesis import given, settings
from hypothesis import strategies as st

from mlops_core.validation import EXPECTED_SCHEMA, validate_schema

# Particionado de columnas esperadas por tipo lógico.
NUMBER_COLUMNS = [c for c, t in EXPECTED_SCHEMA.items() if t == "number"]
STRING_COLUMNS = [c for c, t in EXPECTED_SCHEMA.items() if t == "string"]

# Generadores de valores con el tipo lógico correcto.
# Un valor "number" es int/float real (no bool, no NaN/inf).
number_values = st.one_of(
    st.integers(min_value=-1_000_000, max_value=1_000_000),
    st.floats(
        allow_nan=False, allow_infinity=False, min_value=-1e9, max_value=1e9
    ),
)
# Un valor "string" es cualquier texto: ``_is_number`` siempre es falso para str,
# de modo que el tipo lógico inferido es "string" aunque el texto parezca numérico.
string_values = st.text(alphabet=string.ascii_letters, min_size=1, max_size=10)

# Nombres válidos para una columna extra: identificadores que NO pertenecen al
# esquema esperado.
extra_column_names = st.text(
    alphabet=string.ascii_lowercase + "_", min_size=1, max_size=12
).filter(lambda name: name not in EXPECTED_SCHEMA)


@st.composite
def conforming_records(draw: st.DrawFn) -> List[Dict[str, Any]]:
    """Genera un conjunto no vacío de registros que conforman ``EXPECTED_SCHEMA``.

    Cada registro contiene exactamente las columnas esperadas, con valores del
    tipo lógico correcto y no nulos (para que el tipo inferido sea inequívoco).
    """
    size = draw(st.integers(min_value=1, max_value=8))
    records: List[Dict[str, Any]] = []
    for _ in range(size):
        record: Dict[str, Any] = {}
        for column in NUMBER_COLUMNS:
            record[column] = draw(number_values)
        for column in STRING_COLUMNS:
            record[column] = draw(string_values)
        records.append(record)
    return records


# Feature: mlops-real-estate-platform, Property 6: Validación de esquema detecta toda discrepancia
@settings(max_examples=100)
@given(records=conforming_records(), data=st.data())
def test_schema_validation_detects_every_discrepancy(
    records: List[Dict[str, Any]], data: st.DataObject
) -> None:
    """Validates: Requirements 3.1.

    Dirección directa: un lote conforme se marca como válido. Dirección inversa:
    introducir cualquier discrepancia (columna faltante, columna adicional o
    cambio de tipo) marca el lote como inválido.
    """
    # Dirección 1: registros conformes => válido y sin discrepancias.
    result = validate_schema(records)
    assert result.valid is True
    assert result.missing_columns == []
    assert result.extra_columns == []
    assert result.type_mismatches == []

    # Dirección 2: una mutación introduce una discrepancia => inválido.
    mutation = data.draw(
        st.sampled_from(["drop", "extra", "flip_number", "flip_string"])
    )
    mutated: List[Dict[str, Any]] = [dict(record) for record in records]

    if mutation == "drop":
        # Eliminar una columna esperada de TODOS los registros => columna faltante.
        column = data.draw(st.sampled_from(sorted(EXPECTED_SCHEMA)))
        for record in mutated:
            record.pop(column, None)
        expected_marker = "missing_columns"
    elif mutation == "extra":
        # Añadir una columna no esperada a TODOS los registros => columna adicional.
        extra_column = data.draw(extra_column_names)
        for record in mutated:
            record[extra_column] = data.draw(st.one_of(number_values, string_values))
        expected_marker = "extra_columns"
    elif mutation == "flip_number":
        # Convertir una columna numérica en texto no numérico => cambio de tipo.
        column = data.draw(st.sampled_from(NUMBER_COLUMNS))
        for record in mutated:
            record[column] = "not_a_number"
        expected_marker = "type_mismatches"
    else:  # flip_string
        # Convertir una columna de texto en numérica => cambio de tipo.
        column = data.draw(st.sampled_from(STRING_COLUMNS))
        replacement = data.draw(number_values)
        for record in mutated:
            record[column] = replacement
        expected_marker = "type_mismatches"

    mutated_result = validate_schema(mutated)
    assert mutated_result.valid is False
    # La discrepancia esperada queda efectivamente reportada en el resultado.
    assert getattr(mutated_result, expected_marker)
