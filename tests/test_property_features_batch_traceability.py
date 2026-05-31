"""Prueba basada en propiedades de la preservación de trazabilidad por lote (tarea 4.4).

Implementa la Property 5 de la sección "Correctness Properties" del diseño,
sobre las funciones puras de ``mlops_core.features`` que conforman el
preprocesamiento determinista usado por el adaptador ``preprocess``:

    fit_preprocessor(df) -> params
    transform(df, params, add_split=True) -> df_procesado

El invariante bajo prueba es que la columna de trazabilidad ``batch_number``
(``BATCH_COLUMN``) se conserva por fila a lo largo de todo el pipeline:

- El **multiconjunto** de valores de ``batch_number`` en la salida es idéntico al
  de la entrada (no se inventa ni se descarta ningún ``batch_number``).
- Cada fila de salida conserva exactamente el ``batch_number`` de su fila de
  origen. Como ``transform`` puede reordenar filas al particionar (``split``
  baraja con ``random_state``), la trazabilidad por fila se verifica a través de
  un identificador único (``row_uid``) que viaja con cada fila.

La prueba es PURA: opera solo sobre ``pandas`` y la lógica de ``mlops_core``, sin
I/O de PostgreSQL ni MLflow.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from mlops_core.features import BATCH_COLUMN, fit_preprocessor, transform

# --- Estrategias de generación ------------------------------------------------

# Valores categóricos para status/city/state: incluye nulos y un vocabulario
# acotado para forzar repeticiones y, ocasionalmente, categorías nuevas.
_categorical = st.one_of(
    st.none(),
    st.sampled_from(["for_sale", "sold", "Austin", "Texas", "NY", "rare_cat", ""]),
    st.text(max_size=8),
)

# Valores numéricos: nulos, enteros y flotantes (con NaN/inf descartados).
_numeric = st.one_of(
    st.none(),
    st.integers(min_value=-1_000, max_value=1_000_000),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
)


@st.composite
def property_frame(draw) -> pd.DataFrame:
    """Genera un ``DataFrame`` inmobiliario con ``batch_number`` y un ``row_uid``.

    Cada fila incluye:

    - ``batch_number``: entero de un rango pequeño para que varias filas
      compartan lote (escenario típico de ingesta por lotes).
    - ``status`` / ``city`` / ``state``: variables categóricas (con desconocidos).
    - columnas numéricas del dominio.
    - ``row_uid``: identificador único por fila que permite rastrear cada fila
      tras el barajado del particionado. No es categórica ni numérica conocida,
      por lo que ``transform`` lo conserva intacto.
    """
    rows: List[Dict[str, Any]] = draw(
        st.lists(
            st.fixed_dictionaries(
                {
                    "batch_number": st.integers(min_value=-5, max_value=20),
                    "status": _categorical,
                    "city": _categorical,
                    "state": _categorical,
                    "brokered_by": _numeric,
                    "bed": _numeric,
                    "bath": _numeric,
                    "acre_lot": _numeric,
                    "street": _numeric,
                    "zip_code": _numeric,
                    "house_size": _numeric,
                    "price": _numeric,
                }
            ),
            min_size=1,
            max_size=40,
        )
    )
    # Identificador único y estable por fila (trazabilidad tras el barajado).
    for index, row in enumerate(rows):
        row["row_uid"] = index
    return pd.DataFrame(rows)


# Feature: mlops-real-estate-platform, Property 5: Preservación de trazabilidad por lote
@settings(max_examples=100)
@given(df=property_frame())
def test_batch_number_preserved_per_row(df: pd.DataFrame) -> None:
    """El ``batch_number`` se conserva por fila a lo largo del preprocesamiento.

    *Para todo* conjunto de registros crudos asociados a un ``batch_number``,
    cada fila procesada resultante conserva el mismo ``batch_number`` de su
    registro de origen: el multiconjunto de ``batch_number`` se preserva (no se
    inventa ni se descarta ninguno) y cada fila mantiene su ``batch_number`` de
    origen pese al reordenamiento del particionado.

    **Validates: Requirements 2.5, 11.1**
    """
    # Mapa de origen: row_uid -> batch_number (antes de procesar).
    expected_by_uid = dict(zip(df["row_uid"], df[BATCH_COLUMN]))

    params = fit_preprocessor(df)
    result = transform(df, params, add_split=True)

    # 1) La columna de trazabilidad sobrevive al pipeline.
    assert BATCH_COLUMN in result.columns
    assert "row_uid" in result.columns

    # 2) El número de filas se conserva (ninguna fila se pierde ni se duplica).
    assert len(result) == len(df)

    # 3) El multiconjunto de batch_number se preserva: no se inventa ni descarta
    #    ningún batch_number (independiente del orden tras el barajado).
    assert Counter(result[BATCH_COLUMN].tolist()) == Counter(df[BATCH_COLUMN].tolist())

    # 4) Trazabilidad por fila: cada fila de salida conserva el batch_number de
    #    su fila de origen, identificada por su row_uid único.
    assert Counter(result["row_uid"].tolist()) == Counter(df["row_uid"].tolist())
    for uid, batch in zip(result["row_uid"], result[BATCH_COLUMN]):
        assert batch == expected_by_uid[uid]
