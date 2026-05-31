"""Prueba basada en propiedades para la independencia de la decisión de
entrenamiento respecto a la periodicidad y el conteo bruto de lotes.

Implementa la Property 12 de la sección "Correctness Properties" del diseño,
sobre la función pura ``mlops_core.decision.decide_training`` y el tipo
``mlops_core.types.DecisionInputs`` (tarea 5.3). Valida el requisito RF4.3.

Idea central: ``DecisionInputs`` (``total_records``, ``inserted_records``,
``drift_detected``, ``quality_valid``, ``new_categories_count``) NO contiene
``batch_number`` ni fecha/periodicidad de ejecución. Por construcción, la
función pura no puede depender de ellos. La prueba fija las señales técnicas en
un ``DecisionInputs`` y confirma que la decisión calculada sobre ESE MISMO
``DecisionInputs`` es idéntica con independencia de cualquier ``batch_number`` o
``execution_date`` arbitrario que se sortee (estos valores no son entradas de la
función). Se ejercita la estabilidad/determinismo de la decisión a lo largo de
muchos valores arbitrarios de ``batch_number``/fecha que NO son insumos.
"""

from __future__ import annotations

from datetime import datetime

from hypothesis import given, settings
from hypothesis import strategies as st

from mlops_core.decision import decide_training
from mlops_core.types import DecisionInputs


@st.composite
def decision_inputs(draw):
    """Genera ``DecisionInputs`` con señales técnicas válidas y variadas.

    Cubre tanto la rama de datos insuficientes (``total_records`` pequeño) como
    la de datos suficientes, así como drift, volumen y nuevas categorías, para
    que la propiedad se ejercite sobre decisiones TRAIN y SKIP por igual.
    ``inserted_records`` se acota a ``[0, total_records]`` para mantener la
    proporción de volumen en un rango realista.
    """
    total_records = draw(st.integers(min_value=0, max_value=100_000))
    inserted_records = draw(st.integers(min_value=0, max_value=max(total_records, 0)))
    return DecisionInputs(
        total_records=total_records,
        inserted_records=inserted_records,
        drift_detected=draw(st.booleans()),
        quality_valid=draw(st.booleans()),
        new_categories_count=draw(st.integers(min_value=0, max_value=50)),
    )


# Estrategia de fechas/periodicidades arbitrarias que NO son entradas de la
# función: distintas marcas de tiempo de ejecución del DAG.
execution_dates = st.datetimes(
    min_value=datetime(2000, 1, 1), max_value=datetime(2100, 12, 31)
)

# Estrategia de ``batch_number`` arbitrarios (incluye el centinela ``-1`` usado
# cuando los datos están agotados).
batch_numbers = st.lists(
    st.integers(min_value=-1, max_value=1_000_000), min_size=1, max_size=10
)


# Feature: mlops-real-estate-platform, Property 12: Independencia de periodicidad y conteo bruto de lotes
@settings(max_examples=100)
@given(
    inp=decision_inputs(),
    batch_values=batch_numbers,
    dates=st.lists(execution_dates, min_size=1, max_size=10),
)
def test_property_independencia_periodicidad_y_conteo_bruto(
    inp: DecisionInputs,
    batch_values: list[int],
    dates: list[datetime],
) -> None:
    """Para todo ``DecisionInputs`` con señales técnicas fijas, variar el
    ``batch_number`` o la fecha/periodicidad de ejecución no altera la decisión
    de entrenamiento.

    Como la firma de ``decide_training`` no admite ``batch_number`` ni fecha, la
    única forma de "variarlos" es no pasarlos: se verifica que la decisión sobre
    el MISMO ``DecisionInputs`` es estable y determinista a lo largo de muchos
    valores arbitrarios de ``batch_number``/fecha que no son insumos.
    """
    # Decisión de referencia a partir de las señales técnicas fijas.
    reference = decide_training(inp)

    # Recorrer un producto de valores arbitrarios de batch_number y fecha; la
    # decisión calculada sobre el MISMO DecisionInputs debe ser idéntica en todos
    # los casos (mismo tipo y mismo motivo).
    for batch_number in batch_values:
        for execution_date in dates:
            # batch_number / execution_date son deliberadamente NO usados como
            # entradas: la función pura no puede depender de ellos.
            result = decide_training(inp)

            assert result.decision == reference.decision, (
                "La decisión cambió al variar batch_number/fecha "
                f"(batch_number={batch_number}, execution_date={execution_date})."
            )
            assert result.reason == reference.reason, (
                "El motivo cambió al variar batch_number/fecha "
                f"(batch_number={batch_number}, execution_date={execution_date})."
            )
