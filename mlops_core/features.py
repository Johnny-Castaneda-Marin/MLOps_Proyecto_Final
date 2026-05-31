"""Lógica pura de ingeniería de variables (RF3.5, RF5.4, RF11.3, RF2.5).

Este módulo concentra las transformaciones deterministas y sin I/O usadas por el
adaptador ``preprocess`` de Airflow y, eventualmente, por la API de inferencia:

- **Codificación de categóricas con manejo de desconocidos** (RF3.5): se ajusta
  un vocabulario a partir de las categorías vistas en entrenamiento y, al
  transformar, cualquier categoría no vista (o valor nulo) se mapea a un código
  reservado de "otros" sin lanzar excepción.
- **Particionado train/val/test determinista** (RF11.3): con ``random_state=42``
  por defecto, particionar dos veces produce particiones idénticas.
- **Preservación del ``batch_number``** (RF2.5): las transformaciones operan
  sobre copias del ``DataFrame`` y nunca eliminan la columna de trazabilidad.
- **Captura de parámetros de preprocesamiento** (RF5.4): los encoders, las
  medianas de imputación y la configuración de particionado se exponen de forma
  serializable para registrarlos posteriormente en MLflow.

No contiene dependencias de infraestructura (PostgreSQL, MLflow, HTTP), de modo
que estas funciones son deterministas y testeables de forma aislada.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple, Union

import pandas as pd

# --- Constantes de dominio ---------------------------------------------------

#: Semilla determinista usada en todo el preprocesamiento (RF11.3).
DEFAULT_RANDOM_STATE = 42

#: Columna de trazabilidad que debe preservarse de extremo a extremo (RF2.5).
BATCH_COLUMN = "batch_number"

#: Variables categóricas del dominio inmobiliario (RF3.4/3.5).
CATEGORICAL_COLUMNS: Tuple[str, ...] = ("status", "city", "state")

#: Variables numéricas imputadas por mediana en el preprocesamiento.
NUMERIC_COLUMNS: Tuple[str, ...] = (
    "brokered_by",
    "bed",
    "bath",
    "acre_lot",
    "street",
    "zip_code",
    "house_size",
)

#: Fracciones de particionado por defecto (el resto va a ``test``).
DEFAULT_TRAIN_FRAC = 0.70
DEFAULT_VAL_FRAC = 0.15


def _is_missing(value: Any) -> bool:
    """Devuelve ``True`` para valores nulos (``None``, ``NaN``) de forma robusta.

    Tolera valores no escalares (listas, arrays) devolviendo ``False`` en lugar
    de propagar una excepción, lo que mantiene la codificación libre de fallos.
    """

    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


# --- Codificación de categóricas con manejo de desconocidos (RF3.5) ----------


@dataclass(frozen=True)
class CategoricalEncoder:
    """Vocabulario de codificación para una variable categórica.

    Las categorías vistas en entrenamiento reciben códigos enteros ``0..n-1``
    (en orden determinista). Se reserva ``others_code = n`` como índice dedicado
    para cualquier categoría desconocida o valor nulo, de modo que la
    transformación nunca lanza excepción (RF3.5).
    """

    column: str
    categories: Tuple[Any, ...]
    mapping: Dict[Any, int] = field(default_factory=dict)
    others_code: int = 0

    def encode_value(self, value: Any) -> int:
        """Codifica un único valor; los desconocidos/nulos van a ``others_code``."""

        if _is_missing(value):
            return self.others_code
        try:
            return self.mapping.get(value, self.others_code)
        except TypeError:
            # Valor no hasheable: se trata como categoría desconocida.
            return self.others_code

    def encode(self, values: Iterable[Any]) -> List[int]:
        """Codifica un iterable de valores en su lista de códigos enteros."""

        return [self.encode_value(v) for v in values]

    @property
    def n_categories(self) -> int:
        """Número de categorías conocidas (sin contar el código de 'otros')."""

        return len(self.categories)

    def to_params(self) -> Dict[str, Any]:
        """Parámetros serializables del encoder para registrar en MLflow."""

        return {
            f"encoder_{self.column}_n_categories": self.n_categories,
            f"encoder_{self.column}_others_code": self.others_code,
        }


def fit_categorical_encoder(values: Iterable[Any], column: str) -> CategoricalEncoder:
    """Ajusta un :class:`CategoricalEncoder` a partir de categorías de entrenamiento.

    Las categorías nulas se ignoran durante el ajuste; el código de "otros"
    queda reservado en ``len(categorias_unicas)`` (un entero válido y distinto de
    todos los códigos conocidos).
    """

    unique: List[Any] = []
    seen: set = set()
    for value in values:
        if _is_missing(value):
            continue
        try:
            if value in seen:
                continue
            seen.add(value)
        except TypeError:
            # Valor no hasheable: no puede formar parte del vocabulario.
            continue
        unique.append(value)

    ordered = sorted(unique, key=lambda c: str(c))
    mapping = {category: index for index, category in enumerate(ordered)}
    others_code = len(ordered)
    return CategoricalEncoder(
        column=column,
        categories=tuple(ordered),
        mapping=mapping,
        others_code=others_code,
    )


def fit_categorical_encoders(
    data: Union[pd.DataFrame, Mapping[str, Iterable[Any]]],
    columns: Sequence[str] = CATEGORICAL_COLUMNS,
) -> Dict[str, CategoricalEncoder]:
    """Ajusta un encoder por cada columna categórica presente en ``data``."""

    encoders: Dict[str, CategoricalEncoder] = {}
    for column in columns:
        if isinstance(data, pd.DataFrame):
            if column in data.columns:
                encoders[column] = fit_categorical_encoder(data[column], column)
        elif column in data:
            encoders[column] = fit_categorical_encoder(data[column], column)
    return encoders


def encode_categorical(values: Iterable[Any], encoder: CategoricalEncoder) -> List[int]:
    """Codifica un iterable de valores usando ``encoder`` (maneja desconocidos)."""

    return encoder.encode(values)


def transform_categorical(
    df: pd.DataFrame,
    encoders: Mapping[str, CategoricalEncoder],
) -> pd.DataFrame:
    """Aplica los encoders a sus columnas, devolviendo un nuevo ``DataFrame``.

    Las columnas no codificadas (incluida ``batch_number``) se conservan tal
    cual: la operación trabaja sobre una copia y nunca elimina columnas (RF2.5).
    """

    result = df.copy()
    for column, encoder in encoders.items():
        if column in result.columns:
            result[column] = encoder.encode(result[column])
    return result


# --- Particionado determinista (RF11.3) --------------------------------------


def split_dataset(
    df: pd.DataFrame,
    *,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
    random_state: int = DEFAULT_RANDOM_STATE,
    shuffle: bool = True,
) -> pd.DataFrame:
    """Añade una columna ``split`` (``train``/``val``/``test``) de forma determinista.

    Con la misma semilla (``random_state``), particionar dos veces el mismo
    dataset produce particiones idénticas (RF11.3). El resto de columnas,
    incluida ``batch_number``, se preserva (RF2.5).
    """

    if train_frac <= 0 or val_frac < 0 or train_frac + val_frac > 1:
        raise ValueError(
            "Fracciones inválidas: se requiere train_frac > 0, val_frac >= 0 y "
            "train_frac + val_frac <= 1."
        )

    result = df.copy()
    n = len(result)
    if shuffle and n > 0:
        result = result.sample(frac=1, random_state=random_state).reset_index(drop=True)
    else:
        result = result.reset_index(drop=True)

    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    splits = ["train"] * n
    for i in range(train_end, val_end):
        splits[i] = "val"
    for i in range(val_end, n):
        splits[i] = "test"

    result["split"] = splits
    return result


# --- Captura de parámetros de preprocesamiento (RF5.4) -----------------------


@dataclass(frozen=True)
class PreprocessingParams:
    """Artefactos y configuración del preprocesamiento ajustado.

    Agrupa los encoders de categóricas, las medianas de imputación numérica y la
    configuración de particionado. ``to_dict`` produce una representación plana y
    serializable apta para ``mlflow.log_params`` (RF5.4).
    """

    categorical_columns: Tuple[str, ...] = ()
    encoders: Dict[str, CategoricalEncoder] = field(default_factory=dict)
    numeric_columns: Tuple[str, ...] = ()
    numeric_medians: Dict[str, float] = field(default_factory=dict)
    train_frac: float = DEFAULT_TRAIN_FRAC
    val_frac: float = DEFAULT_VAL_FRAC
    random_state: int = DEFAULT_RANDOM_STATE

    @property
    def test_frac(self) -> float:
        return round(1.0 - self.train_frac - self.val_frac, 10)

    def to_dict(self) -> Dict[str, Any]:
        """Parámetros planos y serializables para registrar en MLflow."""

        params: Dict[str, Any] = {
            "categorical_columns": ",".join(self.categorical_columns),
            "numeric_columns": ",".join(self.numeric_columns),
            "split_train_frac": self.train_frac,
            "split_val_frac": self.val_frac,
            "split_test_frac": self.test_frac,
            "random_state": self.random_state,
        }
        for encoder in self.encoders.values():
            params.update(encoder.to_params())
        for column, median in self.numeric_medians.items():
            params[f"impute_{column}_median"] = median
        return params


def fit_preprocessor(
    df: pd.DataFrame,
    *,
    categorical_columns: Sequence[str] = CATEGORICAL_COLUMNS,
    numeric_columns: Sequence[str] = NUMERIC_COLUMNS,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> PreprocessingParams:
    """Ajusta encoders y medianas de imputación y captura la configuración.

    Devuelve un :class:`PreprocessingParams` reutilizable por :func:`transform`
    y registrable en MLflow vía :meth:`PreprocessingParams.to_dict`.
    """

    encoders = fit_categorical_encoders(df, categorical_columns)

    numeric_medians: Dict[str, float] = {}
    for column in numeric_columns:
        if column in df.columns:
            median = pd.to_numeric(df[column], errors="coerce").median()
            numeric_medians[column] = 0.0 if pd.isna(median) else float(median)

    present_categoricals = tuple(c for c in categorical_columns if c in df.columns)
    present_numerics = tuple(c for c in numeric_columns if c in df.columns)
    return PreprocessingParams(
        categorical_columns=present_categoricals,
        encoders=encoders,
        numeric_columns=present_numerics,
        numeric_medians=numeric_medians,
        train_frac=train_frac,
        val_frac=val_frac,
        random_state=random_state,
    )


def transform(
    df: pd.DataFrame,
    params: PreprocessingParams,
    *,
    add_split: bool = True,
) -> pd.DataFrame:
    """Aplica el preprocesamiento ajustado a ``df``.

    Imputa numéricas con las medianas capturadas, codifica categóricas con manejo
    de desconocidos y, opcionalmente, añade la partición determinista. La columna
    ``batch_number`` se conserva intacta (RF2.5).
    """

    result = df.copy()

    for column, median in params.numeric_medians.items():
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce").fillna(median)

    result = transform_categorical(result, params.encoders)

    if add_split:
        result = split_dataset(
            result,
            train_frac=params.train_frac,
            val_frac=params.val_frac,
            random_state=params.random_state,
        )

    return result


def capture_preprocessing_params(
    encoders: Mapping[str, CategoricalEncoder],
    *,
    numeric_medians: Mapping[str, float] | None = None,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Dict[str, Any]:
    """Construye el diccionario plano de parámetros para MLflow a partir de encoders.

    Útil cuando se dispone de encoders y medianas sueltos y no de un
    :class:`PreprocessingParams` completo.
    """

    numeric_medians = dict(numeric_medians or {})
    params = PreprocessingParams(
        categorical_columns=tuple(encoders.keys()),
        encoders=dict(encoders),
        numeric_columns=tuple(numeric_medians.keys()),
        numeric_medians=numeric_medians,
        train_frac=train_frac,
        val_frac=val_frac,
        random_state=random_state,
    )
    return params.to_dict()
