"""Lógica pura de ingesta incremental (RF1).

Funciones puras, sin dependencias de infraestructura, que encapsulan las reglas
de negocio de la ingesta incremental de lotes:

- ``select_day(batch_index)``: día correspondiente al índice acumulado.
- ``is_exhausted(batch_index, http_status)``: condición de agotamiento.
- ``row_hash(record)``: hash MD5 estable sobre el contenido del registro.
- ``deduplicate(records, existing_hashes)``: exclusión de registros repetidos.

Los adaptadores de Airflow (``fetch_batch``, ``store_raw``) delegan en estas
funciones para que la lógica sea testeable de forma determinista (incluyendo
pruebas basadas en propiedades) sin levantar PostgreSQL ni la Data_API.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Set, Tuple

# Días disponibles para la ingesta incremental. El índice de lote acumulado
# selecciona el día; cuando el índice alcanza ``len(DAYS)`` los datos se
# consideran agotados (RF1.1, RF1.6).
DAYS: List[str] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

# Código HTTP con el que la Data_API señala el fin de los datos (agotamiento).
EXHAUSTION_HTTP_STATUS = 400


def select_day(batch_index: int, days: List[str] = DAYS) -> Optional[str]:
    """Devuelve el día correspondiente al índice de lote acumulado.

    Cuando ``batch_index`` está dentro del rango de días disponibles devuelve
    ``days[batch_index]``; cuando lo iguala o supera devuelve ``None`` (datos
    agotados) sin lanzar excepción (RF1.1, RF1.6).
    """
    if batch_index < 0:
        return None
    if batch_index < len(days):
        return days[batch_index]
    return None


def is_exhausted(
    batch_index: int,
    http_status: Optional[int],
    days: List[str] = DAYS,
) -> bool:
    """Indica si la fuente de datos está agotada (RF1.6).

    Devuelve ``True`` si y solo si el índice de lote acumulado alcanza el número
    de días disponibles (``batch_index >= len(days)``) o la Data_API señala el
    fin de los datos con un HTTP 400 de agotamiento.
    """
    if batch_index >= len(days):
        return True
    return http_status == EXHAUSTION_HTTP_STATUS


def row_hash(record: Dict) -> str:
    """Calcula un hash MD5 estable sobre el contenido completo del registro.

    Usa ``json.dumps`` con ``sort_keys=True`` para que el hash sea independiente
    del orden de las claves del diccionario (RF1.4). ``default=str`` garantiza
    que valores no serializables de forma nativa (p. ej. fechas) no interrumpan
    el cálculo.
    """
    serialized = json.dumps(record, sort_keys=True, default=str)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()


def deduplicate(
    records: List[Dict],
    existing_hashes: Set[str],
) -> Tuple[List[Dict], List[str]]:
    """Excluye registros ya vistos según su ``row_hash`` (RF1.4).

    Calcula el ``row_hash`` de cada registro y devuelve una tupla
    ``(registros_nuevos, hashes_nuevos)`` que excluye:

    - cualquier registro cuyo hash ya esté presente en ``existing_hashes``, y
    - duplicados dentro del propio lote de entrada (solo se conserva la primera
      aparición de cada hash).

    El orden relativo de los registros nuevos se preserva.
    """
    seen: Set[str] = set(existing_hashes)
    new_records: List[Dict] = []
    new_hashes: List[str] = []

    for record in records:
        h = row_hash(record)
        if h in seen:
            continue
        seen.add(h)
        new_records.append(record)
        new_hashes.append(h)

    return new_records, new_hashes
