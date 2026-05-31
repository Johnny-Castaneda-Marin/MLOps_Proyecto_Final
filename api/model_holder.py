"""``ModelHolder`` thread-safe para la API de inferencia (RF7.5, RF7.6, RF7.7).

Este módulo es deliberadamente **ligero**: no importa FastAPI ni mlflow. El
``ModelHolder`` solo conoce un "modelo" como un objeto con método ``predict`` y
una "versión" como una cadena. La carga concreta desde MLflow se inyecta como
un callable (ver ``api.reloader``), de modo que el holder pueda ejercitarse en
pruebas (incluida la Property 15) sin una conexión real a MLflow.

Garantías de consistencia (Property 15 — "Consistencia del modelo servido bajo
recarga y fallo"):

- Una vez inicializado, el holder **siempre** expone un modelo cargado y su
  versión; nunca un estado nulo o parcial.
- El intercambio de modelo (``update``) es **atómico** bajo un lock: una
  predicción concurrente observa el modelo previo completo o el nuevo modelo
  completo, jamás un estado intermedio.
- Ante un intento de actualización inválido (modelo ``None`` o versión vacía)
  el holder **conserva** el último estado válido y lanza, de modo que un
  reloader pueda capturar el fallo y mantener el modelo previo (RF7.6).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Optional, Tuple


class ModelNotInitializedError(RuntimeError):
    """Se lanza al usar el holder antes de cargar un primer modelo válido."""


class InvalidModelError(ValueError):
    """Se lanza cuando se intenta instalar un modelo nulo o sin versión.

    Capturarla en el reloader permite conservar el modelo previo (RF7.6) sin
    dejar nunca el holder en un estado parcial.
    """


@dataclass(frozen=True)
class LoadedModel:
    """Snapshot inmutable del modelo servido y su versión.

    Al ser inmutable y referenciado por una sola variable dentro del holder, el
    intercambio de modelo se reduce a reasignar esta referencia bajo lock, lo
    que hace el swap atómico desde el punto de vista de los lectores.
    """

    model: Any
    version: str


class ModelHolder:
    """Contenedor thread-safe del modelo de producción servido por la API.

    El holder mantiene un único ``LoadedModel`` protegido por un ``RLock``. Los
    lectores (``predict``, ``current``, ``model``, ``version``) toman un snapshot
    bajo el lock y operan sobre la referencia inmutable; los escritores
    (``update``) reasignan la referencia bajo el mismo lock. Esto cumple:

    - RF7.5: durante una recarga se sigue sirviendo el modelo previo hasta que
      el nuevo está completamente listo (el swap ocurre en una sola asignación).
    - RF7.6: un intento de instalar un modelo inválido no reemplaza el actual.
    - RF7.7: las solicitudes concurrentes durante la recarga no observan
      estados corruptos ni nulos.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state: Optional[LoadedModel] = None

    # -- Estado / introspección ------------------------------------------------

    @property
    def is_initialized(self) -> bool:
        """``True`` si ya se cargó al menos un modelo válido."""
        with self._lock:
            return self._state is not None

    @property
    def version(self) -> Optional[str]:
        """Versión del modelo actualmente servido, o ``None`` si no hay modelo."""
        with self._lock:
            return self._state.version if self._state is not None else None

    @property
    def model(self) -> Any:
        """Modelo actualmente servido.

        Raises:
            ModelNotInitializedError: si aún no se ha cargado un modelo válido.
        """
        with self._lock:
            self._require_initialized()
            return self._state.model  # type: ignore[union-attr]

    def current(self) -> LoadedModel:
        """Devuelve un snapshot inmutable ``(model, version)`` de forma atómica.

        Raises:
            ModelNotInitializedError: si aún no se ha cargado un modelo válido.
        """
        with self._lock:
            self._require_initialized()
            return self._state  # type: ignore[return-value]

    # -- Mutación atómica ------------------------------------------------------

    def update(self, model: Any, version: str) -> LoadedModel:
        """Instala atómicamente un nuevo modelo + versión (swap bajo lock).

        Valida que el modelo no sea ``None`` y que la versión sea una cadena no
        vacía **antes** de reemplazar el estado, de modo que un swap inválido
        nunca deje el holder en un estado parcial: si la validación falla, el
        estado previo se conserva intacto.

        Args:
            model: Objeto de modelo con método ``predict``.
            version: Identificador de versión (cadena no vacía).

        Returns:
            El ``LoadedModel`` recién instalado.

        Raises:
            InvalidModelError: si ``model`` es ``None`` o ``version`` es vacía.
        """
        if model is None:
            raise InvalidModelError("El modelo a instalar no puede ser None.")
        if version is None or str(version).strip() == "":
            raise InvalidModelError("La versión del modelo no puede ser vacía.")

        new_state = LoadedModel(model=model, version=str(version))
        with self._lock:
            # Reasignación de referencia bajo lock: swap atómico (RF7.5/7.7).
            self._state = new_state
        return new_state

    # -- Inferencia ------------------------------------------------------------

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """Ejecuta ``predict`` con el modelo actualmente servido (RF7.7).

        Toma un snapshot del modelo bajo el lock y libera el lock antes de
        ejecutar la predicción, de modo que una recarga concurrente no quede
        bloqueada por inferencias largas y la inferencia use siempre un modelo
        completo y coherente.

        Raises:
            ModelNotInitializedError: si aún no se ha cargado un modelo válido.
        """
        with self._lock:
            self._require_initialized()
            model = self._state.model  # type: ignore[union-attr]
        return model.predict(*args, **kwargs)

    def predict_with_version(self, *args: Any, **kwargs: Any) -> Tuple[Any, str]:
        """Como ``predict`` pero devuelve también la versión usada (RF7 / RF8.2).

        Captura modelo y versión en el **mismo** snapshot atómico, garantizando
        que la versión reportada corresponde exactamente al modelo que produjo
        la predicción aunque ocurra una recarga concurrente.
        """
        with self._lock:
            self._require_initialized()
            snapshot = self._state  # type: ignore[assignment]
        prediction = snapshot.model.predict(*args, **kwargs)  # type: ignore[union-attr]
        return prediction, snapshot.version  # type: ignore[union-attr]

    # -- Internos --------------------------------------------------------------

    def _require_initialized(self) -> None:
        if self._state is None:
            raise ModelNotInitializedError(
                "El ModelHolder no tiene un modelo cargado todavía."
            )
