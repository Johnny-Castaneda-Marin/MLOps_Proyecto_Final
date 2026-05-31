"""Configuración compartida de pytest + Hypothesis para la suite de pruebas.

- Garantiza que la raíz del repositorio esté en ``sys.path`` para que el paquete
  ``mlops_core`` sea importable cuando pytest se ejecuta desde cualquier ruta.
- Registra un perfil de Hypothesis con un mínimo de 100 ejemplos, tal como exige
  la estrategia de pruebas del diseño para las pruebas basadas en propiedades.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Raíz del repositorio (carpeta que contiene ``mlops_core``).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from hypothesis import settings

    # Perfil con >= 100 iteraciones para las pruebas basadas en propiedades.
    settings.register_profile("mlops", max_examples=100)
    settings.load_profile("mlops")
except ImportError:  # pragma: no cover - Hypothesis se instala vía requirements-dev
    pass
