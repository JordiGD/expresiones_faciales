"""
Lectura y escritura de imágenes.

Se usa Pillow para leer porque:
  - respeta la orientación EXIF de las fotos de celular (si no, salen giradas),
  - funciona con rutas con tildes o "ñ" en Windows (cv2.imread falla con ellas).
"""

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def _pil_a_bgr(imagen: Image.Image) -> np.ndarray:
    imagen = ImageOps.exif_transpose(imagen).convert("RGB")
    return cv2.cvtColor(np.asarray(imagen), cv2.COLOR_RGB2BGR)


def leer_imagen(ruta: str | Path) -> np.ndarray:
    """Lee una imagen del disco y la devuelve en formato BGR (OpenCV)."""
    with Image.open(ruta) as img:
        return _pil_a_bgr(img)


def leer_imagen_bytes(datos: bytes) -> np.ndarray:
    """Lee una imagen a partir de sus bytes (p. ej. subida desde la web)."""
    with Image.open(io.BytesIO(datos)) as img:
        return _pil_a_bgr(img)


def guardar_imagen(ruta: str | Path, imagen_bgr: np.ndarray) -> None:
    """Guarda una imagen BGR (funciona con rutas con caracteres especiales)."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(ruta.suffix or ".jpg", imagen_bgr)
    if not ok:
        raise ValueError(f"No se pudo codificar la imagen como {ruta.suffix}")
    buffer.tofile(str(ruta))
