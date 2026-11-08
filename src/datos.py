"""
Carga de imágenes para entrenamiento y evaluación.

Se espera la estructura de carpetas de FER-2013 (Kaggle: msambare/fer2013):

    datos/fer2013/
        train/
            angry/  disgust/  fear/  happy/  neutral/  sad/  surprise/
        test/
            angry/  disgust/  ...

Cada subcarpeta contiene las imágenes de esa emoción. Puedes usar tu propio
dataset siempre que respete la misma estructura (una carpeta por clase).
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def resolver_carpeta(datos: str | Path, subcarpeta: str) -> Path:
    """Devuelve datos/<subcarpeta> si existe; si no, la propia carpeta datos."""
    datos = Path(datos)
    candidata = datos / subcarpeta
    if candidata.is_dir():
        return candidata
    if datos.is_dir():
        return datos
    raise FileNotFoundError(
        f"No se encontró la carpeta de datos '{datos}'. "
        "Descarga FER-2013 (ver README) o indica la ruta con --datos."
    )


def contar_imagenes(carpeta: Path) -> dict[str, int]:
    """Cuenta cuántas imágenes hay por clase (subcarpeta)."""
    conteo = {}
    for clase in sorted(p for p in carpeta.iterdir() if p.is_dir()):
        conteo[clase.name] = sum(
            1 for f in clase.iterdir() if f.suffix.lower() in EXTENSIONES
        )
    return conteo


def pesos_de_clase(conteo: dict[str, int]) -> dict[int, float]:
    """
    Calcula pesos para compensar clases desbalanceadas
    (en FER-2013 'disgust' tiene ~15 veces menos imágenes que 'happy').
    """
    etiquetas = np.concatenate(
        [np.full(n, i) for i, n in enumerate(conteo.values())]
    )
    pesos = compute_class_weight(
        "balanced", classes=np.arange(len(conteo)), y=etiquetas
    )
    return {i: float(p) for i, p in enumerate(pesos)}


def _modo_color(canales: int) -> str:
    return "grayscale" if canales == 1 else "rgb"


def cargar_entrenamiento(
    carpeta: Path,
    img_size: int,
    canales: int,
    batch: int,
    validacion: float,
    semilla: int,
):
    """Crea los datasets de entrenamiento y validación (división automática)."""
    entren, valid = tf.keras.utils.image_dataset_from_directory(
        carpeta,
        validation_split=validacion,
        subset="both",
        seed=semilla,
        image_size=(img_size, img_size),
        color_mode=_modo_color(canales),
        batch_size=batch,
        label_mode="int",
    )
    clases = entren.class_names

    # Con imágenes pequeñas en gris cabe todo en memoria y el entrenamiento
    # va mucho más rápido. Con RGB grande se lee del disco en cada época.
    if canales == 1:
        entren = entren.cache()
        valid = valid.cache()
    entren = entren.shuffle(1000, seed=semilla).prefetch(tf.data.AUTOTUNE)
    valid = valid.prefetch(tf.data.AUTOTUNE)
    return entren, valid, clases


def cargar_prueba(
    carpeta: Path, img_size: int, canales: int, batch: int, clases: list[str]
):
    """Dataset de prueba en orden fijo (sin barajar) para evaluar."""
    return tf.keras.utils.image_dataset_from_directory(
        carpeta,
        class_names=clases,
        shuffle=False,
        image_size=(img_size, img_size),
        color_mode=_modo_color(canales),
        batch_size=batch,
        label_mode="int",
    ).prefetch(tf.data.AUTOTUNE)
