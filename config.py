"""
Configuración general del proyecto de reconocimiento de emociones faciales.

Aquí se definen las emociones, su traducción al español, el "sentimiento"
(polaridad) asociado a cada una y los parámetros de cada arquitectura.
"""

from pathlib import Path

# Carpetas del proyecto
RAIZ = Path(__file__).resolve().parent
CARPETA_MODELOS = RAIZ / "modelos"
CARPETA_RESULTADOS = RAIZ / "resultados"
CARPETA_DATOS = RAIZ / "datos"

# Nombre de las carpetas de FER-2013 -> etiqueta en español.
# Si usas tu propio dataset con carpetas en español (p. ej. "felicidad"),
# también funciona: se usará el nombre de la carpeta con mayúscula inicial.
EMOCIONES = {
    "angry": "Enojo",
    "disgust": "Asco",
    "fear": "Miedo",
    "happy": "Felicidad",
    "neutral": "Neutral",
    "sad": "Tristeza",
    "surprise": "Sorpresa",
}

# Polaridad del sentimiento asociada a cada emoción.
# La sorpresa es ambigua (puede ser buena o mala), por eso queda como "Neutral".
SENTIMIENTO = {
    "Felicidad": "Positivo",
    "Sorpresa": "Neutral",
    "Neutral": "Neutral",
    "Tristeza": "Negativo",
    "Enojo": "Negativo",
    "Miedo": "Negativo",
    "Asco": "Negativo",
}

# Colores (formato BGR de OpenCV) para dibujar el recuadro de cada emoción
COLORES = {
    "Felicidad": (0, 200, 0),
    "Sorpresa": (0, 200, 255),
    "Neutral": (200, 200, 200),
    "Tristeza": (255, 120, 0),
    "Enojo": (0, 0, 255),
    "Miedo": (180, 0, 180),
    "Asco": (0, 120, 60),
}
COLOR_POR_DEFECTO = (255, 255, 255)

# Parámetros por arquitectura:
#   img_size -> tamaño (px) al que se redimensiona cada rostro
#   canales  -> 1 = escala de grises, 3 = RGB (los modelos preentrenados lo requieren)
ARQUITECTURAS = {
    "cnn": {"img_size": 48, "canales": 1, "lr": 1e-3},
    "mobilenet": {"img_size": 96, "canales": 3, "lr": 1e-3},
    "efficientnet": {"img_size": 112, "canales": 3, "lr": 1e-3},
}

SEMILLA = 123


def etiqueta_es(nombre_clase: str) -> str:
    """Traduce el nombre de una carpeta de clase a su etiqueta en español."""
    return EMOCIONES.get(nombre_clase.lower(), nombre_clase.capitalize())
