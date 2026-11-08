"""
Detección de rostros con el clasificador Haar Cascade de OpenCV.

El modelo de emociones se entrenó con rostros recortados, así que antes de
clasificar hay que encontrar dónde está cada cara en la foto.
"""

import urllib.request
from pathlib import Path

import cv2

NOMBRE_HAAR = "haarcascade_frontalface_default.xml"
URL_HAAR = (
    "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/"
    + NOMBRE_HAAR
)
COPIA_LOCAL = Path(__file__).resolve().parent.parent / "modelos" / NOMBRE_HAAR


def _ruta_haar() -> Path:
    """Busca el archivo del detector; si no está, lo descarga una sola vez."""
    candidatas = []
    if hasattr(cv2, "data"):
        candidatas.append(Path(cv2.data.haarcascades) / NOMBRE_HAAR)
    candidatas.append(COPIA_LOCAL)
    for ruta in candidatas:
        if ruta.is_file():
            return ruta

    print(f"Descargando el detector de rostros en {COPIA_LOCAL} ...")
    COPIA_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL_HAAR, COPIA_LOCAL)
    return COPIA_LOCAL


class DetectorRostros:
    def __init__(self, tam_minimo: int = 40, lado_max_deteccion: int = 1280):
        self.clasificador = cv2.CascadeClassifier(str(_ruta_haar()))
        if self.clasificador.empty():
            raise RuntimeError("No se pudo cargar el detector de rostros de OpenCV.")
        self.tam_minimo = tam_minimo
        self.lado_max = lado_max_deteccion

    def detectar(self, imagen_bgr) -> list[tuple[int, int, int, int]]:
        """Devuelve una lista de rostros (x, y, ancho, alto), del más grande al más pequeño."""
        gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)

        # Las fotos de celular son enormes: se reducen para detectar más rápido
        alto, ancho = gris.shape
        escala = min(1.0, self.lado_max / max(alto, ancho))
        if escala < 1.0:
            gris = cv2.resize(gris, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)

        gris = cv2.equalizeHist(gris)
        caras = self.clasificador.detectMultiScale(
            gris,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.tam_minimo, self.tam_minimo),
        )
        cajas = [tuple(int(round(v / escala)) for v in c) for c in caras]
        return sorted(cajas, key=lambda c: c[2] * c[3], reverse=True)
