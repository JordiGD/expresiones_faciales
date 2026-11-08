"""
Reconocedor de emociones: detecta los rostros de una imagen y clasifica la
emoción de cada uno con el modelo entrenado.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CARPETA_MODELOS, COLOR_POR_DEFECTO, COLORES, SENTIMIENTO  # noqa: E402
from src.detector import DetectorRostros  # noqa: E402


def buscar_modelos(carpeta: Path = CARPETA_MODELOS) -> list[Path]:
    """Modelos entrenados disponibles, del más reciente al más antiguo."""
    if not carpeta.is_dir():
        return []
    modelos = [p for p in carpeta.glob("*.keras") if p.with_suffix(".json").is_file()]
    return sorted(modelos, key=lambda p: p.stat().st_mtime, reverse=True)


def cargar_metadatos(ruta_modelo: str | Path) -> dict:
    ruta_json = Path(ruta_modelo).with_suffix(".json")
    if not ruta_json.is_file():
        raise FileNotFoundError(
            f"Falta {ruta_json.name} junto al modelo (lo crea entrenar.py)."
        )
    return json.loads(ruta_json.read_text(encoding="utf-8"))


class ReconocedorEmociones:
    def __init__(self, ruta_modelo: str | Path | None = None, margen: float = 0.08):
        if ruta_modelo is None:
            disponibles = buscar_modelos()
            if not disponibles:
                raise FileNotFoundError(
                    "No hay modelos en la carpeta 'modelos/'. "
                    "Entrena uno primero con: python entrenar.py"
                )
            ruta_modelo = disponibles[0]

        self.ruta_modelo = Path(ruta_modelo)
        self.modelo = keras.models.load_model(self.ruta_modelo, compile=False)
        meta = cargar_metadatos(self.ruta_modelo)
        self.metadatos = meta
        self.etiquetas: list[str] = meta["etiquetas"]
        self.img_size: int = meta["img_size"]
        self.canales: int = meta["canales"]
        self.margen = margen
        self.detector = DetectorRostros()

    # ------------------------------------------------------------------ #
    def _preparar(self, recorte_bgr: np.ndarray) -> np.ndarray:
        """Convierte un recorte de rostro al formato que espera el modelo."""
        # Siempre se pasa a gris: FER-2013 es en escala de grises, así que
        # las fotos a color se parecen más a los datos de entrenamiento.
        gris = cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2GRAY)
        gris = cv2.resize(gris, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        if self.canales == 1:
            entrada = gris[..., np.newaxis]
        else:
            entrada = np.repeat(gris[..., np.newaxis], 3, axis=-1)
        return entrada.astype("float32")

    def predecir_rostros(self, recortes: list[np.ndarray]) -> np.ndarray:
        """Probabilidades (n_rostros x n_emociones) para recortes ya hechos."""
        lote = np.stack([self._preparar(r) for r in recortes])
        return np.asarray(self.modelo(lote, training=False))

    def _recortar(self, imagen, caja):
        x, y, w, h = caja
        dx, dy = int(w * self.margen), int(h * self.margen)
        alto, ancho = imagen.shape[:2]
        x0, y0 = max(0, x - dx), max(0, y - dy)
        x1, y1 = min(ancho, x + w + dx), min(alto, y + h + dy)
        return imagen[y0:y1, x0:x1]

    def analizar(self, imagen_bgr: np.ndarray, usar_imagen_completa: bool = True):
        """
        Analiza una imagen y devuelve (resultados, se_detecto_rostro).

        Cada resultado es un diccionario con:
            caja, emocion, confianza, sentimiento, probabilidades
        Si no se detecta ningún rostro y usar_imagen_completa=True, se
        clasifica la imagen entera (útil si la foto ya es un rostro recortado).
        """
        cajas = self.detector.detectar(imagen_bgr)
        hay_rostro = bool(cajas)
        if not cajas:
            if not usar_imagen_completa:
                return [], False
            alto, ancho = imagen_bgr.shape[:2]
            cajas = [(0, 0, ancho, alto)]

        recortes = [self._recortar(imagen_bgr, c) for c in cajas]
        probabilidades = self.predecir_rostros(recortes)

        resultados = []
        for caja, probs in zip(cajas, probabilidades):
            i = int(np.argmax(probs))
            emocion = self.etiquetas[i]
            resultados.append(
                {
                    "caja": tuple(int(v) for v in caja),
                    "emocion": emocion,
                    "confianza": float(probs[i]),
                    "sentimiento": SENTIMIENTO.get(emocion, "Desconocido"),
                    "probabilidades": {
                        e: float(p) for e, p in zip(self.etiquetas, probs)
                    },
                }
            )
        return resultados, hay_rostro

    # ------------------------------------------------------------------ #
    @staticmethod
    def dibujar(imagen_bgr: np.ndarray, resultados: list[dict]) -> np.ndarray:
        """Devuelve una copia de la imagen con un recuadro y etiqueta por rostro."""
        salida = imagen_bgr.copy()
        lado = max(salida.shape[:2])
        grosor = max(2, lado // 300)
        escala = max(0.5, lado / 1200)

        for n, r in enumerate(resultados, start=1):
            x, y, w, h = r["caja"]
            color = COLORES.get(r["emocion"], COLOR_POR_DEFECTO)
            cv2.rectangle(salida, (x, y), (x + w, y + h), color, grosor)

            texto = f"{n}. {r['emocion']} {r['confianza']:.0%}"
            (tw, th), base = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)
            y_texto = y - 8 if y - th - 12 > 0 else y + h + th + 8
            cv2.rectangle(
                salida,
                (x, y_texto - th - 6),
                (x + tw + 8, y_texto + base),
                color,
                cv2.FILLED,
            )
            cv2.putText(
                salida, texto, (x + 4, y_texto - 2),
                cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), grosor, cv2.LINE_AA,
            )
        return salida
