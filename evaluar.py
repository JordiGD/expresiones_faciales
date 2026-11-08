"""
Evalúa un modelo entrenado con el conjunto de prueba (imágenes que nunca vio).

Ejemplo:
    python evaluar.py --modelo modelos/modelo_cnn.keras --datos datos/fer2013

Genera en resultados/:
    reporte_<modelo>.txt            -> precisión, recall y F1 por emoción
    matriz_confusion_<modelo>.png   -> con qué emociones se confunde cada una
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # oculta mensajes técnicos de TensorFlow

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow import keras

from config import CARPETA_RESULTADOS
from src import datos
from src.graficas import graficar_matriz_confusion
from src.predictor import buscar_modelos, cargar_metadatos


def main():
    p = argparse.ArgumentParser(description="Evaluar el modelo con el conjunto de prueba")
    p.add_argument("--modelo", default=None,
                   help="Ruta del .keras (por defecto, el más reciente en modelos/)")
    p.add_argument("--datos", default="datos/fer2013",
                   help="Carpeta del dataset (con subcarpeta test/)")
    p.add_argument("--batch", type=int, default=128)
    args = p.parse_args()

    if args.modelo is None:
        disponibles = buscar_modelos()
        if not disponibles:
            raise SystemExit("No hay modelos entrenados en modelos/. Ejecuta entrenar.py primero.")
        args.modelo = disponibles[0]
    ruta_modelo = Path(args.modelo)

    meta = cargar_metadatos(ruta_modelo)
    modelo = keras.models.load_model(ruta_modelo, compile=False)
    carpeta_test = datos.resolver_carpeta(args.datos, "test")
    prueba = datos.cargar_prueba(
        carpeta_test, meta["img_size"], meta["canales"], args.batch, meta["clases"]
    )

    print(f"Evaluando {ruta_modelo.name} con {carpeta_test} ...")
    y_real = np.concatenate([y.numpy() for _, y in prueba])
    y_pred = np.argmax(modelo.predict(prueba, verbose=1), axis=1)

    etiquetas = meta["etiquetas"]
    precision = accuracy_score(y_real, y_pred)
    reporte = classification_report(
        y_real, y_pred, labels=range(len(etiquetas)),
        target_names=etiquetas, digits=3, zero_division=0,
    )
    matriz = confusion_matrix(y_real, y_pred, labels=range(len(etiquetas)))

    nombre = ruta_modelo.stem
    CARPETA_RESULTADOS.mkdir(exist_ok=True)
    texto = (
        f"Modelo: {ruta_modelo.name}\n"
        f"Imágenes de prueba: {len(y_real)}\n"
        f"Precisión global (accuracy): {precision:.2%}\n\n"
        f"{reporte}\n"
        "Matriz de confusión (filas = real, columnas = predicción):\n"
        f"{'':>10} " + " ".join(f"{e[:6]:>6}" for e in etiquetas) + "\n"
        + "\n".join(
            f"{e:>10} " + " ".join(f"{v:>6}" for v in fila)
            for e, fila in zip(etiquetas, matriz)
        )
    )
    (CARPETA_RESULTADOS / f"reporte_{nombre}.txt").write_text(texto, encoding="utf-8")
    graficar_matriz_confusion(
        matriz, etiquetas,
        CARPETA_RESULTADOS / f"matriz_confusion_{nombre}.png",
        f"Matriz de confusión – {nombre} (accuracy {precision:.1%})",
    )

    print("\n" + texto)
    print(f"\n✔ Reporte y matriz de confusión guardados en {CARPETA_RESULTADOS}/")


if __name__ == "__main__":
    main()
