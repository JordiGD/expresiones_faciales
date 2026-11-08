"""
Entrena el modelo de reconocimiento de emociones.

Ejemplos:
    python entrenar.py --datos datos/fer2013 --modelo cnn
    python entrenar.py --datos datos/fer2013 --modelo efficientnet --epocas 25
    python entrenar.py --datos datos/fer2013 --modelo mobilenet --batch 32

Al terminar se guardan en la carpeta modelos/:
    modelo_<arquitectura>.keras   -> la red entrenada (mejor época en validación)
    modelo_<arquitectura>.json    -> metadatos (clases, tamaño de imagen, ...)
y en resultados/ las curvas de entrenamiento.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # oculta mensajes técnicos de TensorFlow

import argparse
import json
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import (
    ARQUITECTURAS,
    CARPETA_MODELOS,
    CARPETA_RESULTADOS,
    SEMILLA,
    etiqueta_es,
)
from src import datos, modelos
from src.graficas import graficar_historia


def argumentos():
    p = argparse.ArgumentParser(description="Entrenar el modelo de emociones faciales")
    p.add_argument("--datos", default="datos/fer2013",
                   help="Carpeta del dataset (con subcarpetas train/ y test/)")
    p.add_argument("--modelo", choices=list(ARQUITECTURAS), default="cnn",
                   help="Arquitectura a entrenar")
    p.add_argument("--epocas", type=int, default=50,
                   help="Épocas máximas (se detiene antes si deja de mejorar)")
    p.add_argument("--epocas-ajuste", type=int, default=15,
                   help="Épocas de ajuste fino (solo mobilenet/efficientnet)")
    p.add_argument("--batch", type=int, default=64, help="Tamaño de lote")
    p.add_argument("--lr", type=float, default=None, help="Tasa de aprendizaje inicial")
    p.add_argument("--validacion", type=float, default=0.15,
                   help="Fracción de train usada para validación")
    p.add_argument("--paciencia", type=int, default=8,
                   help="Épocas sin mejorar antes de detener")
    p.add_argument("--sin-imagenet", action="store_true",
                   help="No descargar pesos de ImageNet (entrena la base desde cero)")
    return p.parse_args()


def callbacks(ruta_modelo, ruta_log, paciencia, mejor_previo=None):
    return [
        # Guarda solo cuando mejora la precisión en validación. En el ajuste
        # fino se le pasa la mejor precisión de la fase 1 para no empeorarla.
        keras.callbacks.ModelCheckpoint(
            ruta_modelo, monitor="val_accuracy", save_best_only=True, verbose=1,
            initial_value_threshold=mejor_previo,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=paciencia, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(2, paciencia // 3), min_lr=1e-6, verbose=1
        ),
        keras.callbacks.CSVLogger(ruta_log, append=True),
    ]


def main():
    args = argumentos()
    keras.utils.set_random_seed(SEMILLA)
    conf = ARQUITECTURAS[args.modelo]
    img_size, canales = conf["img_size"], conf["canales"]
    lr = args.lr or conf["lr"]

    carpeta_train = datos.resolver_carpeta(args.datos, "train")
    conteo = datos.contar_imagenes(carpeta_train)
    print("\nImágenes de entrenamiento por clase:")
    for clase, n in conteo.items():
        print(f"  {etiqueta_es(clase):<10} ({clase}): {n}")

    entren, valid, clases = datos.cargar_entrenamiento(
        carpeta_train, img_size, canales, args.batch, args.validacion, SEMILLA
    )
    pesos_clase = datos.pesos_de_clase(conteo)
    print("Pesos por clase (compensan el desbalance):",
          {etiqueta_es(c): round(pesos_clase[i], 2) for i, c in enumerate(clases)})

    CARPETA_MODELOS.mkdir(exist_ok=True)
    CARPETA_RESULTADOS.mkdir(exist_ok=True)
    ruta_modelo = CARPETA_MODELOS / f"modelo_{args.modelo}.keras"
    ruta_log = CARPETA_RESULTADOS / f"entrenamiento_{args.modelo}.csv"
    if ruta_log.exists():
        ruta_log.unlink()

    # Metadatos que necesita el predictor para usar el modelo
    metadatos = {
        "arquitectura": args.modelo,
        "clases": clases,
        "etiquetas": [etiqueta_es(c) for c in clases],
        "img_size": img_size,
        "canales": canales,
        "fecha": datetime.now().isoformat(timespec="seconds"),
    }

    historias = []
    if args.modelo == "cnn":
        modelo = modelos.crear_cnn(len(clases), img_size, canales)
        modelos.compilar(modelo, lr)
        modelo.summary()
        h = modelo.fit(
            entren, validation_data=valid, epochs=args.epocas,
            class_weight=pesos_clase,
            callbacks=callbacks(ruta_modelo, ruta_log, args.paciencia),
        )
        historias.append(h.history)
    else:
        pesos = None if args.sin_imagenet else "imagenet"
        modelo, base = modelos.crear_transfer(args.modelo, len(clases), img_size, pesos)

        # Fase 1: solo se entrena la "cabeza" nueva; la base queda congelada
        print("\n=== Fase 1: entrenando la cabeza del modelo (base congelada) ===")
        modelos.compilar(modelo, lr)
        modelo.summary()
        epocas_fase1 = max(1, min(args.epocas, 15))
        h1 = modelo.fit(
            entren, validation_data=valid, epochs=epocas_fase1,
            class_weight=pesos_clase,
            callbacks=callbacks(ruta_modelo, ruta_log, args.paciencia),
        )
        historias.append(h1.history)

        # Fase 2: ajuste fino de las últimas capas con una tasa más baja
        if args.epocas_ajuste > 0:
            print("\n=== Fase 2: ajuste fino (se descongelan las últimas capas) ===")
            modelos.descongelar(base, capas_entrenables=40)
            modelos.compilar(modelo, lr / 20)
            h2 = modelo.fit(
                entren, validation_data=valid, epochs=args.epocas_ajuste,
                class_weight=pesos_clase,
                callbacks=callbacks(
                    ruta_modelo, ruta_log, args.paciencia,
                    mejor_previo=max(h1.history["val_accuracy"]),
                ),
            )
            historias.append(h2.history)

    # Se recarga la mejor versión guardada durante todo el entrenamiento
    modelo = keras.models.load_model(ruta_modelo)
    perdida, precision = modelo.evaluate(valid, verbose=0)
    metadatos["val_accuracy"] = round(float(precision), 4)
    metadatos["epocas_entrenadas"] = int(sum(len(h["loss"]) for h in historias))
    ruta_modelo.with_suffix(".json").write_text(
        json.dumps(metadatos, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    ruta_curvas = CARPETA_RESULTADOS / f"curvas_{args.modelo}.png"
    graficar_historia(historias, ruta_curvas, f"Entrenamiento – {args.modelo}")

    print("\n✔ Entrenamiento terminado")
    print(f"  Precisión en validación: {precision:.2%}")
    print(f"  Modelo guardado en:      {ruta_modelo}")
    print(f"  Curvas guardadas en:     {ruta_curvas}")
    print(f"\nSiguiente paso: python evaluar.py --modelo {ruta_modelo} --datos {args.datos}")


if __name__ == "__main__":
    # Evita que TensorFlow ocupe toda la memoria de la GPU de golpe
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    np.set_printoptions(precision=3, suppress=True)
    main()
