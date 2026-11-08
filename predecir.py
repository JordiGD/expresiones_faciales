"""
Detecta la emoción de los rostros en fotos o en tiempo real con la cámara.

Ejemplos:
    python predecir.py --imagen mi_foto.jpg
    python predecir.py --carpeta fotos/
    python predecir.py --webcam
    python predecir.py --imagen foto.jpg --modelo modelos/modelo_efficientnet.keras

Las imágenes con los rostros marcados se guardan en resultados/predicciones/.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # oculta mensajes técnicos de TensorFlow

import argparse
from pathlib import Path

import cv2

from config import CARPETA_RESULTADOS
from src.imagenes import guardar_imagen, leer_imagen
from src.predictor import ReconocedorEmociones

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def imprimir(nombre: str, resultados: list[dict], hay_rostro: bool) -> None:
    print(f"\n📷 {nombre}")
    if not resultados:
        print("   No se detectaron rostros.")
        return
    if not hay_rostro:
        print("   (No se detectó un rostro: se analizó la imagen completa)")
    for n, r in enumerate(resultados, start=1):
        top3 = sorted(r["probabilidades"].items(), key=lambda kv: kv[1], reverse=True)[:3]
        otras = ", ".join(f"{e} {p:.0%}" for e, p in top3[1:])
        print(f"   Rostro {n}: {r['emocion']} ({r['confianza']:.0%}) "
              f"→ sentimiento {r['sentimiento']}   [luego: {otras}]")


def analizar_archivos(rec: ReconocedorEmociones, rutas: list[Path], completa: bool) -> None:
    salida = CARPETA_RESULTADOS / "predicciones"
    for ruta in rutas:
        try:
            imagen = leer_imagen(ruta)
        except Exception as e:  # imagen corrupta o formato no soportado
            print(f"\n⚠ No se pudo leer {ruta}: {e}")
            continue
        resultados, hay_rostro = rec.analizar(imagen, usar_imagen_completa=completa)
        imprimir(ruta.name, resultados, hay_rostro)
        destino = salida / f"{ruta.stem}_emociones.jpg"
        guardar_imagen(destino, rec.dibujar(imagen, resultados))
        print(f"   Guardada en {destino}")


def webcam(rec: ReconocedorEmociones, camara: int) -> None:
    captura = cv2.VideoCapture(camara)
    if not captura.isOpened():
        raise SystemExit(f"No se pudo abrir la cámara {camara}.")
    print("Cámara activa. Pulsa 'q' para salir o 'g' para guardar una captura.")
    n_captura = 0
    try:
        while True:
            ok, cuadro = captura.read()
            if not ok:
                break
            cuadro = cv2.flip(cuadro, 1)  # efecto espejo, más natural
            resultados, _ = rec.analizar(cuadro, usar_imagen_completa=False)
            anotado = rec.dibujar(cuadro, resultados)
            cv2.imshow("Emociones (q = salir, g = guardar)", anotado)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            if tecla == ord("g"):
                n_captura += 1
                destino = CARPETA_RESULTADOS / "predicciones" / f"webcam_{n_captura}.jpg"
                guardar_imagen(destino, anotado)
                print(f"Captura guardada en {destino}")
    finally:
        captura.release()
        cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description="Reconocer emociones en imágenes o con la cámara")
    grupo = p.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--imagen", help="Ruta de una imagen")
    grupo.add_argument("--carpeta", help="Carpeta con varias imágenes")
    grupo.add_argument("--webcam", action="store_true", help="Usar la cámara en tiempo real")
    p.add_argument("--modelo", default=None,
                   help="Ruta del .keras (por defecto, el más reciente en modelos/)")
    p.add_argument("--camara", type=int, default=0, help="Índice de la cámara (0, 1, ...)")
    p.add_argument("--solo-rostros", action="store_true",
                   help="Si no hay rostro, no analizar la imagen completa")
    args = p.parse_args()

    rec = ReconocedorEmociones(args.modelo)
    print(f"Modelo: {rec.ruta_modelo.name}  |  Emociones: {', '.join(rec.etiquetas)}")

    if args.webcam:
        webcam(rec, args.camara)
    elif args.imagen:
        analizar_archivos(rec, [Path(args.imagen)], not args.solo_rostros)
    else:
        rutas = sorted(f for f in Path(args.carpeta).iterdir() if f.suffix.lower() in EXTENSIONES)
        if not rutas:
            raise SystemExit(f"No hay imágenes en {args.carpeta}")
        analizar_archivos(rec, rutas, not args.solo_rostros)


if __name__ == "__main__":
    main()
