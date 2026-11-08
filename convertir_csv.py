"""
Convierte la versión CSV de FER-2013 (fer2013.csv) a carpetas de imágenes.

Solo es necesario si descargaste el archivo fer2013.csv (versión original del
reto de Kaggle). Si descargaste la versión con carpetas train/ y test/, no
hace falta.

Uso:
    python convertir_csv.py --csv fer2013.csv --salida datos/fer2013
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

# Códigos numéricos del CSV original
CODIGOS = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "surprise", 6: "neutral"}
# Training -> train ; PublicTest y PrivateTest -> test
PARTICION = {"Training": "train", "PublicTest": "test", "PrivateTest": "test"}


def main():
    p = argparse.ArgumentParser(description="Convertir fer2013.csv a carpetas")
    p.add_argument("--csv", required=True, help="Ruta de fer2013.csv")
    p.add_argument("--salida", default="datos/fer2013")
    args = p.parse_args()

    salida = Path(args.salida)
    for particion in set(PARTICION.values()):
        for emocion in CODIGOS.values():
            (salida / particion / emocion).mkdir(parents=True, exist_ok=True)

    total = 0
    with open(args.csv, newline="", encoding="utf-8") as f:
        for i, fila in enumerate(csv.DictReader(f)):
            pixeles = np.array(fila["pixels"].split(), dtype=np.uint8).reshape(48, 48)
            emocion = CODIGOS[int(fila["emotion"])]
            particion = PARTICION.get(fila.get("Usage", "Training"), "train")
            Image.fromarray(pixeles).save(salida / particion / emocion / f"{i:05d}.png")
            total += 1
            if total % 5000 == 0:
                print(f"  {total} imágenes convertidas...")

    print(f"✔ {total} imágenes guardadas en {salida}/")


if __name__ == "__main__":
    main()
