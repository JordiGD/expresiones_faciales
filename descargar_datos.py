"""
Descarga el dataset FER-2013 desde Kaggle y lo deja en datos/fer2013/.

Uso:
    pip install kagglehub
    python descargar_datos.py

Si Kaggle pide autenticación: crea un token en https://www.kaggle.com/settings
(sección "API" -> "Create New Token") y guarda el archivo kaggle.json en
~/.kaggle/ (Linux/Mac) o C:\\Users\\<tu_usuario>\\.kaggle\\ (Windows).

Alternativa manual: descarga el zip desde
https://www.kaggle.com/datasets/msambare/fer2013 y descomprímelo en datos/fer2013/
(deben quedar las carpetas datos/fer2013/train y datos/fer2013/test).
"""

import shutil
from pathlib import Path

from config import CARPETA_DATOS

DESTINO = CARPETA_DATOS / "fer2013"


def main():
    try:
        import kagglehub
    except ImportError:
        raise SystemExit("Falta kagglehub. Instálalo con: pip install kagglehub")

    if (DESTINO / "train").is_dir() and (DESTINO / "test").is_dir():
        print(f"El dataset ya está en {DESTINO}/")
        return

    print("Descargando FER-2013 desde Kaggle (~60 MB)...")
    origen = Path(kagglehub.dataset_download("msambare/fer2013"))

    # Busca la carpeta que contiene train/ y test/
    raiz = next((p for p in [origen, *origen.rglob("*")] if (p / "train").is_dir()), None)
    if raiz is None:
        raise SystemExit(f"No se encontraron las carpetas train/test dentro de {origen}")

    DESTINO.mkdir(parents=True, exist_ok=True)
    for particion in ("train", "test"):
        shutil.copytree(raiz / particion, DESTINO / particion, dirs_exist_ok=True)
    print(f"✔ Dataset listo en {DESTINO}/")


if __name__ == "__main__":
    main()
