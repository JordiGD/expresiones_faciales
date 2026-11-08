"""Gráficas de entrenamiento y evaluación."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # permite guardar gráficas sin pantalla (servidores, Colab)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def graficar_historia(historias: list[dict], ruta: Path, titulo: str) -> None:
    """Curvas de precisión y pérdida (une las fases de entrenamiento)."""
    union = {}
    for h in historias:
        for clave, valores in h.items():
            union.setdefault(clave, []).extend(valores)

    epocas = np.arange(1, len(union["loss"]) + 1)
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
    for eje, metrica, nombre in [
        (ejes[0], "accuracy", "Precisión (accuracy)"),
        (ejes[1], "loss", "Pérdida (loss)"),
    ]:
        eje.plot(epocas, union[metrica], label="Entrenamiento")
        eje.plot(epocas, union[f"val_{metrica}"], label="Validación")
        eje.set_title(nombre)
        eje.set_xlabel("Época")
        eje.grid(alpha=0.3)
        eje.legend()

    if len(historias) > 1:
        corte = len(historias[0]["loss"]) + 0.5
        for eje in ejes:
            eje.axvline(corte, color="gray", linestyle="--", linewidth=1)
            eje.text(corte, eje.get_ylim()[1], " ajuste fino", va="top", color="gray")

    fig.suptitle(titulo)
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def graficar_matriz_confusion(matriz: np.ndarray, etiquetas: list[str], ruta: Path, titulo: str) -> None:
    """Matriz de confusión normalizada por fila (qué % de cada emoción real se predijo como cada clase)."""
    normal = matriz / np.maximum(matriz.sum(axis=1, keepdims=True), 1)
    fig, eje = plt.subplots(figsize=(8, 7))
    im = eje.imshow(normal, cmap="Blues", vmin=0, vmax=1)
    eje.set_xticks(range(len(etiquetas)), etiquetas, rotation=45, ha="right")
    eje.set_yticks(range(len(etiquetas)), etiquetas)
    eje.set_xlabel("Predicción del modelo")
    eje.set_ylabel("Emoción real")
    for i in range(len(etiquetas)):
        for j in range(len(etiquetas)):
            valor = normal[i, j]
            eje.text(
                j, i, f"{valor:.0%}\n({matriz[i, j]})",
                ha="center", va="center", fontsize=8,
                color="white" if valor > 0.5 else "black",
            )
    fig.colorbar(im, ax=eje, fraction=0.046)
    eje.set_title(titulo)
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
