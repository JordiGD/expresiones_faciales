"""
Aplicación web para reconocer emociones: sube una imagen o toma una foto.

Ejecutar con:
    streamlit run app.py
Se abrirá en el navegador (normalmente http://localhost:8501).
También funciona desde el celular si está en la misma red Wi-Fi
(usa la dirección "Network URL" que muestra Streamlit).
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # oculta mensajes técnicos de TensorFlow

import cv2
import streamlit as st

from config import SENTIMIENTO
from src.imagenes import leer_imagen_bytes
from src.predictor import ReconocedorEmociones, buscar_modelos

EMOJIS = {
    "Felicidad": "😄", "Tristeza": "😢", "Enojo": "😠", "Miedo": "😨",
    "Sorpresa": "😮", "Asco": "🤢", "Neutral": "😐",
}
COLOR_SENTIMIENTO = {"Positivo": "green", "Negativo": "red", "Neutral": "gray"}

st.set_page_config(page_title="Detector de emociones", page_icon="🙂", layout="wide")


@st.cache_resource(show_spinner="Cargando el modelo...")
def cargar_reconocedor(ruta: str) -> ReconocedorEmociones:
    return ReconocedorEmociones(ruta)


def barra_lateral():
    st.sidebar.header("⚙️ Configuración")
    modelos = buscar_modelos()
    if not modelos:
        st.sidebar.error("No hay modelos entrenados.")
        st.error(
            "No se encontró ningún modelo en la carpeta `modelos/`.\n\n"
            "Entrena uno primero, por ejemplo:\n\n"
            "`python entrenar.py --datos datos/fer2013 --modelo cnn`"
        )
        st.stop()

    ruta = st.sidebar.selectbox("Modelo", modelos, format_func=lambda p: p.name)
    completa = st.sidebar.checkbox(
        "Si no hay rostro, analizar la imagen completa", value=True,
        help="Útil cuando la foto ya es un rostro recortado de cerca.",
    )
    rec = cargar_reconocedor(str(ruta))

    meta = rec.metadatos
    st.sidebar.caption(
        f"Arquitectura: **{meta['arquitectura']}** · entrada {meta['img_size']}px\n\n"
        f"Precisión en validación: **{meta.get('val_accuracy', 0):.1%}**"
    )
    st.sidebar.markdown("**Emociones que reconoce**")
    st.sidebar.markdown(
        "\n".join(
            f"- {EMOJIS.get(e, '')} {e} · _{SENTIMIENTO.get(e, '—')}_" for e in rec.etiquetas
        )
    )
    return rec, completa


def mostrar_resultados(rec: ReconocedorEmociones, datos: bytes, completa: bool) -> None:
    try:
        imagen = leer_imagen_bytes(datos)
    except Exception:
        st.error("No se pudo leer la imagen. Prueba con un JPG o PNG.")
        return

    with st.spinner("Analizando..."):
        resultados, hay_rostro = rec.analizar(imagen, usar_imagen_completa=completa)
    anotada = cv2.cvtColor(rec.dibujar(imagen, resultados), cv2.COLOR_BGR2RGB)

    col_img, col_res = st.columns([3, 2], gap="large")
    with col_img:
        st.image(anotada, width="stretch")

    with col_res:
        if not resultados:
            st.warning("No se detectó ningún rostro. Prueba con una foto de frente y bien iluminada.")
            return
        if not hay_rostro:
            st.info("No se detectó un rostro; se analizó la imagen completa.")
        else:
            st.success(f"Se detectaron {len(resultados)} rostro(s).")

        for n, r in enumerate(resultados, start=1):
            with st.container(border=True):
                emocion = r["emocion"]
                color = COLOR_SENTIMIENTO.get(r["sentimiento"], "gray")
                st.markdown(f"#### Rostro {n}: {EMOJIS.get(emocion, '')} {emocion}")
                st.markdown(
                    f"Confianza **{r['confianza']:.0%}** · "
                    f"Sentimiento :{color}[**{r['sentimiento']}**]"
                )
                ordenadas = sorted(r["probabilidades"].items(), key=lambda kv: kv[1], reverse=True)
                for etiqueta, prob in ordenadas:
                    st.progress(prob, text=f"{EMOJIS.get(etiqueta, '')} {etiqueta}: {prob:.1%}")


def main():
    st.title("🙂 Detector de emociones faciales")
    st.write(
        "Sube una imagen o toma una foto. El modelo detecta cada rostro y estima "
        "qué emoción expresa y si el sentimiento es positivo, negativo o neutral."
    )
    rec, completa = barra_lateral()

    pestaña_subir, pestaña_camara = st.tabs(["📁 Subir imagen", "📷 Tomar foto"])
    with pestaña_subir:
        archivo = st.file_uploader(
            "Elige una imagen", type=["jpg", "jpeg", "png", "webp", "bmp"]
        )
        if archivo is not None:
            mostrar_resultados(rec, archivo.getvalue(), completa)
    with pestaña_camara:
        foto = st.camera_input("Mira a la cámara y toma la foto")
        if foto is not None:
            mostrar_resultados(rec, foto.getvalue(), completa)

    st.caption(
        "Nota: el modelo interpreta la expresión facial visible, no lo que la "
        "persona siente realmente. Los resultados son orientativos."
    )


main()
