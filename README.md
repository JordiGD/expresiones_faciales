# 🙂 Reconocimiento de emociones faciales con Deep Learning

Proyecto de **análisis de sentimientos a partir de imágenes**: se sube una foto (o se toma con la cámara), el sistema detecta cada rostro y una red neuronal convolucional identifica la emoción que expresa y si el sentimiento es **positivo, negativo o neutral**.

| Emoción   | Sentimiento |
|-----------|-------------|
| Felicidad | Positivo    |
| Sorpresa  | Neutral (ambigua) |
| Neutral   | Neutral     |
| Tristeza  | Negativo    |
| Enojo     | Negativo    |
| Miedo     | Negativo    |
| Asco      | Negativo    |

---

## 1. ¿Cómo funciona?

```
 Foto / cámara ──► Detección de rostros ──► Recorte y preprocesamiento ──► Red neuronal (CNN) ──► Emoción + sentimiento
                   (OpenCV Haar Cascade)    (gris, redimensionar)          (softmax, 7 clases)
```

1. **Detección de rostros** (`src/detector.py`): un clasificador Haar Cascade de OpenCV encuentra la posición de cada cara. Si hay varias personas, se analizan todas.
2. **Preprocesamiento** (`src/predictor.py`): cada rostro se recorta, se convierte a escala de grises y se redimensiona (48×48 o 96–112 px según el modelo), igual que las imágenes con las que se entrenó.
3. **Clasificación** (`src/modelos.py`): la red neuronal devuelve una probabilidad para cada una de las 7 emociones. La de mayor probabilidad es la predicción.
4. **Sentimiento** (`config.py`): cada emoción se asocia a una polaridad (positivo / negativo / neutral). Puedes cambiar esta asignación en `SENTIMIENTO`.

### Dataset: FER-2013
Unas **35.900 imágenes** de rostros en escala de grises de 48×48 px, etiquetadas con 7 emociones (≈28.700 de entrenamiento y ≈7.200 de prueba). Está desbalanceado: *disgust* (asco) tiene muy pocas imágenes, por eso el entrenamiento usa **pesos por clase**.

### Modelos disponibles

| `--modelo`     | Tipo | Entrada | Ventajas |
|----------------|------|---------|----------|
| `cnn`          | CNN propia desde cero (4 bloques conv + BatchNorm + Dropout) | 48×48 gris | Rápida, fácil de explicar, entrena bien en CPU |
| `mobilenet`    | *Transfer learning* con MobileNetV2 (ImageNet) | 96×96 | Ligera, buena para celular |
| `efficientnet` | *Transfer learning* con EfficientNetB0 (ImageNet) | 112×112 | Suele dar la mejor precisión; conviene GPU |

Los modelos de *transfer learning* se entrenan en dos fases: primero solo la "cabeza" nueva (base congelada) y luego un **ajuste fino** de las últimas 40 capas con una tasa de aprendizaje 20 veces menor.

Técnicas incluidas: aumento de datos (volteo, rotación, zoom, desplazamiento, contraste), *Dropout*, *Batch Normalization*, *Early Stopping*, reducción automática de la tasa de aprendizaje y guardado del mejor modelo según validación.

---

## 2. Instalación

Requiere **Python 3.10 – 3.12** (TensorFlow aún no soporta todas las versiones nuevas).

```bash
# 1) Crear y activar un entorno virtual
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / Mac:
source venv/bin/activate

# 2) Instalar dependencias
pip install -r requirements.txt
```

## 3. Descargar los datos

**Opción A (automática):**
```bash
python descargar_datos.py
```

**Opción B (manual):** descarga el zip desde https://www.kaggle.com/datasets/msambare/fer2013 y descomprímelo para que quede así:

```
datos/fer2013/
├── train/  angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
└── test/   angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
```

Si tienes el archivo original `fer2013.csv`, conviértelo con:
```bash
python convertir_csv.py --csv fer2013.csv --salida datos/fer2013
```

## 4. Entrenar

```bash
# CNN propia (recomendado para empezar; en CPU tarda aprox. 1-3 min por época)
python entrenar.py --datos datos/fer2013 --modelo cnn

# Transfer learning (mejor con GPU, por ejemplo en Google Colab)
python entrenar.py --datos datos/fer2013 --modelo efficientnet
```

Opciones útiles: `--epocas 30`, `--batch 32` (si te quedas sin memoria), `--paciencia 10`, `--lr 0.0005`.

Se guardan:
- `modelos/modelo_<arquitectura>.keras` y `.json` → el modelo y sus metadatos
- `resultados/curvas_<arquitectura>.png` → gráficas de precisión y pérdida
- `resultados/entrenamiento_<arquitectura>.csv` → métricas por época

> 💡 **¿Sin GPU?** Usa el cuaderno `entrenamiento_colab.ipynb` en [Google Colab](https://colab.research.google.com) (Entorno de ejecución → Cambiar tipo → GPU T4). Luego descarga la carpeta `modelos/` y cópiala a tu proyecto.

## 5. Evaluar

```bash
python evaluar.py --modelo modelos/modelo_cnn.keras --datos datos/fer2013
```

Genera `resultados/reporte_<modelo>.txt` (precisión, recall y F1 por emoción) y `resultados/matriz_confusion_<modelo>.png`.

**Precisión esperada en FER-2013:** en la literatura una CNN de este tipo suele quedar alrededor del 63–67 %, y el *transfer learning* con ajuste fino alrededor del 66–70 %. Como referencia, se estima que los humanos aciertan cerca del 65 % en este dataset, porque muchas imágenes son ambiguas. Es normal que el modelo confunda *miedo* con *sorpresa* o *tristeza*, y que *asco* sea la emoción más difícil.

## 6. Usar el modelo

### Aplicación web (subir imagen o tomar foto)
```bash
streamlit run app.py
```
Se abre en el navegador. Tiene dos pestañas: **Subir imagen** y **Tomar foto** (usa la cámara del computador). Muestra cada rostro marcado, la emoción, la confianza, el sentimiento y las probabilidades de todas las emociones.

Para usarla desde el **celular**, conéctalo a la misma red Wi-Fi y abre la dirección *Network URL* que muestra Streamlit. (Algunos navegadores solo permiten la cámara en `localhost` o con HTTPS; en ese caso usa la pestaña de subir imagen, que también deja tomar una foto desde el celular.)

### Línea de comandos
```bash
python predecir.py --imagen mi_foto.jpg           # una imagen
python predecir.py --carpeta mis_fotos/           # varias imágenes
python predecir.py --webcam                       # tiempo real (q = salir, g = guardar)
```
Las imágenes con los resultados dibujados se guardan en `resultados/predicciones/`.

### Desde tu propio código
```python
from src.predictor import ReconocedorEmociones
from src.imagenes import leer_imagen

rec = ReconocedorEmociones("modelos/modelo_cnn.keras")
resultados, hay_rostro = rec.analizar(leer_imagen("foto.jpg"))
for r in resultados:
    print(r["emocion"], r["confianza"], r["sentimiento"])
```

---

## 7. Estructura del proyecto

```
emociones_faciales/
├── app.py                    # Aplicación web (Streamlit)
├── entrenar.py               # Entrenamiento
├── evaluar.py                # Evaluación con el conjunto de prueba
├── predecir.py               # Predicción por consola / webcam
├── descargar_datos.py        # Descarga FER-2013 desde Kaggle
├── convertir_csv.py          # Convierte fer2013.csv a carpetas
├── config.py                 # Emociones, sentimientos, colores, parámetros
├── entrenamiento_colab.ipynb # Cuaderno para entrenar con GPU en Colab
├── requirements.txt
├── src/
│   ├── datos.py              # Carga del dataset y pesos de clase
│   ├── modelos.py            # Arquitecturas (CNN, MobileNetV2, EfficientNetB0)
│   ├── detector.py           # Detección de rostros (OpenCV)
│   ├── predictor.py          # Clase ReconocedorEmociones
│   ├── imagenes.py           # Lectura/escritura de imágenes
│   └── graficas.py           # Curvas y matriz de confusión
├── datos/                    # (aquí va el dataset)
├── modelos/                  # (modelos entrenados)
└── resultados/               # (gráficas, reportes y predicciones)
```

## 8. Limitaciones (importante para el informe)

- El modelo reconoce la **expresión facial visible**, no lo que la persona siente por dentro: alguien puede sonreír estando triste.
- FER-2013 tiene etiquetas ruidosas y poca diversidad en algunas edades y tonos de piel, lo que puede generar sesgos.
- Funciona mejor con rostros **de frente, bien iluminados y sin obstrucciones** (gafas oscuras, tapabocas, manos).
- El detector Haar Cascade puede fallar con rostros de perfil o muy inclinados.
- No debe usarse para tomar decisiones sobre personas (contratación, vigilancia, evaluaciones) sin supervisión humana y consentimiento.

## 9. Ideas para mejorar

- Probar el dataset **FER+** (re-etiquetado de FER-2013 por 10 personas, menos ruido) o **AffectNet** / **RAF-DB**.
- Reemplazar el detector Haar por uno más robusto (YuNet de OpenCV o MediaPipe).
- Promediar predicciones de varios modelos (*ensemble*) o con la imagen volteada (*test-time augmentation*).
- Exportar el modelo a TensorFlow Lite para una app móvil.
- Combinar con análisis de texto (por ejemplo, el pie de foto) para un análisis multimodal.
