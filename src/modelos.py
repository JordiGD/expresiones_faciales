"""
Arquitecturas de redes neuronales para clasificar emociones.

- cnn:          red convolucional propia, entrenada desde cero (rápida, 48x48 gris).
- mobilenet:    transfer learning con MobileNetV2 preentrenada en ImageNet.
- efficientnet: transfer learning con EfficientNetB0 preentrenada en ImageNet
                (suele dar la mejor precisión, pero es más lenta de entrenar).

La normalización de píxeles y el aumento de datos van DENTRO del modelo,
así que para predecir basta con pasarle la imagen redimensionada (0-255).
"""

from tensorflow import keras
from tensorflow.keras import layers


def capa_aumentacion() -> keras.Sequential:
    """Variaciones aleatorias de las imágenes (solo activas al entrenar)."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.08, 0.08),
            layers.RandomContrast(0.15),
        ],
        name="aumentacion",
    )


def _bloque_conv(x, filtros: int):
    x = layers.Conv2D(filtros, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(x)


def crear_cnn(num_clases: int, img_size: int = 48, canales: int = 1) -> keras.Model:
    """CNN tipo VGG con 4 bloques convolucionales."""
    entradas = keras.Input(shape=(img_size, img_size, canales), name="imagen")
    x = capa_aumentacion()(entradas)
    x = layers.Rescaling(1.0 / 255)(x)

    for filtros, abandono in [(64, 0.25), (128, 0.3), (256, 0.35), (512, 0.4)]:
        x = _bloque_conv(x, filtros)
        x = _bloque_conv(x, filtros)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(abandono)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)
    salidas = layers.Dense(num_clases, activation="softmax", name="emocion")(x)
    return keras.Model(entradas, salidas, name="cnn_emociones")


def crear_transfer(
    arquitectura: str, num_clases: int, img_size: int, pesos: str | None = "imagenet"
):
    """
    Modelo con una red preentrenada como extractor de características.

    Devuelve (modelo, base) para poder descongelar la base en la fase
    de ajuste fino (fine-tuning).
    """
    forma = (img_size, img_size, 3)
    entradas = keras.Input(shape=forma, name="imagen")
    x = capa_aumentacion()(entradas)

    if arquitectura == "mobilenet":
        # MobileNetV2 espera píxeles en el rango [-1, 1]
        x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)
        base = keras.applications.MobileNetV2(
            include_top=False, weights=pesos, input_shape=forma
        )
    elif arquitectura == "efficientnet":
        # EfficientNet normaliza internamente: recibe píxeles 0-255
        base = keras.applications.EfficientNetB0(
            include_top=False, weights=pesos, input_shape=forma
        )
    else:
        raise ValueError(f"Arquitectura desconocida: {arquitectura}")

    base.trainable = False
    # training=False mantiene la BatchNormalization de la base en modo
    # inferencia incluso cuando la descongelemos (recomendado por Keras).
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    salidas = layers.Dense(num_clases, activation="softmax", name="emocion")(x)
    modelo = keras.Model(entradas, salidas, name=f"{arquitectura}_emociones")
    return modelo, base


def descongelar(base: keras.Model, capas_entrenables: int) -> None:
    """Permite entrenar las últimas `capas_entrenables` capas de la base."""
    base.trainable = True
    for capa in base.layers[:-capas_entrenables]:
        capa.trainable = False
    # Las BatchNormalization se dejan congeladas para no romper lo aprendido
    for capa in base.layers:
        if isinstance(capa, layers.BatchNormalization):
            capa.trainable = False


def compilar(modelo: keras.Model, lr: float) -> None:
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
