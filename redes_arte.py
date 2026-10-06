
"""
CNN para clasificación de movimientos artísticos   utilizando  varias ramas
(Multi-Stream CNN)
Cada rama evalúa: 

1. Estilo y Composición Global (ConvNeXtBase - 384x384)
2. Trazos de Pincel y Textura (EfficientNetV2S - Patch 224x224)
3. Paleta de Colores y Armonía Cromática (MLP Denso con Histogramas HSV/RGB)

El dataset original tiene 5 movimientos de arte, utilizando para este trabajo práctico únicamente los movimientos
impresionista, cubista y art nouveau 

Descargar data set : import kagglehub

# Download latest version
path = kagglehub.dataset_download("stefaniasilva19/mi-dataset-limpio")

print("Path to dataset files:", path)

Recordar hacer rm de carpetas puntillismo y expresionismo. (rm -rf path)
"""

import os
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ConvNeXtBase, EfficientNetV2S
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, concatenate
from tensorflow.keras.models import Model


#Configuraciones iniciales 
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

#Se trabajó con Google Colab, realizando una configuración según entorno disponible CPU, GPU o TPU
try:
  tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
  tf.config.experimental_connect_to_cluster(tpu)
  tf.tpu.experimental.initialize_tpu_system(tpu)
  strategy = tf.distribute.TPUStrategy(tpu)
  print(f"TPU detectada. Réplicas: {strategy.num_replicas_in_sync}")
except (ValueError, tf.errors.NotFoundError):
  gpus = tf.config.list_physical_devices("GPU")
  if gpus:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    strategy = (
        tf.distribute.MirroredStrategy()
        if len(gpus) > 1
        else tf.distribute.get_strategy()
    )
    print(f"GPU(s) detectada(s): {len(gpus)}")
  else:
    strategy = tf.distribute.get_strategy()
    print("No se detectó TPU ni GPU. Corriendo en CPU (va a ser lento).")

#Tamaño del lote, es decir, cuántas imágenes procesará cada núcleo en un solo paso
BATCH_SIZE_PER_REPLICA = 16  # Reducido levemente para soportar imágenes de 384x384 y evitar problemas de memoria
batch_size = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync  #Calcula el batch size global real. Multiplica las 16 muestras por la cantidad de réplicas en paralelo que detectó strategy (# núcleos)
print(f"batch_size efectivo: {batch_size}")

TARGET_SIZE_GLOBAL = (384, 384)  # Alta resolución para la rama global
TARGET_SIZE_PATCH = (224, 224)  # Resolución para recortes de microtextura
output_dir = "/content/data/"  #Path de mi dataset


# EXTRACCIÓN NATIVA DE CARACTERÍSTICAS CROMÁTICAS (COLOR)
def extract_color_features_tf(image):
  """Extrae un vector de características cromáticas (HSV + RGB statistics e histogramas)

  utilizando operaciones nativas de TensorFlow compatibles con GPU/TPU.
  """
  img_norm = tf.cast(image, tf.float32) / 255.0  # Normalización de datos = Rango [0, 1] 


  # Conversión de RGB a espacio de color HSV (Tono, Saturación y valor)
  hsv = tf.image.rgb_to_hsv(img_norm)
  h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
  r, g, b = img_norm[..., 0], img_norm[..., 1], img_norm[..., 2]

  # Estadísticos descriptivos (Media y Desviación Estándar)
  means = [
      tf.reduce_mean(r),
      tf.reduce_mean(g),
      tf.reduce_mean(b),
      tf.reduce_mean(h),
      tf.reduce_mean(s),
      tf.reduce_mean(v),
  ]
  stds = [
      tf.math.reduce_std(r),
      tf.math.reduce_std(g),
      tf.math.reduce_std(b),
      tf.math.reduce_std(h),
      tf.math.reduce_std(s),
      tf.math.reduce_std(v),
  ]
  # con estos datos estadísticos se tiene una idea de cupal es el color dominante o tono
  # el stds por otro lado  mide la variedad cromática y el contraste. Ejemplo, una desviacióm estándar alta en v indica alto contraste 


  # 2. Histogramas 1D normalizados (distribución del color)
  num_pixels = tf.cast(tf.shape(image)[0] * tf.shape(image)[1], tf.float32)

  h_hist = (
      tf.cast(
          tf.histogram_fixed_width(h, value_range=[0.0, 1.0], nbins=16),
          tf.float32,
      )
      / num_pixels
  )
  s_hist = (
      tf.cast(
          tf.histogram_fixed_width(s, value_range=[0.0, 1.0], nbins=8),
          tf.float32,
      )
      / num_pixels
  )
  v_hist = (
      tf.cast(
          tf.histogram_fixed_width(v, value_range=[0.0, 1.0], nbins=8),
          tf.float32,
      )
      / num_pixels
  )

  r_hist = (
      tf.cast(
          tf.histogram_fixed_width(r, value_range=[0.0, 1.0], nbins=8),
          tf.float32,
      )
      / num_pixels
  )
  g_hist = (
      tf.cast(
          tf.histogram_fixed_width(g, value_range=[0.0, 1.0], nbins=8),
          tf.float32,
      )
      / num_pixels
  )
  b_hist = (
      tf.cast(
          tf.histogram_fixed_width(b, value_range=[0.0, 1.0], nbins=8),
          tf.float32,
      )
      / num_pixels
  )

  # Unir en un solo vector de 68 dimensiones
  stats = tf.stack(means + stds)
  color_vector = tf.concat(
      [stats, h_hist, s_hist, v_hist, r_hist, g_hist, b_hist], axis=0
  )
  return color_vector


#DATA AUGMENTATION & PIPELINE MULTI-ENTRADA (tf.data)

AUTOTUNE = tf.data.AUTOTUNE
''' 
Le indica a TensorFlow que calcule automáticamente cuántos recursos de tu procesador (CPU) 
debe asignar para tareas como leer, transformar o precargar las imágenes 
en segundo plano (prefetch). Esto evita que la GPU se quede "esperando" a que la CPU 
termine de procesar los datos, maximizando la velocidad de entrenamiento.
'''

#aumentos de datos
augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.08),
        tf.keras.layers.RandomFlip("horizontal"),
    ],
    name="augmentation",
)


def process_multi_input_train(image, label):
  """Preprocesamiento para Entrenamiento: genera entradas para las 3 ramas."""
  augmented = augmentation(image, training=True)

  # Entrada 1: Imagen Global (384x384)
  img_global = tf.image.resize(augmented, TARGET_SIZE_GLOBAL)
  img_global = tf.keras.applications.convnext.preprocess_input(img_global)

  # Entrada 2: Crop de Textura / Trazos (224x224)
  img_patch = tf.image.random_crop(augmented, size=[224, 224, 3])
  img_patch = tf.keras.applications.efficientnet_v2.preprocess_input(img_patch)

  # Entrada 3: Vector Cromático
  color_features = extract_color_features_tf(augmented)

  return (
      {
          "img_global": img_global,
          "img_patch": img_patch,
          "color_hist": color_features,
      },
      label,
  )


#tener los tres parametros necesarios de una imagen
def process_multi_input_eval(image, label):
  """Preprocesamiento para Validación/Test (determinista)."""
  # Entrada 1: Imagen Global
  img_global = tf.image.resize(image, TARGET_SIZE_GLOBAL)
  img_global = tf.keras.applications.convnext.preprocess_input(img_global)
  #ConvNeXt detecta patrones, características generales

  # Entrada 2: Crop Central de Textura
  img_patch = tf.image.central_crop(image, central_fraction=0.7)
  img_patch = tf.image.resize(img_patch, TARGET_SIZE_PATCH)
  img_patch = tf.keras.applications.efficientnet_v2.preprocess_input(img_patch)
  #para mayor detalle se utilizó  EfficientNet , para analisis de miicrotexturas y trazos, detalles finos

  # Entrada 3: Vector Cromático
  color_features = extract_color_features_tf(image)

  return (
      {
          "img_global": img_global,
          "img_patch": img_patch,
          "color_hist": color_features,
      },
      label,
  )


# Cargar datasets en bruto (unbatched para mapeo individual)
train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    os.path.join(output_dir, "train"),
    image_size=TARGET_SIZE_GLOBAL,
    batch_size=None,
    label_mode="categorical",
    shuffle=True,
    seed=SEED,
)

val_ds_raw = tf.keras.utils.image_dataset_from_directory(
    os.path.join(output_dir, "val"),
    image_size=TARGET_SIZE_GLOBAL,
    batch_size=None, #no se puede realizar por lotes debido a mi función. El agrupamiento se hará al final
    label_mode="categorical",
    shuffle=False, #aleatorio para evitar aprendizaje con orden sesgado
)

test_ds_raw = tf.keras.utils.image_dataset_from_directory(
    os.path.join(output_dir, "test"),
    image_size=TARGET_SIZE_GLOBAL,
    batch_size=None,
    label_mode="categorical",
    shuffle=False,
)

class_names = train_ds_raw.class_names
output_n = len(class_names)
np.save("label_list.npy", class_names) #traducir mis índices a la 'etiqueta' correcta 
print("Clases detectadas:", class_names)

# Construcción del Pipeline tf.data
train_ds = (
    train_ds_raw.map(process_multi_input_train, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)

val_ds = (
    val_ds_raw.map(process_multi_input_eval, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)

test_ds = (
    test_ds_raw.map(process_multi_input_eval, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)

# ============================================================
# 5. CONSTRUCCIÓN DE LA ARQUITECTURA MULTI-RAMA (FUNCTIONAL API)
# ============================================================
with strategy.scope():
  # --- RAMA 1: Estilo y Composición Global ---
  input_global = Input(
      shape=(384, 384, 3), name="img_global", dtype=tf.float32
  )
  base_global = ConvNeXtBase(
      weights="imagenet", include_top=False, input_shape=(384, 384, 3)
  )
  base_global.trainable = False
  x1 = base_global(input_global)
  x1 = GlobalAveragePooling2D()(x1)
  x1 = Dense(512, activation="relu")(x1)

  # --- RAMA 2: Trazos de Pincel y Textura ---
  input_patch = Input(shape=(224, 224, 3), name="img_patch", dtype=tf.float32)
  base_patch = EfficientNetV2S(
      weights="imagenet", include_top=False, input_shape=(224, 224, 3)
  )
  base_patch.trainable = False
  x2 = base_patch(input_patch)
  x2 = GlobalAveragePooling2D()(x2)
  x2 = Dense(256, activation="relu")(x2)

  # --- RAMA 3: Paleta de Colores (HSV / RGB Statistics) ---
  input_color = Input(shape=(68,), name="color_hist", dtype=tf.float32)
  x3 = Dense(128, activation="relu")(input_color)
  x3 = Dense(64, activation="relu")(x3)

  # --- FUSIÓN Y CABEZAL CLASIFICADOR ---
  combined = concatenate([x1, x2, x3])
  dense_1 = Dense(512, activation="relu")(combined)
  drop_1 = Dropout(0.5)(dense_1)
  dense_2 = Dense(256, activation="relu")(drop_1)
  drop_2 = Dropout(0.3)(dense_2)
  output = Dense(output_n, activation="softmax", name="art_movement_output")(
      drop_2
  )

  model = Model(
      inputs=[input_global, input_patch, input_color], outputs=output
  )

  # Pérdida con Label Smoothing (0.1) para mitigar overconfidence y reducir Loss
  loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
      loss=loss_fn,
      metrics=["accuracy"],
  )

model.summary()

# ============================================================
# 6. CALLBACKS Y ENTRENAMIENTO — FASE 1 (Cabezal Multi-Rama)
# ============================================================
early_stop = EarlyStopping(
    monitor="val_loss", patience=7, restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=3, min_lr=1e-7
)
checkpoint = ModelCheckpoint(
    "mejor_modelo_multired.keras", monitor="val_accuracy", save_best_only=True
)
callbacks = [early_stop, checkpoint, reduce_lr]

epochs_fase1 = 12
print("\n🚀 Iniciando Fase 1: Entrenamiento del Cabezal Multired...")
history_fase1 = model.fit(
    train_ds,
    epochs=epochs_fase1,
    validation_data=val_ds,
    callbacks=callbacks,
    verbose=2,
)

# ============================================================
# 7. FINE-TUNING PROFUNDO — FASE 2
# ============================================================
print("\n🔓 Descongelando capas superiores para Fine-Tuning Profundo...")
with strategy.scope():
  # Descongelar las últimas capas de ConvNeXt y EfficientNetV2
  base_global.trainable = True
  for layer in base_global.layers[:-30]:
    layer.trainable = False

  base_patch.trainable = True
  for layer in base_patch.layers[:-20]:
    layer.trainable = False

  # MANTENER CONGELADAS las capas de Batch Normalization
  for base in [base_global, base_patch]:
    for layer in base.layers:
      if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

  # Re-compilar con un Learning Rate bajo
  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
      loss=loss_fn,
      metrics=["accuracy"],
  )

epochs_fase2 = 20
history_fase2 = model.fit(
    train_ds,
    epochs=epochs_fase2,
    validation_data=val_ds,
    callbacks=callbacks,
    verbose=2,
)

# ============================================================
# 8. EVALUACIÓN FINAL Y GUARDADO DEL MODELO
# ============================================================
print("\n📊 Evaluando modelo final en Test Set...")
resultado = model.evaluate(test_ds, verbose=0)
print(f"Test Loss Final: {resultado[0]:.4f}")
print(f"Test Accuracy Final: {resultado[1]*100:.2f}%")

# Gráficas de rendimiento
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_fase2.history["accuracy"], label="Entrenamiento (FT)")
plt.plot(history_fase2.history["val_accuracy"], label="Validación (FT)")
plt.title("Precisión del Modelo Multi-Rama")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_fase2.history["loss"], label="Entrenamiento (FT)")
plt.plot(history_fase2.history["val_loss"], label="Validación (FT)")
plt.title("Pérdida del Modelo Multi-Rama")
plt.legend()
plt.show()

# Guardar arquitectura y pesos finales
model.save("modelo_movimientos_artisticos_multired.keras")
print(
    "✅ Modelo guardado exitosamente como"
    " modelo_movimientos_artisticos_multired.keras"
)


#convertir modelo a tensorflowjs para poder
#  pip install tensorflowjs
#  tensorflowjs_converter --input_format=keras mov_arte_multired.keras model_js/