"""
Entrenamiento del Modelo CNN para Reconocimiento de Señales de Tráfico
Dataset: GTSRB (German Traffic Sign Recognition Benchmark)
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

print("TensorFlow version:", tf.__version__)

# ============================================
# 1. DESCARGAR Y CARGAR DATASET GTSRB
# ============================================

def descargar_gtsrb():
    """Descarga el dataset GTSRB desde Kaggle usando kagglehub"""
    dataset_dir = 'gtsrb_dataset'
    
    # Verificar si ya existe localmente
    if os.path.exists(dataset_dir) and os.path.exists(os.path.join(dataset_dir, 'Train')):
        print("✓ Dataset GTSRB ya existe localmente")
        return dataset_dir
    
    print("\n" + "="*60)
    print("DESCARGANDO DATASET GTSRB DESDE KAGGLE")
    print("="*60)
    
    try:
        import kagglehub
        
        print("Descargando dataset GTSRB... (Esto puede tardar varios minutos)")
        print("Por favor espera...")
        
        # Descargar la última versión del dataset
        path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
        
        print(f"\n✓ Dataset descargado exitosamente en: {path}")
        
        # Verificar si necesitamos copiar/mover archivos
        if os.path.exists(os.path.join(path, 'Train')):
            return path
        
        # Buscar la carpeta Train en subdirectorios
        for root, dirs, files in os.walk(path):
            if 'Train' in dirs:
                return root
        
        print(f"Usando ruta del dataset: {path}")
        return path
        
    except ImportError:
        print("\n⚠ ERROR: kagglehub no está instalado")
        print("\nPor favor, instala kagglehub ejecutando:")
        print("  pip install kagglehub")
        print("\nO descarga el dataset manualmente desde:")
        print("  https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
        return None
    except Exception as e:
        print(f"\n⚠ Error descargando el dataset: {e}")
        print("\nPuedes descargar el dataset manualmente desde:")
        print("  https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
        print("\nY extraer el contenido en una carpeta llamada 'gtsrb_dataset'")
        return None

def cargar_imagenes_desde_carpeta(data_dir, img_size=30):
    """Carga imágenes y etiquetas desde la estructura de carpetas"""
    imagenes = []
    etiquetas = []
    
    # Para entrenamiento: Train/ con subcarpetas por clase
    train_dir = os.path.join(data_dir, 'Train')
    if not os.path.exists(train_dir):
        # Alternativa: directorio directo con clases
        train_dir = data_dir
    
    print(f"Cargando imágenes desde: {train_dir}")
    
    # Iterar sobre las carpetas de clases (0-42)
    for clase in range(43):
        clase_dir = os.path.join(train_dir, str(clase))
        if not os.path.exists(clase_dir):
            print(f"Advertencia: No se encontró la carpeta para la clase {clase}")
            continue
        
        archivos = [f for f in os.listdir(clase_dir) if f.endswith(('.png', '.ppm', '.jpg'))]
        print(f"Clase {clase}: {len(archivos)} imágenes")
        
        for archivo in archivos:
            try:
                ruta_imagen = os.path.join(clase_dir, archivo)
                imagen = Image.open(ruta_imagen)
                
                # Redimensionar la imagen a tamaño fijo ANTES de convertir a array
                imagen = imagen.resize((img_size, img_size))
                imagen = np.array(imagen)
                
                # Verificar que sea RGB (3 canales)
                if imagen is not None and len(imagen.shape) == 3 and imagen.shape[2] == 3:
                    imagenes.append(imagen)
                    etiquetas.append(clase)
            except Exception as e:
                print(f"Error cargando {archivo}: {e}")
                continue
    
    # Ahora todas las imágenes tienen el mismo tamaño, se puede convertir a array
    return np.array(imagenes), np.array(etiquetas)

# Intentar cargar el dataset
dataset_dir = descargar_gtsrb()

if dataset_dir is None:
    print("\n" + "="*60)
    print("NO SE PUDO DESCARGAR EL DATASET")
    print("="*60)
    print("\nPor favor:")
    print("1. Instala kagglehub: pip install kagglehub")
    print("2. O descarga el dataset manualmente desde Kaggle")
    print("3. Vuelve a ejecutar este script")
    exit(1)
    
else:
    # Cargar el dataset GTSRB real
    print("\nCargando dataset GTSRB...")
    x_all, y_all = cargar_imagenes_desde_carpeta(dataset_dir)
    
    if len(x_all) == 0:
        raise ValueError("No se pudieron cargar imágenes. Verifica la estructura del dataset.")
    
    # Dividir en entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    
    NUM_CLASSES = 43
    
    print(f"\nDataset cargado exitosamente!")
    print(f"Número de clases: {NUM_CLASSES}")
    print(f"Imágenes de entrenamiento: {len(x_train)}")
    print(f"Imágenes de prueba: {len(x_test)}")

# ============================================
# 2. PREPROCESAMIENTO DE DATOS
# ============================================
IMG_SIZE = 30  # Tamaño estándar para todas las imágenes
BATCH_SIZE = 32

def preprocesar_datos(x_data, y_data):
    """
    Preprocesa las imágenes: normaliza los valores de píxeles
    """
    # Las imágenes ya están redimensionadas a 30x30
    # Solo necesitamos normalizar valores de píxeles de [0, 255] a [0, 1]
    x_data = x_data.astype('float32') / 255.0
    
    return x_data, y_data

print("\nPreprocesando imágenes...")
x_train, y_train = preprocesar_datos(x_train, y_train)
x_test, y_test = preprocesar_datos(x_test, y_test)

print("Preprocesamiento completado.")
print(f"Forma de datos de entrenamiento: {x_train.shape}")
print(f"Forma de datos de prueba: {x_test.shape}")

# ============================================
# 3. CONSTRUCCIÓN DEL MODELO CNN
# ============================================
def crear_modelo():
    """
    Crea una Red Neuronal Convolucional (CNN) para clasificación de señales de tráfico.
    
    Arquitectura:
    - 3 bloques convolucionales con pooling
    - Capas densas para clasificación
    - Dropout para prevenir overfitting
    """
    modelo = keras.Sequential([
        # Capa de entrada
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        
        # Primer bloque convolucional
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Segundo bloque convolucional
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Tercer bloque convolucional
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # Aplanar y capas densas
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Capa de salida (43 clases)
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return modelo

# Crear el modelo
modelo = crear_modelo()

# Resumen del modelo
print("\n" + "="*60)
print("ARQUITECTURA DEL MODELO")
print("="*60)
modelo.summary()

# ============================================
# 4. COMPILACIÓN Y ENTRENAMIENTO
# ============================================
# Compilar el modelo
modelo.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks para mejorar el entrenamiento
callbacks = [
    # Reducir learning rate cuando el accuracy se estanque
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=2,
        verbose=1,
        min_lr=1e-7
    ),
    # Guardar el mejor modelo
    keras.callbacks.ModelCheckpoint(
        'mejor_modelo_trafico.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print("\n" + "="*60)
print("INICIANDO ENTRENAMIENTO")
print("="*60)

# Entrenar el modelo
EPOCHS = 10
historial = modelo.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ============================================
# 5. EVALUACIÓN DEL MODELO
# ============================================
print("\n" + "="*60)
print("EVALUACIÓN FINAL DEL MODELO")
print("="*60)

# Evaluar en el conjunto de prueba
test_loss, test_accuracy = modelo.evaluate(x_test, y_test, verbose=0)
print(f"\nPérdida en prueba: {test_loss:.4f}")
print(f"Precisión en prueba: {test_accuracy*100:.2f}%")

# Guardar el modelo final
modelo.save('modelo_trafico.h5')
print("\n✓ Modelo guardado como 'modelo_trafico.h5'")
print("✓ Mejor modelo guardado como 'mejor_modelo_trafico.h5'")

# ============================================
# 6. VISUALIZACIÓN DE RESULTADOS
# ============================================
print("\nGenerando gráficas de entrenamiento...")

# Crear figura con dos subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Gráfica de Accuracy (Precisión)
ax1.plot(historial.history['accuracy'], label='Entrenamiento', marker='o')
ax1.plot(historial.history['val_accuracy'], label='Validación', marker='s')
ax1.set_title('Precisión del Modelo', fontsize=14, fontweight='bold')
ax1.set_xlabel('Época')
ax1.set_ylabel('Precisión')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Gráfica de Loss (Pérdida)
ax2.plot(historial.history['loss'], label='Entrenamiento', marker='o')
ax2.plot(historial.history['val_loss'], label='Validación', marker='s')
ax2.set_title('Pérdida del Modelo', fontsize=14, fontweight='bold')
ax2.set_xlabel('Época')
ax2.set_ylabel('Pérdida')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('resultado_entrenamiento.png', dpi=300, bbox_inches='tight')
print("✓ Gráfica guardada como 'resultado_entrenamiento.png'")

plt.show()

print("\n" + "="*60)
print("¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
print("="*60)
print("\nArchivos generados:")
print("  - modelo_trafico.h5 (modelo final)")
print("  - mejor_modelo_trafico.h5 (mejor modelo durante entrenamiento)")
print("  - resultado_entrenamiento.png (gráficas)")
print("\nAhora puedes ejecutar 'streamlit run app.py' para probar el modelo.")