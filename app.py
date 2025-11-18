"""
Aplicación Web de Reconocimiento de Señales de Tráfico
Usando Streamlit y TensorFlow
"""

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# ============================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================
st.set_page_config(
    page_title="Reconocimiento de Señales de Tráfico",
    page_icon="🚦",
    layout="wide"
)

# ============================================
# DICCIONARIO DE CLASES (43 señales de tráfico)
# ============================================
# Dataset GTSRB - German Traffic Sign Recognition Benchmark
CLASES_SEÑALES = {
    0: 'Límite de velocidad (20 km/h)',
    1: 'Límite de velocidad (30 km/h)',
    2: 'Límite de velocidad (50 km/h)',
    3: 'Límite de velocidad (60 km/h)',
    4: 'Límite de velocidad (70 km/h)',
    5: 'Límite de velocidad (80 km/h)',
    6: 'Fin de límite de velocidad (80 km/h)',
    7: 'Límite de velocidad (100 km/h)',
    8: 'Límite de velocidad (120 km/h)',
    9: 'Prohibido adelantar',
    10: 'Prohibido adelantar a camiones',
    11: 'Intersección con prioridad',
    12: 'Carretera con prioridad',
    13: 'Ceda el paso',
    14: 'Stop',
    15: 'Prohibido el paso de vehículos',
    16: 'Prohibido el paso de camiones',
    17: 'Prohibido el paso',
    18: 'Peligro general',
    19: 'Curva peligrosa a la izquierda',
    20: 'Curva peligrosa a la derecha',
    21: 'Doble curva',
    22: 'Carretera con baches',
    23: 'Carretera resbaladiza',
    24: 'Estrechamiento de la calzada por la derecha',
    25: 'Obras',
    26: 'Semáforo',
    27: 'Peatones',
    28: 'Niños cruzando',
    29: 'Cruce de bicicletas',
    30: 'Peligro de hielo/nieve',
    31: 'Animales salvajes',
    32: 'Fin de todas las restricciones de velocidad',
    33: 'Gire a la derecha',
    34: 'Gire a la izquierda',
    35: 'Solo adelante',
    36: 'Adelante o derecha',
    37: 'Adelante o izquierda',
    38: 'Mantenga su derecha',
    39: 'Mantenga su izquierda',
    40: 'Rotonda obligatoria',
    41: 'Fin de prohibición de adelantar',
    42: 'Fin de prohibición de adelantar a camiones'
}

# ============================================
# CARGAR MODELO
# ============================================
@st.cache_resource
def cargar_modelo():
    """
    Carga el modelo entrenado.
    Usa @st.cache_resource para cargar el modelo solo una vez.
    """
    try:
        modelo = tf.keras.models.load_model('modelo_trafico.h5')
        return modelo
    except:
        st.error("❌ No se encontró el archivo 'modelo_trafico.h5'. Por favor, ejecuta primero 'python entrenamiento.py'")
        st.stop()

# ============================================
# FUNCIÓN DE PREPROCESAMIENTO
# ============================================
def preprocesar_imagen(imagen):
    """
    Preprocesa la imagen para que sea compatible con el modelo.
    
    Args:
        imagen: Imagen PIL
    
    Returns:
        Imagen preprocesada como array numpy
    """
    # Convertir a RGB si es necesario
    if imagen.mode != 'RGB':
        imagen = imagen.convert('RGB')
    
    # Redimensionar a 30x30 (mismo tamaño del entrenamiento)
    imagen = imagen.resize((30, 30))
    
    # Convertir a array numpy
    img_array = np.array(imagen)
    
    # Normalizar (dividir por 255)
    img_array = img_array / 255.0
    
    # Agregar dimensión del batch (el modelo espera un batch de imágenes)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# ============================================
# INTERFAZ DE USUARIO
# ============================================

# Título y descripción
st.title("🚦 Reconocimiento de Señales de Tráfico")
st.markdown("### Sistema de Clasificación Automática usando Deep Learning")
st.markdown("---")

# Información del proyecto
with st.expander("ℹ️ Acerca de este proyecto"):
    st.write("""
    **Proyecto Universitario de Machine Learning**
    
    Este sistema utiliza una Red Neuronal Convolucional (CNN) entrenada con el dataset 
    GTSRB (German Traffic Sign Recognition Benchmark) que contiene 43 tipos diferentes 
    de señales de tráfico.
    
    **Características:**
    - 🧠 Modelo: CNN con TensorFlow/Keras
    - 📊 Dataset: GTSRB (más de 50,000 imágenes)
    - 🎯 Clases: 43 tipos de señales de tráfico
    - 🖼️ Entrada: Imágenes de 30x30 píxeles
    """)

# Cargar modelo
modelo = cargar_modelo()
st.success("✅ Modelo cargado exitosamente")

# Crear dos columnas
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Cargar Imagen")
    
    # File uploader
    archivo_subido = st.file_uploader(
        "Selecciona una imagen de una señal de tráfico",
        type=['jpg', 'jpeg', 'png'],
        help="Formatos aceptados: JPG, JPEG, PNG"
    )
    
    if archivo_subido is not None:
        # Cargar y mostrar imagen original
        imagen = Image.open(archivo_subido)
        st.image(imagen, caption='Imagen cargada', use_container_width=True)
        
        # Botón para realizar predicción
        if st.button("🔍 Analizar Señal de Tráfico", type="primary", use_container_width=True):
            with st.spinner('Analizando imagen...'):
                # Preprocesar imagen
                img_procesada = preprocesar_imagen(imagen)
                
                # Realizar predicción
                predicciones = modelo.predict(img_procesada, verbose=0)
                
                # Obtener clase predicha y confianza
                clase_predicha = np.argmax(predicciones[0])
                confianza = predicciones[0][clase_predicha] * 100
                
                # Guardar resultados en session_state
                st.session_state.clase_predicha = clase_predicha
                st.session_state.confianza = confianza
                st.session_state.predicciones = predicciones[0]

with col2:
    st.subheader("🎯 Resultado del Análisis")
    
    if 'clase_predicha' in st.session_state:
        # Mostrar resultado principal
        st.markdown("### Predicción:")
        
        # Crear un contenedor destacado para el resultado
        resultado_container = st.container()
        with resultado_container:
            # Nombre de la señal
            nombre_señal = CLASES_SEÑALES[st.session_state.clase_predicha]
            st.markdown(f"## 🚸 **{nombre_señal}**")
            
            # Barra de confianza
            st.markdown(f"**Confianza:** {st.session_state.confianza:.2f}%")
            st.progress(st.session_state.confianza / 100)
            
            # Interpretación de confianza
            if st.session_state.confianza > 90:
                st.success("✅ Predicción muy confiable")
            elif st.session_state.confianza > 70:
                st.info("ℹ️ Predicción confiable")
            else:
                st.warning("⚠️ Predicción con baja confianza")
        
        st.markdown("---")
        
        # Top 3 predicciones
        st.markdown("### 📊 Top 3 Predicciones:")
        
        # Obtener índices de las 3 clases con mayor probabilidad
        top_3_indices = np.argsort(st.session_state.predicciones)[-3:][::-1]
        
        for i, idx in enumerate(top_3_indices, 1):
            probabilidad = st.session_state.predicciones[idx] * 100
            nombre = CLASES_SEÑALES[idx]
            
            col_num, col_nombre, col_prob = st.columns([0.5, 3, 1])
            with col_num:
                st.markdown(f"**{i}.**")
            with col_nombre:
                st.markdown(f"{nombre}")
            with col_prob:
                st.markdown(f"`{probabilidad:.1f}%`")
    
    else:
        st.info("👆 Carga una imagen y presiona 'Analizar' para ver los resultados")

# ============================================
# SECCIÓN ADICIONAL: LISTA DE SEÑALES
# ============================================
st.markdown("---")
st.subheader("📋 Lista Completa de Señales Reconocidas")

with st.expander("Ver todas las señales (43 clases)"):
    # Mostrar en 3 columnas
    cols = st.columns(3)
    
    for idx, nombre in CLASES_SEÑALES.items():
        col_idx = idx % 3
        with cols[col_idx]:
            st.markdown(f"**{idx}.** {nombre}")

# ============================================
# PIE DE PÁGINA
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Desarrollado con ❤️ usando TensorFlow y Streamlit</p>
    <p>Proyecto Universitario - 2025</p>
</div>
""", unsafe_allow_html=True)