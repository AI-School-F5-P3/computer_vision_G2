import streamlit as st
import cv2
import torch
import numpy as np
from pathlib import Path
import sys
from PIL import Image
import io

# Añadir el directorio src al path
sys.path.append(str(Path(__file__).parent.parent))

from inference.detect_logos import detect_logos
from data_preparation.download_nike_images import download_nike_logos

def main():
    st.title("Detector de Logos Nike")
    
    # Sidebar para configuración
    st.sidebar.header("Configuración")
    confidence_threshold = st.sidebar.slider("Umbral de confianza", 0.0, 1.0, 0.5)
    
    # Menú principal
    menu = ["Inicio", "Descargar Dataset", "Detectar Logos"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Inicio":
        st.write("""
        # Bienvenido al Detector de Logos Nike
        
        Esta aplicación te permite:
        1. Descargar un dataset de logos de Nike
        2. Detectar logos en imágenes
        
        Selecciona una opción del menú para comenzar.
        """)
        
    elif choice == "Descargar Dataset":
        st.header("Descargar Dataset de Nike")
        num_images = st.number_input("Número de imágenes a descargar", min_value=1, value=50)
        
        if st.button("Descargar Dataset"):
            output_dir = "data/raw/nike_logos"
            with st.spinner('Descargando imágenes...'):
                download_nike_logos(output_dir, num_images)
            st.success(f"Dataset descargado en {output_dir}")
            
    elif choice == "Detectar Logos":
        st.header("Detección de Logos")
        uploaded_file = st.file_uploader("Sube una imagen", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            # Convertir archivo subido a imagen
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagen subida", use_column_width=True)
            
            if st.button("Detectar Logos"):
                with st.spinner('Detectando logos...'):
                    # Guardar imagen temporalmente
                    img_path = "temp_image.jpg"
                    image.save(img_path)
                    
                    # Realizar detección
                    try:
                        model_path = "models/trained/nike_detector.pth"
                        boxes, scores, labels = detect_logos(img_path, model_path, confidence_threshold)
                        
                        # Visualizar resultados
                        img = cv2.imread(img_path)
                        for box, score in zip(boxes, scores):
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img, f'Nike: {score:.2f}', (x1, y1-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                        st.image(img, caption="Detecciones", use_column_width=True)
                        
                    except Exception as e:
                        st.error(f"Error en la detección: {str(e)}")
                    
                    # Limpiar archivo temporal
                    Path(img_path).unlink(missing_ok=True)

if __name__ == "__main__":
    main()
