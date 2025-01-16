# src/streamlit_app.py
import streamlit as st
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import logging
import sys
import os
import tempfile
from datetime import timedelta

# Get the project root directory
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))

from src.training.train_model import train_model
from src.inference.detect_logos import LogoDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_time(seconds):
    """Formato tiempo en HH:MM:SS"""
    return str(timedelta(seconds=int(seconds)))

def process_video_frames(video_path, detector, conf_threshold=0.25):
    """
    Procesa el video frame por frame y detecta logos
    Returns:
        dict con estadísticas y detecciones
    """
    cap = cv2.VideoCapture(video_path)
    
    # Obtener propiedades del video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    # Variables para seguimiento
    frame_count = 0
    logo_frames = 0
    frame_detections = []  # Lista para guardar detecciones por frame
    progress_bar = st.progress(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convertir frame a RGB y formato PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        
        # Detectar logos en el frame
        boxes, scores, labels = detector.detect(frame_pil, conf_threshold)
        
        if len(boxes) > 0:
            logo_frames += 1
            frame_detections.append({
                'frame_number': frame_count,
                'timestamp': frame_count / fps,
                'num_logos': len(boxes),
                'boxes': boxes.tolist(),
                'scores': scores.tolist()
            })
        
        # Actualizar progreso
        if frame_count % 30 == 0:  # Actualizar cada 30 frames
            progress = frame_count / total_frames
            progress_bar.progress(progress)
        
        frame_count += 1
    
    cap.release()
    progress_bar.empty()
    
    # Procesar estadísticas
    total_detections = sum(frame['num_logos'] for frame in frame_detections)
    avg_logos_per_frame = total_detections / logo_frames if logo_frames > 0 else 0
    
    return {
        'total_frames': total_frames,
        'logo_frames': logo_frames,
        'fps': fps,
        'duration': duration,
        'logo_percentage': (logo_frames / total_frames) * 100,
        'frame_detections': frame_detections,
        'total_detections': total_detections,
        'avg_logos_per_frame': avg_logos_per_frame
    }

def main():
    st.set_page_config(page_title="Nike Logo Detector", layout="wide")
    
    st.title("Nike Logo Detector")
    
    # Sidebar
    st.sidebar.header("Configuration")
    page = st.sidebar.radio("Select Page", ["Train Model", "Detect Logos"])
    
    # Get project root directory
    project_root = Path(__file__).parent.parent
    
    # Initialize detector
    @st.cache_resource
    def load_detector():
        model_path = project_root / "runs" / "detect" / "train7" / "best.pt"
        return LogoDetector(model_path if model_path.exists() else None)
    
    try:
        detector = load_detector()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        detector = LogoDetector()  # Initialize without model
    
    if page == "Train Model":
        st.header("Train Model")
        
        # Check if data directory exists and has required structure
        data_path = project_root / 'data'
        if not data_path.exists():
            st.error(f"Data directory not found at {data_path}")
            return
            
        required_dirs = ['train/images', 'valid/images', 'test/images']
        missing_dirs = []
        for dir_path in required_dirs:
            if not (data_path / dir_path).exists():
                missing_dirs.append(dir_path)
                
        if missing_dirs:
            st.error(f"Missing required directories: {', '.join(missing_dirs)}")
            st.write("Please ensure your data directory has the following structure:")
            st.code("""
data/
├── train/
│   └── images/
├── valid/
│   └── images/
└── test/
    └── images/
            """)
            return
        
        # Training parameters
        epochs = st.number_input("Number of Epochs", min_value=1, value=10)
        batch_size = st.number_input("Batch Size", min_value=1, value=16)
        
        # Training button
        if st.button("Start Training"):
            try:
                with st.spinner("Training model... This may take a while."):
                    model_path = train_model(
                        data_path='data',  # Use relative path
                        num_epochs=epochs,
                        batch_size=batch_size
                    )
                st.success(f"Training completed! Model saved at {model_path}")
                # Reload the detector with new model
                detector = load_detector()
            except Exception as e:
                st.error(f"Training error: {str(e)}")
                logger.error(f"Training error: {str(e)}", exc_info=True)
    
    else:  # Detect Logos page
        st.header("Detect Logos")
        
        # Mode selection
        mode = st.radio("Select Detection Mode", ["Image", "Video"])
        
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.25,
            step=0.05
        )
        
        if mode == "Image":
            uploaded_file = st.file_uploader(
                "Choose an image file", 
                type=['png', 'jpg', 'jpeg']
            )
            
            if uploaded_file is not None:
                try:
                    # Display original image
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                    
                    if st.button("Detect Logos"):
                        with st.spinner("Performing detection..."):
                            boxes, scores, labels = detector.detect(
                                image, 
                                conf_threshold=confidence_threshold
                            )
                            
                            if len(boxes) > 0:
                                # Draw detections
                                img_array = np.array(image)
                                for box, score in zip(boxes, scores):
                                    x1, y1, x2, y2 = map(int, box)
                                    cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.putText(img_array, f'Nike: {score:.2f}', 
                                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                              0.9, (0, 255, 0), 2)
                                
                                st.image(img_array, caption="Detection Results", use_column_width=True)
                                
                                # Statistics
                                st.write(f"Found {len(boxes)} Nike logo(s)")
                                for i, score in enumerate(scores, 1):
                                    st.write(f"Detection {i}: Confidence = {score:.2%}")
                            else:
                                st.info("No Nike logos detected in the image.")
                
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    logger.error(f"Processing error: {str(e)}", exc_info=True)
                    
        else:  # Video mode
            uploaded_file = st.file_uploader(
                "Choose a video file",
                type=['mp4', 'avi', 'mov']
            )
            
            if uploaded_file is not None:
                try:
                    # Crear archivo temporal
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        video_path = tmp_file.name
                    
                    if st.button("Analyze Video"):
                        with st.spinner("Processing video..."):
                            # Procesar video
                            results = process_video_frames(
                                video_path,
                                detector,
                                conf_threshold=confidence_threshold
                            )
                            
                            # Mostrar resultados
                            st.subheader("Video Analysis Results")
                            
                            # Métricas principales
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Video Duration", 
                                         format_time(results['duration']))
                            with col2:
                                st.metric("Frames with Logos", 
                                         f"{results['logo_frames']} / {results['total_frames']}")
                            with col3:
                                st.metric("Logo Presence", 
                                         f"{results['logo_percentage']:.1f}%")
                            
                            # Estadísticas adicionales
                            st.subheader("Detection Statistics")
                            st.write(f"Total logo detections: {results['total_detections']}")
                            st.write(f"Average logos per frame: {results['avg_logos_per_frame']:.2f}")
                            
                            # Gráfico de detecciones a lo largo del tiempo
                            if len(results['frame_detections']) > 0:
                                st.subheader("Logo Detections Timeline")
                                timeline_data = [
                                    {"time": format_time(det['timestamp']), 
                                     "logos": det['num_logos']} 
                                    for det in results['frame_detections']
                                ]
                                st.line_chart(
                                    data=timeline_data,
                                    x="time",
                                    y="logos"
                                )
                            
                            # Detalles por frame
                            st.subheader("Detailed Detections")
                            for det in results['frame_detections']:
                                with st.expander(
                                    f"Time {format_time(det['timestamp'])} - {det['num_logos']} logos"):
                                    st.write(f"Frame number: {det['frame_number']}")
                                    st.write(f"Number of logos: {det['num_logos']}")
                                    for i, (box, score) in enumerate(zip(det['boxes'], det['scores'])):
                                        st.write(f"Logo {i+1}: Confidence = {score:.2f}")
                    
                    # Limpiar archivo temporal
                    os.unlink(video_path)
                    
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                    logger.error(f"Video processing error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()