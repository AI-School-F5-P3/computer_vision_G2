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

# Get the project root directory (one level up from this file)
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))

from src.inference.detect_logos import LogoDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        video_path = tmp_file.name
                    
                    if st.button("Analyze Video"):
                        with st.spinner("Processing video..."):
                            # Process video
                            results = detector.process_video(
                                video_path,
                                conf_threshold=confidence_threshold
                            )
                            
                            # Display results
                            st.subheader("Video Analysis Results")
                            
                            # Main metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Video Duration", 
                                         detector.format_time(results['video_duration']))
                            with col2:
                                st.metric("Total Logo Time", 
                                         detector.format_time(results['total_logo_time']))
                            with col3:
                                st.metric("Logo Presence", 
                                         f"{results['logo_percentage']:.1f}%")
                            
                            # Detailed appearances
                            st.subheader("Logo Appearances")
                            for i, app in enumerate(results['appearances'], 1):
                                with st.expander(f"Appearance {i}"):
                                    st.write(f"Start: {detector.format_time(app['start_time'])}")
                                    st.write(f"End: {detector.format_time(app['end_time'])}")
                                    st.write(f"Duration: {detector.format_time(app['duration'])}")
                                    st.write(f"Logos detected: {app['detections']}")
                    
                    # Clean up
                    os.unlink(video_path)
                    
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                    logger.error(f"Video processing error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()