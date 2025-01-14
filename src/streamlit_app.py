import streamlit as st
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import logging
from ultralytics import YOLO
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LogoDetector:
    def __init__(self, model_path=None):
        """Initialize YOLO model"""
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            # If no model exists, create a new one from YOLO base model
            self.model = YOLO('yolov8n.pt')
    
    def train(self, data_path: str, epochs: int = 100):
        """Train the model"""
        # Create dataset config
        dataset_config = {
            'path': data_path,
            'train': os.path.join(data_path, 'train/images'),
            'val': os.path.join(data_path, 'val/images'),
            'test': os.path.join(data_path, 'test/images'),
            'names': {0: 'nike_logo'}
        }
        
        # Save dataset config
        config_path = os.path.join(data_path, 'dataset.yaml')
        with open(config_path, 'w') as f:
            import yaml
            yaml.dump(dataset_config, f)
        
        # Train
        self.model.train(
            data=config_path,
            epochs=epochs,
            imgsz=640,
            batch=16,
            name='nike_detector_yolo'
        )
        
        # Save model
        output_path = 'models/trained'
        os.makedirs(output_path, exist_ok=True)
        self.model.save(os.path.join(output_path, 'nike_detector.pt'))

    def detect(self, image: Image.Image, conf_threshold: float = 0.25) -> tuple:
        """Detect logos in image"""
        results = self.model(image, conf=conf_threshold)
        
        boxes = []
        scores = []
        labels = []
        
        for r in results:
            boxes.extend(r.boxes.xyxy.cpu().numpy())
            scores.extend(r.boxes.conf.cpu().numpy())
            labels.extend(r.boxes.cls.cpu().numpy())
            
        return np.array(boxes), np.array(scores), np.array(labels)

def main():
    st.set_page_config(page_title="Nike Logo Detector", layout="wide")
    
    st.title("Nike Logo Detector")
    
    # Sidebar
    st.sidebar.header("Configuration")
    page = st.sidebar.radio("Select Page", ["Train Model", "Detect Logos"])
    
    # Initialize detector
    @st.cache_resource
    def load_detector():
        model_path = "models/trained/nike_detector.pt"
        return LogoDetector(model_path)
    
    try:
        detector = load_detector()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        detector = LogoDetector()  # Initialize without model
    
    if page == "Train Model":
        st.header("Train Model")
        
        # Training parameters
        epochs = st.number_input("Number of Epochs", min_value=1, value=100)
        
        # Training button
        if st.button("Start Training"):
            try:
                with st.spinner("Training model... This may take a while."):
                    detector.train('data', epochs=epochs)
                st.success("Training completed! Model saved in models/trained/nike_detector.pt")
            except Exception as e:
                st.error(f"Training error: {str(e)}")
                logger.error(f"Training error: {str(e)}", exc_info=True)
    
    else:  # Detect Logos page
        st.header("Detect Logos")
        
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.25,
            step=0.05
        )
        
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

if __name__ == "__main__":
    main()