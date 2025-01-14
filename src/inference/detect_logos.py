# src/inference/detect_logos.py
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os
import logging

logger = logging.getLogger(__name__)

class LogoDetector:
    def __init__(self, model_path=None):
        """Initialize YOLO model"""
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            # If no model exists, create a new one from YOLO base model
            self.model = YOLO('yolov8n.pt')
            logger.warning("No model found, initialized with base YOLO model")

    def detect(self, image: Image.Image, conf_threshold: float = 0.25) -> tuple:
        """
        Detect logos in image
        Args:
            image: PIL Image
            conf_threshold: Confidence threshold
        Returns:
            tuple: (boxes, scores, labels)
        """
        results = self.model(image, conf=conf_threshold)
        
        boxes = []
        scores = []
        labels = []
        
        for r in results:
            boxes.extend(r.boxes.xyxy.cpu().numpy())
            scores.extend(r.boxes.conf.cpu().numpy())
            labels.extend(r.boxes.cls.cpu().numpy())
            
        return np.array(boxes), np.array(scores), np.array(labels)