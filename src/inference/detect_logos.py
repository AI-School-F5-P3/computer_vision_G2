# src/inference/detect_logos.py
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import os
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)

class LogoDetector:
    def __init__(self, model_path=None):
        """Initialize YOLO model"""
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            self.model = YOLO('yolov8n.pt')
            logger.warning("No model found, initialized with base YOLO model")

    def detect(self, image: Image.Image, conf_threshold: float = 0.25) -> tuple:
        """Detect logos in image"""
        # Tu código actual de detección en imágenes
        results = self.model(image, conf=conf_threshold)
        
        boxes = []
        scores = []
        labels = []
        
        for r in results:
            boxes.extend(r.boxes.xyxy.cpu().numpy())
            scores.extend(r.boxes.conf.cpu().numpy())
            labels.extend(r.boxes.cls.cpu().numpy())
            
        return np.array(boxes), np.array(scores), np.array(labels)

    def process_video(self, video_path: str, conf_threshold: float = 0.25) -> dict:
        """
        Process video and detect logos
        Args:
            video_path: Path to video file
            conf_threshold: Confidence threshold for detections
        Returns:
            dict: Statistics about logo appearances
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        # Variables para seguimiento
        frame_count = 0
        logo_frames = 0
        frame_detections = []  # Guardamos las detecciones de cada frame
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Detectar logos
            boxes, scores, labels = self.detect(frame_pil, conf_threshold)
            
            # Guardar información del frame
            if len(boxes) > 0:
                logo_frames += 1
                frame_detections.append({
                    'frame_number': frame_count,
                    'timestamp': frame_count / fps,
                    'num_logos': len(boxes),
                    'scores': scores.tolist(),
                    'boxes': boxes.tolist()
                })
            
            frame_count += 1
            if frame_count % 30 == 0:
                logger.info(f"Procesado: {(frame_count/total_frames)*100:.1f}%")
        
        cap.release()
        
        # Procesar las detecciones para agrupar apariciones consecutivas
        appearances = []
        current_appearance = None
        
        for i, frame_info in enumerate(frame_detections):
            if current_appearance is None:
                current_appearance = {
                    'start_frame': frame_info['frame_number'],
                    'start_time': frame_info['timestamp'],
                    'max_logos': frame_info['num_logos'],
                    'frames_info': [frame_info]
                }
            else:
                # Si es el frame siguiente en la secuencia
                if frame_info['frame_number'] - current_appearance['frames_info'][-1]['frame_number'] <= 2:
                    current_appearance['frames_info'].append(frame_info)
                    current_appearance['max_logos'] = max(
                        current_appearance['max_logos'],
                        frame_info['num_logos']
                    )
                else:
                    # Finalizar aparición actual
                    end_frame = current_appearance['frames_info'][-1]
                    current_appearance['end_frame'] = end_frame['frame_number']
                    current_appearance['end_time'] = end_frame['timestamp']
                    current_appearance['duration'] = (
                        current_appearance['end_time'] - 
                        current_appearance['start_time']
                    )
                    appearances.append(current_appearance)
                    
                    # Iniciar nueva aparición
                    current_appearance = {
                        'start_frame': frame_info['frame_number'],
                        'start_time': frame_info['timestamp'],
                        'max_logos': frame_info['num_logos'],
                        'frames_info': [frame_info]
                    }
        
        # Añadir última aparición si existe
        if current_appearance is not None:
            end_frame = current_appearance['frames_info'][-1]
            current_appearance['end_frame'] = end_frame['frame_number']
            current_appearance['end_time'] = end_frame['timestamp']
            current_appearance['duration'] = (
                current_appearance['end_time'] - 
                current_appearance['start_time']
            )
            appearances.append(current_appearance)
        
        # Calcular estadísticas
        total_logo_time = sum(app['duration'] for app in appearances)
        max_logos_in_frame = max(app['max_logos'] for app in appearances) if appearances else 0
        
        return {
            'total_frames': total_frames,
            'logo_frames': logo_frames,
            'video_duration': duration,
            'total_logo_time': total_logo_time,
            'logo_percentage': (total_logo_time / duration) * 100,
            'max_logos_in_frame': max_logos_in_frame,
            'appearances': [
                {
                    'start_time': app['start_time'],
                    'end_time': app['end_time'],
                    'duration': app['duration'],
                    'max_logos': app['max_logos']
                }
                for app in appearances
            ]
        }

    def format_time(self, seconds: float) -> str:
        """Format seconds to HH:MM:SS"""
        return str(timedelta(seconds=int(seconds)))