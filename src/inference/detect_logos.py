# src/inference/detect_logos.py
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import os
import logging
from datetime import timedelta
import sqlite3
import torch
from torchvision.transforms import functional as F
from config import *
# Aquí van las clases:
# - DetectionVisualizer
# - DetectionStorage
# - VideoProcessor

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
    
class DetectionVisualizer:
    def __init__(self, class_names=['nike']):
        self.class_names = class_names
        
    def draw_detections(self, image, boxes, labels, scores=None):
        """
        Dibuja los bounding boxes y etiquetas en la imagen
        Args:
            image: Imagen numpy array (BGR)
            boxes: tensor de bounding boxes [x1, y1, x2, y2]
            labels: tensor de etiquetas de clase
            scores: tensor de scores de confianza (opcional)
        """
        image_copy = image.copy()
        
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        if scores is not None and isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
            
        for idx, (box, label) in enumerate(zip(boxes, labels)):
            x1, y1, x2, y2 = map(int, box)
            
            # Dibujar bounding box
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Preparar texto con clase y score
            text = self.class_names[label]
            if scores is not None:
                text += f" {scores[idx]:.2f}"
                
            # Calcular posición del texto (debajo del bbox)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = x1
            text_y = y2 + text_size[1] + 5
            
            # Dibujar fondo para el texto
            cv2.rectangle(image_copy, 
                        (text_x, text_y - text_size[1] - 5),
                        (text_x + text_size[0], text_y + 5),
                        (0, 255, 0), 
                        -1)
            
            # Dibujar texto
            cv2.putText(image_copy, 
                       text,
                       (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.7,
                       (0, 0, 0),
                       2)
            
        return image_copy
    
class DetectionStorage:
    def __init__(self, db_path, saves_dir):
        self.db_path = db_path
        self.saves_dir = saves_dir
        os.makedirs(saves_dir, exist_ok=True)
        self.init_db()
        
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Crear tablas si no existen
        c.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_file TEXT,
                class_name TEXT,
                confidence REAL,
                timestamp DATETIME,
                bbox_image_path TEXT,
                x1 REAL,
                y1 REAL,
                x2 REAL,
                y2 REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def save_detection(self, image, box, class_name, score, source_file, frame_number=None):
        """
        Guarda una detección individual
        """
        # Crear nombre único para la imagen recortada
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if frame_number is not None:
            image_filename = f"{timestamp}_frame{frame_number}_{class_name}.jpg"
        else:
            image_filename = f"{timestamp}_{class_name}.jpg"
        
        image_path = os.path.join(self.saves_dir, image_filename)
        
        # Recortar y guardar la imagen del bbox
        x1, y1, x2, y2 = map(int, box)
        cropped_image = image[y1:y2, x1:x2]
        cv2.imwrite(image_path, cropped_image)
        
        # Guardar información en la base de datos
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            INSERT INTO detections 
            (source_file, class_name, confidence, timestamp, bbox_image_path, x1, y1, x2, y2)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            source_file,
            class_name,
            float(score) if score is not None else None,
            timestamp,
            image_path,
            float(x1),
            float(y1),
            float(x2),
            float(y2)
        ))
        
        conn.commit()
        conn.close()
        
class VideoProcessor:
    def __init__(self, model, visualizer, storage, device='cuda'):
        self.model = model
        self.visualizer = visualizer
        self.storage = storage
        self.device = device
        self.model.to(device)
        self.model.eval()
        
    def process_frame(self, frame):
        """Procesa un frame individual"""
        # Preparar imagen para el modelo
        image_tensor = F.to_tensor(frame).to(self.device)
        image_tensor = image_tensor.unsqueeze(0)
        
        with torch.no_grad():
            predictions = self.model(image_tensor)
            
        # Obtener predicciones
        boxes = predictions[0]['boxes']
        labels = predictions[0]['labels']
        scores = predictions[0]['scores']
        
        # Filtrar por umbral de confianza
        mask = scores > 0.5
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]
        
        # Dibujar detecciones
        frame_with_detections = self.visualizer.draw_detections(
            frame, boxes, labels, scores
        )
        
        return frame_with_detections, boxes, labels, scores
    
    def process_video(self, video_path, stframe):
        """
        Procesa un video completo y muestra los frames en Streamlit
        Args:
            video_path: ruta al video
            stframe: st.image() object de Streamlit para mostrar los frames
        """
        cap = cv2.VideoCapture(video_path)
        
        # Configurar writer para guardar el video procesado
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        output_path = video_path.rsplit('.', 1)[0] + '_processed.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Procesar frame
            frame_with_detections, boxes, labels, scores = self.process_frame(frame)
            
            # Guardar detecciones
            for box, label, score in zip(boxes, labels, scores):
                self.storage.save_detection(
                    frame,
                    box.cpu().numpy(),
                    self.visualizer.class_names[label],
                    score.item(),
                    video_path,
                    frame_count
                )
            
            # Mostrar frame en Streamlit
            frame_rgb = cv2.cvtColor(frame_with_detections, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", use_column_width=True)
            
            # Guardar frame procesado
            out.write(frame_with_detections)
            
            frame_count += 1
                
        cap.release()
        out.release()
        
# Opcional: main solo para pruebas locales
def main():
    # Este main() solo se usaría para pruebas locales
    # El video_path real vendrá del frontend
    pass

if __name__ == "__main__":
    main()