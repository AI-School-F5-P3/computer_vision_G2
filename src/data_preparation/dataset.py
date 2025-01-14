# src/data_preparation/dataset.py
import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from pathlib import Path

class LogoDataset(Dataset):
    def __init__(self, dataset_path, split='train', transform=None):
        """
        Args:
            dataset_path: Ruta base del dataset
            split: 'train', 'test' o 'valid'
            transform: Transformaciones a aplicar a las imágenes
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.transform = transform
        
        # Definir rutas
        self.images_path = self.dataset_path / split / 'images'
        self.labels_path = self.dataset_path / split / 'labels'
        
        # Obtener lista de imágenes
        self.image_files = sorted([f for f in self.images_path.glob('*.jpg') or self.images_path.glob('*.png')])
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        # Cargar imagen
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Cargar etiqueta correspondiente
        label_path = self.labels_path / f"{img_path.stem}.txt"
        boxes = []
        labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    data = line.strip().split()
                    # Formato YOLO: class x_center y_center width height
                    class_id = int(data[0])
                    x_center, y_center, width, height = map(float, data[1:])
                    
                    # Convertir formato YOLO a coordenadas absolutas
                    h, w = image.shape[:2]
                    x1 = (x_center - width/2) * w
                    y1 = (y_center - height/2) * h
                    x2 = (x_center + width/2) * w
                    y2 = (y_center + height/2) * h
                    
                    boxes.append([x1, y1, x2, y2])
                    labels.append(class_id)
        
        # Convertir a tensores
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        if self.transform:
            image = self.transform(image)
        
        return image, target