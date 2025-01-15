# src/training/minimal_train.py
from ultralytics import YOLO
import torch
import yaml
from pathlib import Path

def check_gpu():
    """Verifica la disponibilidad y configuración de la GPU"""
    print("\nVerificando GPU:")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
        print(f"Versión CUDA: {torch.version.cuda}")
        print(f"Número de GPUs: {torch.cuda.device_count()}")
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def create_dataset_yaml():
    """Crea el archivo dataset.yaml con las rutas correctas"""
    project_root = Path(__file__).resolve().parent.parent.parent
    data_path = project_root / 'data'
    
    dataset_config = {
        'path': str(data_path),
        'train': str(data_path / 'train' / 'images'),
        'val': str(data_path / 'valid' / 'images'),
        'test': str(data_path / 'test' / 'images'),
        'names': {0: 'nike_logo'},
        'nc': 1
    }
    
    yaml_path = data_path / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(dataset_config, f, sort_keys=False)
    
    print(f"Dataset YAML creado en: {yaml_path}", flush=True)
    return yaml_path

def train_minimal():
    try:
        print("1. Verificando GPU...", flush=True)
        device = check_gpu()
        
        print("2. Creando archivo YAML...", flush=True)
        yaml_path = create_dataset_yaml()
        
        print("3. Configurando modelo...", flush=True)
        model = YOLO('yolov8n.pt')
        
        print("4. Iniciando entrenamiento...", flush=True)
        model.train(
        data=str(yaml_path),
        epochs=30,              # Aumentado a 30 épocas
        imgsz=640,             # Volvemos a 640 para mejor detección
        batch=8,               # Aumentado pero manteniendo estabilidad
        device=device,
        cache=False,
        workers=2,             # Añadimos algunos workers ya que la GPU funciona
        amp=False,             # Mantenemos AMP desactivado
        plots=True,
        save=True,
        optimizer='AdamW',
        lr0=0.001,            # Learning rate inicial
        lrf=0.0001,           # Learning rate final
        momentum=0.937,        # Momentum ajustado
        weight_decay=0.0005,
        warmup_epochs=3,       # Más épocas de warmup
        verbose=True,
        single_cls=True,
        deterministic=True,
        seed=42,
        mosaic=0.5,           # Activamos mosaic parcialmente
        degrees=10.0,         # Añadimos algo de rotación
        translate=0.2,        # Más traducción
        scale=0.5,
        fliplr=0.5,
        patience=5,           # Early stopping si no hay mejora
        cos_lr=True,          # Learning rate con curva coseno
        close_mosaic=10,      # Desactivar mosaic en las últimas épocas
        overlap_mask=False    # Mantener simple
    )
        
        print("5. Entrenamiento completado", flush=True)
        
    except Exception as e:
        print(f"Error: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_minimal()