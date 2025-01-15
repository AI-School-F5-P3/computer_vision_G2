# src/training/train_model.py
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

def create_dataset_yaml(data_path):
    """Crea el archivo dataset.yaml con las rutas correctas"""
    data_path = Path(data_path)
    
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

def train_model(data_path=None, num_epochs=10, batch_size=4):
    """
    Train the YOLO model with specified parameters
    
    Args:
        data_path (str): Path to the data directory
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
    
    Returns:
        str: Path to the trained model
    """
    try:
        if data_path is None:
            data_path = Path(__file__).resolve().parent.parent.parent / 'data'
        else:
            data_path = Path(data_path)

        print("1. Verificando GPU...", flush=True)
        device = check_gpu()
        
        print("2. Creando archivo YAML...", flush=True)
        yaml_path = create_dataset_yaml(data_path)
        
        print("3. Configurando modelo...", flush=True)
        model = YOLO('yolov8n.pt')
        
        print("4. Iniciando entrenamiento...", flush=True)
        results = model.train(
            data=str(yaml_path),
            epochs=num_epochs,
            imgsz=416,
            batch=batch_size,
            device=device,
            cache=False,
            workers=0,
            amp=False,
            plots=True,
            save=True,
            optimizer='AdamW',
            lr0=0.001,
            momentum=0.9,
            weight_decay=0.0005,
            warmup_epochs=1,
            verbose=True,
            single_cls=True,
            deterministic=True,
            seed=42,
            mosaic=0.0,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            patience=0
        )
        
        print("5. Entrenamiento completado", flush=True)
        
        # Return the path to the best model
        return str(model.trainer.best)  # This will return the path to the best model
        
    except Exception as e:
        print(f"Error: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    train_model()