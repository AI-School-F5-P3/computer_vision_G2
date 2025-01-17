# src/training/train_model.py
from ultralytics import YOLO
import torch
from pathlib import Path
import yaml
import logging

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Variables globales
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / 'data'
YAML_PATH = DATA_PATH / 'dataset.yaml'

def check_gpu():
    """Verifica la disponibilidad y configuración de la GPU"""
    logger.info("\nVerificando GPU:")
    is_cuda = torch.cuda.is_available()
    logger.info(f"CUDA disponible: {is_cuda}")
    if is_cuda:
        logger.info(f"GPU detectada: {torch.cuda.get_device_name(0)}")
        logger.info(f"Versión CUDA: {torch.version.cuda}")
        logger.info(f"Número de GPUs: {torch.cuda.device_count()}")
    return 'cuda' if is_cuda else 'cpu'

def create_dataset_yaml():
    """Crea el archivo dataset.yaml con las rutas correctas"""
    dataset_config = {
        'path': str(DATA_PATH),
        'train': str(DATA_PATH / 'train' / 'images'),
        'val': str(DATA_PATH / 'valid' / 'images'),
        'test': str(DATA_PATH / 'test' / 'images'),
        'names': {0: 'nike_logo'},
        'nc': 1
    }
    
    with open(YAML_PATH, 'w') as f:
        yaml.safe_dump(dataset_config, f, sort_keys=False)
    
    logger.info(f"Dataset YAML creado en: {YAML_PATH}")
    return YAML_PATH

def load_best_params():
    """Carga los mejores hiperparámetros si existen"""
    params_path = PROJECT_ROOT / 'models' / 'best_params.yaml'
    if params_path.exists():
        with open(params_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def train_model():
    """Entrena el modelo usando los mejores hiperparámetros"""
    try:
        device = check_gpu()
        yaml_path = create_dataset_yaml()
        model = YOLO('yolov8n.pt')
        
        # Cargar mejores parámetros
        best_params = load_best_params()
        logger.info(f"Usando parámetros: {best_params}")
        
        # Parámetros base
        training_params = {
            'data': str(yaml_path),
            'device': device,
            'epochs': 50,
            'batch': 8,
            'imgsz': 640,
            'cache': False,
            'workers': 2,
            'amp': False,
            'plots': True,
            'save': True,
            'optimizer': 'AdamW',
            'verbose': True,
            'single_cls': True,
            'deterministic': True,
            'seed': 42
        }
        
        # Actualizar con mejores parámetros si existen
        if best_params:
            training_params.update(best_params)
        
        results = model.train(**training_params)
        return results
        
    except Exception as e:
        logger.error(f"Error en entrenamiento: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    logger.info("Iniciando entrenamiento con mejores parámetros...")
    train_model()