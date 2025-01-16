# src/training/train_model.py
from ultralytics import YOLO
import torch
import optuna
from pathlib import Path
import yaml
import logging
import argparse

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

def train_model(data_path: str = None, num_epochs: int = 50, batch_size: int = 8, **kwargs):
    """Función principal de entrenamiento"""
    try:
        device = check_gpu()
        yaml_path = create_dataset_yaml()
        model = YOLO('yolov8n.pt')
        
        training_params = {
            'data': str(yaml_path),
            'epochs': num_epochs,
            'imgsz': 640,
            'batch': batch_size,
            'device': device,
            'cache': False,
            'workers': 2,
            'amp': False,
            'plots': True,
            'save': True,
            'optimizer': 'AdamW',
            'lr0': 0.0005,
            'lrf': 0.00001,
            'momentum': 0.937,
            'weight_decay': 0.001,
            'warmup_epochs': 5,
            'verbose': True,
            'single_cls': True,
            'deterministic': True,
            'seed': 42,
            'mosaic': 0.3,
            'degrees': 5.0,
            'translate': 0.1,
            'scale': 0.5,
            'fliplr': 0.5,
            'patience': 10,
            'cos_lr': True,
            'close_mosaic': 15
        }
        
        # Actualizar con kwargs si se proporcionan
        training_params.update(kwargs)
        
        results = model.train(**training_params)
        return results
        
    except Exception as e:
        logger.error(f"Error en entrenamiento: {str(e)}", exc_info=True)
        raise

def objective(trial):
    """Función objetivo para Optuna"""
    params = {
        'lr0': trial.suggest_float('lr0', 1e-5, 1e-1, log=True),
        'batch': trial.suggest_int('batch', 4, 32),
        'imgsz': trial.suggest_categorical('imgsz', [416, 512, 640]),
        'momentum': trial.suggest_float('momentum', 0.5, 0.99),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
        'warmup_epochs': trial.suggest_int('warmup_epochs', 1, 5),
        'mosaic': trial.suggest_float('mosaic', 0.0, 1.0),
        'degrees': trial.suggest_float('degrees', 0.0, 10.0)
    }
    
    try:
        results = train_model(num_epochs=10, **params)
        mAP50 = results.results_dict.get('metrics/mAP50(B)', 0)
        logger.info(f"Trial {trial.number} completed with mAP50: {mAP50}")
        return mAP50
    except Exception as e:
        logger.error(f"Error in trial {trial.number}: {str(e)}")
        return 0

def run_optimization(n_trials=20):
    """Ejecutar la optimización de hiperparámetros"""
    study = optuna.create_study(
        study_name="nike_logo_detection",
        direction="maximize",
        storage="sqlite:///nike_logo_study.db",
        load_if_exists=True
    )
    
    study.optimize(objective, n_trials=n_trials)
    
    logger.info("\nMejores hiperparámetros encontrados:")
    logger.info(study.best_params)
    logger.info(f"\nMejor valor de mAP50: {study.best_value}")
    
    params_path = PROJECT_ROOT / 'models' / 'best_params.yaml'
    with open(params_path, 'w') as f:
        yaml.dump(study.best_params, f)
    
    return study.best_params

def main():
    parser = argparse.ArgumentParser(description='Entrenamiento y optimización de modelo YOLO')
    parser.add_argument('--optimize', action='store_true', help='Ejecutar optimización de hiperparámetros')
    parser.add_argument('--trials', type=int, default=20, help='Número de trials para optimización')
    args = parser.parse_args()
    
    if args.optimize:
        logger.info("Iniciando optimización de hiperparámetros...")
        best_params = run_optimization(n_trials=args.trials)
        # Entrenar modelo final con mejores parámetros
        train_model(**best_params)
    else:
        logger.info("Iniciando entrenamiento normal...")
        train_model()

if __name__ == "__main__":
    main()