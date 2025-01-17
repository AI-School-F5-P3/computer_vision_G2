# src/training/hyperparameter_search.py
import optuna
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
        model = YOLO('yolov8n.pt')
        results = model.train(
            data=str(YAML_PATH),
            epochs=10,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            **params
        )
        mAP50 = results.results_dict.get('metrics/mAP50(B)', 0)
        logger.info(f"Trial {trial.number} completed with mAP50: {mAP50}")
        return mAP50
    except Exception as e:
        logger.error(f"Error in trial {trial.number}: {str(e)}")
        return 0

def run_optimization(n_trials=20):
    """Ejecutar la optimización de hiperparámetros"""
    # Crear archivo dataset.yaml si no existe
    if not YAML_PATH.exists():
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
    
    # Guardar mejores parámetros
    params_path = PROJECT_ROOT / 'models' / 'best_params.yaml'
    params_path.parent.mkdir(parents=True, exist_ok=True)
    with open(params_path, 'w') as f:
        yaml.dump(study.best_params, f)
    
    return study.best_params

if __name__ == "__main__":
    logger.info("Iniciando búsqueda de hiperparámetros...")
    run_optimization()