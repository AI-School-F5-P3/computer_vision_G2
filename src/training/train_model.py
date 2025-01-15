# src/training/train_model.py
from ultralytics import YOLO
import logging
from pathlib import Path
import yaml
import sys
import traceback

# Configurar logging más detallado
logging.basicConfig(
    level=logging.DEBUG,  # Cambiar a DEBUG para más detalles
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def train_model(data_path: str, num_epochs: int = 50, batch_size: int = 8):
    """
    Train YOLO model with detailed error tracking
    """
    try:
        logger.info("Iniciando proceso de entrenamiento...")
        project_root = Path(__file__).resolve().parent.parent.parent
        data_path = project_root / data_path
        
        # Verificar memoria disponible
        logger.info("Verificando recursos del sistema...")
        try:
            import psutil
            mem = psutil.virtual_memory()
            logger.info(f"Memoria total: {mem.total / (1024**3):.2f} GB")
            logger.info(f"Memoria disponible: {mem.available / (1024**3):.2f} GB")
            logger.info(f"Memoria en uso: {mem.percent}%")
        except ImportError:
            logger.warning("No se pudo importar psutil para verificar memoria")

        # Configuración del dataset
        logger.info("Creando configuración del dataset...")
        dataset_config = {
            'path': str(data_path),
            'train': str(data_path / 'train' / 'images'),
            'val': str(data_path / 'valid' / 'images'),
            'test': str(data_path / 'test' / 'images'),
            'names': {0: 'nike_logo'},
            'nc': 1
        }

        config_path = data_path / 'dataset.yaml'
        with open(config_path, 'w') as f:
            yaml.safe_dump(dataset_config, f, sort_keys=False)
        logger.info(f"Configuración guardada en {config_path}")

        # Inicializar modelo con try/except específico
        logger.info("Inicializando modelo...")
        try:
            model = YOLO('yolov8n.pt')
        except Exception as e:
            logger.error(f"Error al inicializar modelo: {str(e)}")
            logger.error(traceback.format_exc())
            raise

        # Configurar parámetros de entrenamiento
        logger.info("Configurando parámetros de entrenamiento...")
        train_args = dict(
            data=str(config_path),
            epochs=num_epochs,
            imgsz=640,
            batch=batch_size,
            patience=20,
            device='cpu',
            cache='disk',
            name='nike_detector_yolo',
            project=str(project_root / 'models' / 'trained'),
            exist_ok=True,
            pretrained=True,
            verbose=True,
            single_cls=True,
            rect=False,
            resume=False,
            optimizer='SGD',
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            plots=True,
            save=True,
            save_period=5,
            workers=4
        )

        # Iniciar entrenamiento con manejo de excepciones específico
        logger.info("Iniciando entrenamiento...")
        try:
            results = model.train(**train_args)
            logger.info("Entrenamiento completado exitosamente")
        except KeyboardInterrupt:
            logger.info("Entrenamiento interrumpido por el usuario")
            raise
        except Exception as e:
            logger.error(f"Error durante el entrenamiento: {str(e)}")
            logger.error(traceback.format_exc())
            raise

        # Guardar modelo
        logger.info("Guardando modelo final...")
        final_model_path = project_root / 'models' / 'trained' / 'nike_detector.pt'
        try:
            model.save(str(final_model_path))
            logger.info(f"Modelo guardado en {final_model_path}")
        except Exception as e:
            logger.error(f"Error al guardar el modelo: {str(e)}")
            logger.error(traceback.format_exc())
            raise

        return str(final_model_path)

    except Exception as e:
        logger.error(f"Error general en el proceso: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    try:
        logger.info("Iniciando script de entrenamiento...")
        model_path = train_model('data', num_epochs=50, batch_size=8)
        logger.info(f"Proceso completado. Modelo guardado en: {model_path}")
    except Exception as e:
        logger.error(f"Error en el script principal: {str(e)}")
        logger.error(traceback.format_exc())