# src/training/train_model.py
from ultralytics import YOLO
import logging
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

def train_model(data_path: str, num_epochs: int = 100, batch_size: int = 16):
    """
    Train YOLO model on Nike logo dataset
    Args:
        data_path: Path to dataset directory
        num_epochs: Number of training epochs
        batch_size: Batch size for training
    """
    try:
        # Get project root and verify paths
        project_root = Path(__file__).resolve().parent.parent.parent
        data_path = project_root / data_path
        
        # Verify data structure
        train_images = list((data_path / 'train' / 'images').glob('*.jpg'))
        train_labels = list((data_path / 'train' / 'labels').glob('*.txt'))
        
        if not train_images:
            raise FileNotFoundError("No training images found!")
        if not train_labels:
            raise FileNotFoundError("No training labels found!")
            
        logger.info(f"Found {len(train_images)} training images and {len(train_labels)} labels")
        
        # Create simple dataset config
        dataset_config = {
            'path': str(data_path),
            'train': str(data_path / 'train' / 'images'),
            'val': str(data_path / 'valid' / 'images'),
            'test': str(data_path / 'test' / 'images'),
            'names': {0: 'nike_logo'},
            'nc': 1
        }
        
        # Save config
        config_path = data_path / 'dataset.yaml'
        with open(config_path, 'w') as f:
            yaml.safe_dump(dataset_config, f, sort_keys=False)
        
        # Initialize and train model
        model = YOLO('yolov8n.pt')
        
        results = model.train(
            data=str(config_path),
            epochs=num_epochs,
            imgsz=640,
            batch=batch_size,
            name='nike_detector_yolo',
            project=str(project_root / 'models' / 'trained'),
            exist_ok=True
        )
        
        # Save final model
        final_model_path = project_root / 'models' / 'trained' / 'nike_detector.pt'
        model.save(str(final_model_path))
        logger.info(f"Model saved to {final_model_path}")
        
        return str(final_model_path)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise