# src/training/train_model.py
from ultralytics import YOLO
import os
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
    # Convert to Path object for easier path manipulation
    data_path = Path(data_path)
    
    # Create dataset config
    dataset_config = {
        'path': str(data_path.absolute()),  # Use absolute path
        'train': str((data_path / 'train' / 'images').absolute()),
        'val': str((data_path / 'valid' / 'images').absolute()),  # Note: using 'valid' instead of 'val'
        'test': str((data_path / 'test' / 'images').absolute()),
        'names': {0: 'nike_logo'},
        'nc': 1  # number of classes
    }
    
    # Create directory for config if it doesn't exist
    config_dir = Path('models/config')
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Save dataset config
    config_path = config_dir / 'dataset.yaml'
    with open(config_path, 'w') as f:
        yaml.safe_dump(dataset_config, f, sort_keys=False)
    
    logger.info(f"Created dataset config at {config_path}")
    logger.info(f"Training data path: {dataset_config['train']}")
    logger.info(f"Validation data path: {dataset_config['val']}")
    logger.info(f"Test data path: {dataset_config['test']}")
    
    # Initialize model
    model = YOLO('yolov8n.pt')
    
    # Create output directory if it doesn't exist
    output_path = Path('models/trained')
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Train model
    logger.info("Starting training...")
    try:
        model.train(
            data=str(config_path),
            epochs=num_epochs,
            imgsz=640,
            batch=batch_size,
            name='nike_detector_yolo',
            project=str(output_path),
            exist_ok=True
        )
        
        # Save model
        final_model_path = output_path / 'nike_detector.pt'
        model.save(str(final_model_path))
        logger.info(f"Model saved to {final_model_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise e

    return str(final_model_path)