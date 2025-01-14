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
    # Get the absolute path of the current file
    current_file = Path(__file__).resolve()
    logger.info(f"Current file location: {current_file}")
    
    # Get the project root directory (3 levels up from this file: src/training/train_model.py -> src -> root)
    project_root = current_file.parent.parent.parent
    logger.info(f"Project root directory: {project_root}")
    
    # Convert data_path to absolute path from project root
    data_path = (project_root / data_path).resolve()
    logger.info(f"Full data path: {data_path}")
    
    # Print all directories in data path
    if data_path.exists():
        logger.info("Contents of data directory:")
        for item in data_path.iterdir():
            logger.info(f"- {item}")
            if item.is_dir():
                for subitem in item.iterdir():
                    logger.info(f"  - {subitem}")
    else:
        logger.error(f"Data directory does not exist: {data_path}")
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    # Verify required directories exist
    train_path = data_path / 'train' / 'images'
    valid_path = data_path / 'valid' / 'images'
    test_path = data_path / 'test' / 'images'
    
    logger.info(f"Checking paths:")
    logger.info(f"Train path exists: {train_path.exists()} - {train_path}")
    logger.info(f"Valid path exists: {valid_path.exists()} - {valid_path}")
    logger.info(f"Test path exists: {test_path.exists()} - {test_path}")
    
    # Check if directories exist
    missing_dirs = []
    if not train_path.exists():
        missing_dirs.append(str(train_path))
    if not valid_path.exists():
        missing_dirs.append(str(valid_path))
    if not test_path.exists():
        missing_dirs.append(str(test_path))
        
    if missing_dirs:
        error_msg = f"Missing required directories:\n{chr(10).join(missing_dirs)}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Create dataset config
    dataset_config = {
        'path': str(data_path),
        'train': str(train_path),
        'val': str(valid_path),
        'test': str(test_path),
        'names': {0: 'nike_logo'},
        'nc': 1  # number of classes
    }
    
    # Create directory for config if it doesn't exist
    config_dir = project_root / 'models' / 'config'
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Save dataset config
    config_path = config_dir / 'dataset.yaml'
    with open(config_path, 'w') as f:
        yaml.safe_dump(dataset_config, f, sort_keys=False)
    
    logger.info(f"Created dataset config at {config_path}")
    logger.info(f"Config contents:")
    with open(config_path, 'r') as f:
        logger.info(f"\n{f.read()}")
    
    # Initialize model
    model = YOLO('yolov8n.pt')
    
    # Create output directory if it doesn't exist
    output_path = project_root / 'models' / 'trained'
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