# src/train_script.py
from pathlib import Path
import sys

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Now we can import from src
from src.training.train_model import train_model

def main():
    # Get paths relative to script location
    data_path = project_root / 'data'
    
    print(f"Using data path: {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found at {data_path}")
    
    try:
        # Train the model with desired parameters
        model_path = train_model(
            data_path=str(data_path),
            num_epochs=50,
            batch_size=8
        )
        print(f"Training completed successfully! Model saved at: {model_path}")
        
    except Exception as e:
        print(f"Training failed: {str(e)}")

if __name__ == "__main__":
    main()