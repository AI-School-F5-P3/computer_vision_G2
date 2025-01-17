# Configuración global
BRAND_CLASSES = ['nike', 'adidas', 'puma']
NUM_CLASSES = len(BRAND_CLASSES) + 1  # +1 para la clase de fondo

# Configuración de detección
CONFIDENCE_THRESHOLD = 0.5
SAVE_DIR = 'detected_images'
DB_PATH = 'detections.db'

# Otras configuraciones que puedas necesitar
MODEL_PATH = 'runs/detect/train7/best.pt'