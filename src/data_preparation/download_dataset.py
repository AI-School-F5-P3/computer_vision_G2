# data_preparation/download_dataset.py
import os
import requests
import zipfile
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_flickr_logos(base_path: str = "data/raw"):
    """
    Descarga el dataset FlickrLogos-47 si no existe localmente
    Args:
        base_path: Ruta base donde se guardará el dataset
    """
    # URL del dataset (esto es un ejemplo, deberías reemplazarlo con la URL real)
    dataset_url = "https://example.com/flickrlogos47.zip"
    
    # Crear directorios si no existen
    Path(base_path).mkdir(parents=True, exist_ok=True)
    
    zip_path = os.path.join(base_path, "flickrlogos47.zip")
    extract_path = os.path.join(base_path, "flickrlogos47")
    
    if os.path.exists(extract_path):
        logger.info("Dataset ya existe localmente")
        return extract_path
        
    try:
        logger.info("Descargando dataset...")
        response = requests.get(dataset_url, stream=True)
        response.raise_for_status()
        
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        logger.info("Extrayendo archivos...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            
        # Limpiar archivo zip
        os.remove(zip_path)
        logger.info("Dataset descargado y extraído exitosamente")
        
        return extract_path
        
    except Exception as e:
        logger.error(f"Error al descargar el dataset: {str(e)}")
        raise

if __name__ == "__main__":
    download_flickr_logos()