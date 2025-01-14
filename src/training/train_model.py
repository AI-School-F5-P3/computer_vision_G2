# src/training/train_model.py
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ..data_preparation.dataset import LogoDataset
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_model(num_classes):
    """
    Obtiene modelo pre-entrenado y modifica la capa final
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def train_model(data_path, num_epochs=10, batch_size=2):
    """
    Entrena el modelo usando el dataset proporcionado
    Args:
        data_path: Ruta al dataset (conteniendo carpetas train, test, valid)
        num_epochs: Número de épocas de entrenamiento
        batch_size: Tamaño del batch
    """
    # Preparar datasets
    train_dataset = LogoDataset(data_path, split='train')
    val_dataset = LogoDataset(data_path, split='valid')
    
    # Preparar dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    
    # Preparar modelo
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model(num_classes=2)  # Background + Logo
    model.to(device)
    
    # Optimizador
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    # Entrenamiento
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for images, targets in train_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            total_loss += losses.item()
            
        # Log progress
        logger.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}')
        
        # Guardar modelo
        if (epoch + 1) % 5 == 0:
            save_path = Path('models/trained')
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path / f'logo_detector_epoch_{epoch+1}.pth')
    
    return model