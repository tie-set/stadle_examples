import torch
import ultralytics

def get_model():
    model = torch.load('base_yolo_model.pt')
    return model