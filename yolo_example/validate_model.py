import os
import argparse
import random

import torch

import time

from ultralytics import YOLO

model_container = YOLO("yolo11n.yaml")
model_container.model = torch.load('base_yolo_model.pt').to('cuda')
model_container.model.load_state_dict(torch.load("val_model.pt"))

model_container.predict('all_veg_test.png', imgsz=640, iou=0.01)[0].save(filename='all_veg_detections.png')