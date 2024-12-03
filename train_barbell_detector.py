from ultralytics import YOLO
import cv2
import numpy as np
import os

# Initialize YOLO model
model = YOLO('models/best_barbell_detector.pt')

# Fine-tune YOLO model
model.train(
    data="/users/eleves-a/2022/amine.chraibi/AI-Deadlift-Analysis/dataset/data.yaml",  
    epochs=250,           
    batch=128,             
    imgsz=640            
)

metrics = model.val(data="/users/eleves-a/2022/amine.chraibi/AI-Deadlift-Analysis/dataset/data.yaml") 

model.export(format="onnx")



