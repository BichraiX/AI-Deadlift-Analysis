from ultralytics import YOLO
import cv2
import numpy as np
import os

# Initialize YOLO model
model = YOLO('best.pt')

# Fine-tune YOLO model
model.train(
    data="/users/eleves-a/2022/amine.chraibi/AI-Deadlift-Analysis/dataset/data.yaml",  
    epochs=1000,           
    batch=64,             
    imgsz=640            
)

metrics = model.val(data="/users/eleves-a/2022/amine.chraibi/AI-Deadlift-Analysis/dataset/data.yaml") 
print(metrics) 




