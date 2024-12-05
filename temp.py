from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import helper_functions as hf

model_barbell = YOLO('models/best_barbell_detector 2.pt')

hf.extract_barbell_positions('dataset/good/good006.mp4', model_barbell, 'barbell_positions.csv', debug=True)