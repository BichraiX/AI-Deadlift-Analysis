import os
import cv2
import numpy as np
import random
from tqdm import tqdm

def horizontal_flip(frame):
    return cv2.flip(frame, 1)

def adjust_brightness(frame, factor):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,2] = hsv[:,:,2]*factor
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    frame_bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return frame_bright

def adjust_contrast(frame, factor):
    f = 131*(factor + 127)/(127*(131 - factor))
    alpha_c = f
    gamma_c = 127*(1 - f)
    frame_contrast = cv2.addWeighted(frame, alpha_c, frame, 0, gamma_c)
    return frame_contrast

def add_gaussian_noise(frame, mean=0, sigma=15):
    gauss = np.random.normal(mean, sigma, frame.shape).astype('uint8')
    noisy = cv2.add(frame, gauss)
    return noisy

def process_video(video_path, augmentations, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    augmented_videos = []

    for aug in augmentations:
        augmented_frames = frames.copy()
        suffix = ""
        if 'flip' in aug:
            augmented_frames = [horizontal_flip(frame) for frame in augmented_frames]
            suffix += "_flip"
        if 'brightness' in aug:
            factor = random.uniform(0.7, 1.3)  
            augmented_frames = [adjust_brightness(frame, factor) for frame in augmented_frames]
            suffix += f"_bright{int(factor*100)}"
        if 'contrast' in aug:
            factor = random.uniform(0.7, 1.3) 
            augmented_frames = [adjust_contrast(frame, factor) for frame in augmented_frames]
            suffix += f"_contrast{int(factor*100)}"
        if 'noise' in aug:
            augmented_frames = [add_gaussian_noise(frame) for frame in augmented_frames]
            suffix += "_noise"

        base, ext = os.path.splitext(output_path)
        new_output_path = f"{base}{suffix}{ext}"
        augmented_videos.append((aug, new_output_path, augmented_frames))
    
    for aug, out_path, aug_frames in augmented_videos:
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        for frame in aug_frames:
            out.write(frame)
        out.release()
        print(f"Saved augmented video: {out_path}")

def augment_dataset(base_dir, num_augmentations=2):
    """
    Perform data augmentation on all mp4 videos in the base_dir and its subdirectories.
    
    Parameters:
        base_dir (str): Path to the base directory containing subfolders with videos.
        num_augmentations (int): Number of augmented versions to create per video.
    """
    possible_augmentations = [
        ['flip'],
        ['brightness'],
        ['contrast'],
        ['noise'],
        ['flip', 'brightness'],
        ['flip', 'contrast'],
        ['flip', 'noise'],
        ['flip'],
        ['brightness', 'contrast'],
        ['brightness', 'noise'],
        ['brightness'],
        ['contrast', 'noise'],
        ['contrast'],
        ['noise'],
        ['flip', 'brightness', 'contrast'],
        ['flip', 'brightness', 'noise'],
        ['flip', 'brightness'],
        ['flip', 'contrast', 'noise'],
        ['flip', 'contrast'],
        ['flip', 'noise'],
        ['brightness', 'contrast', 'noise'],
        ['brightness', 'contrast'],
        ['brightness', 'noise'],
        ['contrast', 'noise'],
        ['flip', 'brightness', 'contrast', 'noise'],
        ['flip', 'brightness', 'contrast'],
        ['flip', 'brightness', 'noise'],
        ['flip', 'contrast', 'noise'],
        ['brightness', 'contrast', 'noise'],
        ['flip', 'brightness', 'contrast', 'noise']
    ]

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith('.mp4'):
                video_path = os.path.join(root, file)
                print(f"Processing video: {video_path}")
                
                for i in range(num_augmentations):
                    aug = random.choice(possible_augmentations)
                    process_video(
                        video_path,
                        augmentations=[aug],
                        output_path=video_path
                    )

base_directory = "processed_videos/train"  
augmentations_per_video = 3  

augment_dataset(base_directory, num_augmentations=augmentations_per_video)
