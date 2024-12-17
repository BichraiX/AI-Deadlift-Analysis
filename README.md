# AI Deadlift Analysis

## Introduction

This project aims to provide an AI-powered solution to analyze deadlift exercises. The tool uses computer vision and machine learning techniques to evaluate the form and technique of deadlifts, helping users improve their performance and avoid injuries.

## How to use ?

To use it, just run pipeline.py and give it the path to your video when prompted to do so.

## Code description

### demo.mp4

Video demonstration of what our model can do !

### classifiers_training

Contains the code for our main model's architecture, as well as the training code.

### dataset

Contains the dataset used to finetune YOLO on barbell detection

### models

Contains the finetuned YOLO models used for keypoints extraction. It doesn't contain our phase classifiers as they are too big to be pushed to Github.

### data_augmentation.py

The code used to perform data augmentation on our training set's videos.

### helper_functions

Contains functions to perform video separation, video labelling based on keypoints only, and all functions used in the pipeline.

### labelize.py

Code to labelize our dataset's videos.

### pipeline.py

Code for our pipeline, main file of our project.
