# healthSensor_IF_RGB

EE 434 Final Project - Health Diagnostic Sensor

This repository contains all of the scripts and custom libraries necessary to operate the sensor described in the report.
All library requirements are listed in the main directories *requirements.txt files

GITHUB REPOSITORY LINK - https://github.com/asvilesov/health_diagnostic_sensor

The repository is divided into two main Folders (a tree directory is listed at the bottom for reference):

Follow instructions for each:

1. health_sensor_jetson_env/ - This directory includes all files related to operating on the on-board Jetson nano. Some files are missing because they are too big to fit on the repository.
        Missing files can be downloaded from - https://drive.google.com/drive/folders/1LP4PDDWhFvKRrbhrgDX0DKWMaO_QGN85?usp=sharing - "skin_seg", "skin_seg_spherical2", 
        a. skin_seg - Model weights for the skin segmentation model
        b. skin_seg_spherical2 - Model weights for the bioFaces model (sorry for the weird name)
        
        Once downloaded:
        1. create a "models/" folder in the directory where you would like to run a scripts or training
        2. place the downloaded models into that folder
        
2. health_sensor_laptop_env_rgb_only/ - This directory includes all files related to training done off the jetson and running bioFaces on a laptop. Some files are missing because they are too big to fit on the repository.
        Missing files can be downloaded from (MUST BE ACCESSED WITH VALID USC EMAIL) - https://drive.google.com/drive/folders/1LP4PDDWhFvKRrbhrgDX0DKWMaO_QGN85?usp=sharing - "skin_seg", "skin_seg_spherical2", "celebA_img_norm_mask_light.h5", "celebA_shade.h5"
        a. skin_seg - Model weights for the skin segmentation model
        b. skin_seg_spherical2 - Model weights for the bioFaces model (sorry for the weird name)
        c. celebA_img_norm_mask_light.h5 - dataset containing celebA square images, mask images - warning this file is over 5 GB
        d. celebA_shade.h5 - dataset containing pseudo ground diffuse truth shading for images  - warning this file is over 5 GB

        once downloaded

        For Inference:
        1. create a "models/" folder in the directory where you would like to run a scripts or training
        2. place the downloaded models into that folder
        
        For Training:
        1. Download c. and d. from the google drive, then modify the input arguments in the training scripts when passing paths to the dataset object constructor
            a. For example: celebA_data = ds.image_dataset(filepath='/home/sasha/Desktop/bioFaces/data/celebA_img_norm_mask_light.h5', filepath_shade='/home/sasha/Desktop/bioFaces/data/celebA_shade.h5')

./
|
|    
|-----health_sensor_jetson_env/
|       |(scripts)
|       |- IR_temp.py - Measures inner eye temperatures, displays IR feed and projected facial bounding box, and fever classification result.
|       |- testDual_seg.py - Shows RGB and IR feed. RGB feed has facial feature bounding box, while IR feed displays max/min temperature value and location on screen.
|       |- testDual.py - Shows RGB and IR feed only - IR feed is colorized to show relative temperature.
|       |- testIR.py - Shows IR feed (colorized).
|       |- uvc-radiometry.py - Taken from UVC repository - shows radiometric IR feed (absolute temperature readings).
|       |- testRGB.py - Shows RGB feed.
|       |
|       |- IR_bio.py - DOES NOT WORK DUE TO COMPATIBILITY ISSUES - See line 300 of script for additional details. Measures inner eye temperatures, displays IR feed and
|       |         projected facial bounding box, fever classification result, and hemoglobin + melanin maps.
|       |
|-----health_sensor_laptop_env_rgb_only/
        |
        |(scripts)
        |face_detect_segment.py - runs 3 models on live video at once, the MTCNN, the skin segmentation model, and the bioFaces model
        |
        |-----training_only/
        |        |
        |        |-/biofaces_training/bioface_main.ipynb (run notebook with preloaded weights or train) The authors original code in Matlab can be found at (https://github.com/ssma502/BioFaces/blob/master/mainBioFaces.m)
        |        |-/biofaces_training/biofaces.py (main library with pyTorch code)
        |        |
        |        |-/skin_segmentation_training/test_skinseg.ipynb (run notebook with preloaded weights or train)
        |        |-/skin_segmentation_training/skin_seg.py (main library with pyTorch code)
        |
        |-----models/
        |        | This folder is empty, download model weights from to run the "face_detect_segment.py"
        



