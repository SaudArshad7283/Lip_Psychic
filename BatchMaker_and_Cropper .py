#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 02:06:37 2019

@author: pr-lab
"""
#Importing Libs
import numpy as np
import glob
import os
import numpy
import cv2
import random
import pandas as pd
import tensorflow as tf
from imutils import face_utils
import dlib

lengths = [0,0]
# Dlib Tracker Initializations
p = "shape_predictor_68_face_landmarks.dat"             # Dlib Pretrained Facial Landmarks Tracker
detector = dlib.get_frontal_face_detector()             # Getting Face Detector Model
predictor = dlib.shape_predictor(p)                     # Predictor of 65 Facial Landmarks

# Function to make custom batches of the dataset and return a list of batches
"""      
def MakeBatches(path, settype, batch_size):
    paths, dirs, fil = next(os.walk(path+"/"))
    files_list = []
    for directory in dirs:
        path1 = os.path.join(paths,directory)
        path1 = (path1+'/'+settype+'/')
        files = glob.glob(os.path.join(path1, '*.mp4'))
        files = pd.DataFrame(files)
        files.insert(1, 'label', (path1.replace(path+"/", "")).replace("/"+settype+"/", ""))
        files = files.values
        files_list.extend(files)
        
    batch_list = []
    for i in range(int (len(files_list) / batch_size)):
        random.shuffle(files_list)
        batch = files_list[len(files_list)-batch_size:]
        files_list = files_list[:len(files_list)-batch_size]
        batch_list.append(batch)        
    
    return (np.array(batch_list))
"""


# Function that makes list of all videos in the Dataset to be processed 
# Takes in as Directory of Dataset and specified dataset-type e.g train, test and Val
def MakeBatches(path, settype):
    paths, dirs, fil = next(os.walk(path+"/"))      # Walk the directory to find all files
    files_list = []
    for directory in dirs:                          # For All directories (Classes)
        path1 = os.path.join(paths,directory)       # joining paths to specify the whole path from working directory
        path1 = (path1+'/'+settype+'/')
        files = glob.glob(os.path.join(path1, '*.mp4')) # Dataset Video File Type specification
        files = pd.DataFrame(files)
        files.insert(1, 'label', (path1.replace(path+"/", "")).replace("/"+settype+"/", ""))
        files = files.values
        files_list.extend(files)
    return files_list


# Create directories for Test, Train and val for preprocessed dataset.
def make_target_folders(batch_list, settype):
    for i in range(len(batch_list)):
        wordl = batch_list[i][0].split('/')[1]      #
        if wordl not in os.listdir('Class/'):
            os.mkdir('Class/'+wordl)
        if settype not in os.listdir('Class/'+wordl+'/'):
            os.mkdir('Class/'+wordl+'/'+settype)
    return 


        
# Code to calculate the optimum width of the crop rectangle
def Cal_boundrect_Size(batch_list):
    global lip_boundrect, lengths
    for sample in range(len(batch_list)):
        path = batch_list[sample][0]
        cap = cv2.VideoCapture(path)            # Video Reader Handle

        for i in range(24):
            ret, fr = cap.read()                # Read New Frame
            if ret == False:
                continue            
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)     # RGB to Gray Conversion
            rects = detector(gray, 0)                       # Detects Faces
            if (len(rects) != 1):                           # If no face is detected or more than one faces are detected then pass.
                continue
        
            shape = predictor(gray, rects[0])               # Predict Facial Landmarks of Detected face
            shape = face_utils.shape_to_np(shape)
            shape = shape[49:65]                            # Only Lips Landmarks
            x_min, y_max, x_max, y_min = np.min(shape[:,0]) , np.max(shape[:,1]) , np.max(shape[:,0]) , np.min(shape[:,1])          # Finding min and max points of X and Y directions. 
            x_len, y_len = x_max-x_min+2, y_max-y_min+2     # Max-min in each direction is length of Lip region in that frame
            if x_min > 0 and lengths[0] < x_len :
                lengths[0] =  x_len
            if y_min > 0 and lengths[1] < y_len:
                lengths[1] =  y_len
            
    return 0



# Following function croppes video spatially for 60x80 frames size with 24 frames a video. This basically croppes frames
# around lips region because this is only region that has useful information for Lip-reading. The Size 60x80 was 
# calculated by cal_boundrect_Size() for whole dataset.
    
def Cropper(batch_list):
    codec_func = cv2.VideoWriter_fourcc    # Initializing VideoWriter Codec function
    codec = codec_func(*'XVID')            # Codec defining
    for sample in range(len(batch_list)):
        path = batch_list[sample][0]
        print (path)
        cap = cv2.VideoCapture(path)       # Video Reader Handle
        out = cv2.VideoWriter((path.replace("lipread", "Class")).replace("mp4", "avi"),codec, 25.0, (80,60))    #Video Writer Handle with specified to folder names as Class
        rec, prev_frame, frame_counts, shape = 0,"",0, []
        while(True):
            ret, fr = cap.read()            # Read a frame from video reader Handle
            if ret == False:
                break
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)     #RGB to Gray Conversion of frame
            rects = detector(gray, 0)                       #Detecting Faces in Video using Dlib Tracker
            if (len(rects) == 0 ):
                continue
            shape = predictor(gray, rects[0])               # Detecting Facial Landmarks of First face (All these videos have usually one face) 
            shape = face_utils.shape_to_np(shape)           # Conversion to numpy format
            shape = shape[48:67]                            # Only identifying points which are in specified as Lips Landmarks
            np.delete(shape, np.argwhere(shape == [0,0]))   # Dropping the points which are not detected properly

# Writing Lip points to a file in text
#            with open((path.replace("lipread", "Class")).replace("mp4", "txt"), "w") as file:
#                for i in shape:                    
#                    prev_frame = str(i[0])+","+str(i[1])
#                    file.write(prev_frame)
            x_arr, y_arr = int(shape[:,0].mean()), int(shape[:,1].mean())   # Calculating centroid of Lip Landmarks in frame
            rec = gray[y_arr-30: y_arr+30 , x_arr-40: x_arr+40]             # Cropping from around centroid 60 along X direction and 80 along Y Direction
            out.write( cv2.cvtColor(rec, cv2.COLOR_GRAY2RGB) )              # Write this cropped frame using Video Writer Handle
            frame_counts = frame_counts + 1                                 # Written frames count
            if frame_counts >=24:                                           # Check to temporal length = 24
                break                                                       # If length Exceeds then Loop breaks and handle is released
        out.release()
    return 0

"""     A dedicated function that crops video temporally to    24-frames a video

def vid_length_Croper(batch_list):
    codec_func = cv2.VideoWriter_fourcc
    codec = codec_func(*'XVID')
    for sample in range(len(batch_list)):
        path = batch_list[sample][0]
        print (path)
        cap = cv2.VideoCapture(path)
        out = cv2.VideoWriter((path.replace("lipread_processed_datastep", "processed24frames")).replace("avi", "avi"),codec, 15.0, (80,60))
        vid = []        
        ret, frame = cap.read()
        while ret:
            vid.append(frame)          
            ret, frame = cap.read()
        if (len(vid) > 24):
            vid = vid[:24]
        elif (len(vid) > 16):
            x = vid[0]*0
            while (len(vid)!=24):
                vid.append(x)
        vid = (np.array(vid))
        if (vid.shape[0] == 24):
            for x in range(len(vid)):
                out.write( cv2.cvtColor(vid[x], cv2.COLOR_RGB2BGR )  )  
            out.release()
    return 0
"""    

batch_list = MakeBatches("lipread", "test")
make_target_folders(batch_list,'test')
batch_list = Cropper(batch_list)

batch_list = MakeBatches("lipread", "val")
make_target_folders(batch_list,'val')
batch_list = Cropper(batch_list)

batch_list = MakeBatches("lipread", "train")
make_target_folders(batch_list,'train')
batch_list = Cropper(batch_list)
