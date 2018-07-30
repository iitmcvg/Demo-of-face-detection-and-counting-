
# coding: utf-8

# In[2]:


import sys
#sys.path.append('./Tiny_Faces_in_Tensorflow/')
#import tiny_face_eval as tiny
import evaluate
from metrics import *
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import mean_squared_error as mse
import glob
import os
import cv2
import pandas as pd
from sklearn.svm import SVC
import numpy as np
import imp
import time
import random
import detect
import dlib
from imgaug import augmenters as iaa
#imp.reload(tiny)
imp.reload(detect)
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


weights_path = './hr_res101.pkl'


# ## Saving frames

# In[3]:


cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
initial_target = 0
final_target = int(49 * fps) + 10
i = 0
frames = []
while(True):
    ret, frame = cap.read()
    if not ret :
        continue
    i +=1 
    #print(frame)
    if i in range(initial_target, final_target+10):
        frames.append(frame[:,:,::-1])
    if i == final_target:
        break
    if i == 100:
        break
    
print(np.size(frames))
frames = np.asarray(frames)
cv2.VideoWriter('countinggif.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

# In[4]:


images = []
for k in range(0, len(frames), 10):
    try:
        imgs = [frames[k], frames[k+1], frames[k+2], frames[k+10]]
    except IndexError:
        imgs = [frames[k], frames[k+1], frames[k+2], frames[len(frames)-1]]
    images.append(imgs)


# ## Detection

# In[5]:
print(len(images))

all_detections = []
for frames in images:
    detections = []
    for frame in frames:
        with tf.Graph().as_default():
            b = evaluate.evaluate(weight_file_path=weights_path,  img=frame)
        detections.append(b)
    all_detections.append(detections)


# ## Matching 

# In[37]:


threshold = 0.55
matcheds = []
t0 = time.time()
for j in range(len(images)):
    frames = images[j]
    detections = all_detections[j]
    matched = 0
    t0bis = time.time()
    for p in range(len(detections[0])):
        neigh_detect, distances = detect.train_binclas(frames, detections, p)
        idx_max, val_max = np.argmax(distances[:,1]), np.max(distances[:,1])
        if val_max > threshold:
            matched += 1
    matcheds.append(matched)
    t1 = time.time()
    print('It took %.1f sec i.e %.2f/detection' % (t1-t0bis, (t1-t0bis)/len(detections[0])))
print('Total : %.1f' % (time.time() - t0))


# ## Counting

# In[62]:

print(len(all_detections))
s = 0
for j in range(10):
    detections = all_detections[j]
    s += len(detections[0]) - matcheds[j]
s += len(detections[3])


# In[63]:


s


# ## Gif Producting with counting

# In[78]:


cap = cv2.VideoCapture('./countinggif.avi')
fps = cap.get(cv2.CAP_PROP_FPS)
initial_target = int(45 * fps) + 10
final_target = int(49 * fps) + 10
i = 0
frames = []
while(True):
    ret, frame = cap.read()
    i +=1 
    if i in range(initial_target, final_target, 1):
        frames.append(frame[:,:,::-1])
    if i == final_target:
        break


# In[79]:


detections = []
for i, frame in enumerate(frames):
    with tf.Graph().as_default():
        b = tiny.evaluate(weight_file_path=weights_path, data_dir='.jpg', output_dir='', framee=frame,
                          prob_thresh=0.5, nms_thresh=0.1, lw=3, 
                          display=False, save=False, draw=False, print_=0)
    detections.append(b[0])
    time.sleep(0.5)


# In[72]:


## Computing incremental count
nbs = []
init = len(all_detections[0][0])
for j in range(1, 10):
    nbs.append(init)
    detections_ = all_detections[j]
    init += len(detections_[0]) - matcheds[j-1]
init += len(detections_[3]) - matcheds[j]
nbs.append(init)


# In[117]:


k = 0
l = 0
images = []
ff = []
font = cv2.FONT_HERSHEY_SIMPLEX
for j, frame in enumerate(frames):
    img = frame.copy()
    for detect_ in detections[j]:
        pt1, pt2 = tuple(detect_[:2]), tuple(detect_[2:])
        cv2.rectangle(img, pt1, pt2, (255, 0, 0), 2)
    cv2.putText(img, 'Incremental count : %d' % nbs[l], (1750,1300), font, 1.5, (0, 255, 0), 3)
    if j in range(10, 89, 9):
        l += 1
    images.append(img)   
    cv2.imwrite('./output_video/frames_%05d.png' % j, img[:,:,::-1])


# ## Gif Production without counting

# In[33]:
cap.release()

'''cap = cv2.VideoCapture('/home/alexattia/Work/RecVis/famvk.avi')
fps = cap.get(cv2.CAP_PROP_FPS)
initial_target = int(45 * fps) + 10
final_target = int(49 * fps) + 10
i = 0
frames = []
while(True):
    ret, frame = cap.read()
    i +=1 
    if i in range(initial_target, final_target, 1):
        frames.append(frame[:,:,::-1])
    if i == final_target:
        break


# In[32]:


detections = []
for i, frame in enumerate(frames):
    with tf.Graph().as_default():
        b = tiny.evaluate(weight_file_path=weights_path, data_dir='.jpg', output_dir='', framee=frame,
                          prob_thresh=0.5, nms_thresh=0.1, lw=3, 
                          display=False, save=False, draw=False, print_=0)
    detections.append(b[0])
    time.sleep(0.5)


# In[41]:


k = 0
images = []
for j, frame in enumerate(frames):
    img = frame.copy()
    for detect_ in detections[k]:
        pt1, pt2 = tuple(detect_[:2]), tuple(detect_[2:])
        cv2.rectangle(img, pt1, pt2, (255, 0, 0), 2)
    images.append(img)
    if j in range(0, 94, 2):
        k += 1
    cv2.imwrite('./output_video/frame_%05d.png' % j, img[:,:,::-1])'''

