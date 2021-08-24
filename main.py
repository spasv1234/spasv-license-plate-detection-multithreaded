import jetson.utils
#Modules for multithreading
import threading

#Modules for working with database
#import database

#Modules for working with license plate regex and email
import lp_filter,lp_alert

#Modules for working with time
import time
from datetime import datetime
import schedule

#Modules for working with paths
import os.path

#Modules for working with AI and images
from PIL import Image
import cv2
import numpy as np
import easyocr

#Config file
import config

#Loads object detection model
import jetson.inference
net = jetson.inference.detectNet(argv=['--model=../python/training/detection/ssd/models/license_plate_512_2/ssd-mobilenet.onnx','--labels=../python/training/detection/ssd/models/license_plate_512_2/labels.txt',
                                       '--input-blob=input_0','--output-cvg=scores','--output-bbox=boxes','--threshold=0.5'])
#Loads OCR model
reader = easyocr.Reader(['en'],gpu=True) 

#Video IO settings
camera = jetson.utils.videoSource(config.video_source,["--input-width=1280","--input-height=720","--input-flip=rotate-180","--input-loop=-1"]) 

jetson.utils.cudaDeviceSynchronize()

global bgr_img,frames_with_detection,lock
bgr_img=None
frames_with_detection=[]
lock = threading.Lock()

#Function to crop image
def crop(img,x,y,w,h):
        crop_roi = (x, y, x+w, y+h)
        crop_img = jetson.utils.cudaAllocMapped(width=w,height=h,format=img.format)
        jetson.utils.cudaDeviceSynchronize()
        try:
                jetson.utils.cudaCrop(img, crop_img, crop_roi)
                return crop_img
        except:
                print("error cropping image")
                return img

#Function to convert cudaImage(rgb format) to bgr format for opencv
def convert_to_cv_img(rgb_img):
        global bgr_img
        if bgr_img is None:
                bgr_img = jetson.utils.cudaAllocMapped(width=rgb_img.width,height=rgb_img.height,format='bgr8')
        jetson.utils.cudaConvertColor(rgb_img, bgr_img)
        jetson.utils.cudaDeviceSynchronize()
        cv_img = jetson.utils.cudaToNumpy(bgr_img)
        return cv_img
                        
#Function to make detections, return cv2 frame and detections info
def detect_license_plate():
        global frames_with_detection
        while True:
                cuda_frame = camera.Capture()
                detections = net.Detect(cuda_frame,overlay='none') 
                cv2_frame = convert_to_cv_img(cuda_frame)
                if len(detections) > 0:
                        for detection in detections:
                                x,y,w,h = round(detection.Left),round(detection.Top),round(detection.Width),round(detection.Height)                                             #Gets information for cropping and drawing boxes
                                frames_with_detection.append([cv2_frame.copy(),[x,y,w,h]])
                                cv2.rectangle(cv2_frame,(x,y),(x+w,y+h),color=(0,255,0),thickness=2)
                cv2.imshow("Video",cv2_frame)
                cv2.waitKey(1)

#Function to read license plates
def read_license_plate():
        global frames_with_detection
        while True:
                if len(frames_with_detection)>0:
                        cv2_frame = frames_with_detection[0][0]
                        x,y,w,h = frames_with_detection[0][1]
                        cv2_frame = cv2_frame[y:y+h,x:x+w]
                        results = reader.readtext(cv2_frame, detail=0)
                        license_plate_number=""
                        for result in results:
                                result = result.upper()
                                license_plate_number += result
                                license_plate_number = lp_filter.remove_noise(license_plate_number)
                                print("License Plate Detected: " + str(license_plate_number))
                        frames_with_detection.pop(0)
                        
                                
                                

if __name__ == "__main__":
        main_thread = threading.Thread(target=detect_license_plate)
        ocr_thread = threading.Thread(target=read_license_plate)
        main_thread.start()
        ocr_thread.start()
