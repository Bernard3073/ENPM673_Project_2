#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 15:26:08 2021

@author: bernard
"""
import cv2
import numpy as np

def adjust_gamma(image, gamma=1.0):
    # Reference: https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def main():
    cap = cv2.VideoCapture('./Night Drive - 2689.mp4')
    # a = np.zeros((256,),dtype=np.float16)
    
    if (cap.isOpened()==False):
        print("Error")
    while cap.isOpened(): 
        ret, frame = cap.read()
        
        if ret == True:
            # Resize the video frame
            scale_percent = 60 # percent of original size
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            original = cv2.resize(frame, (width, height), interpolation = cv2.INTER_AREA)
            
            blur = cv2.GaussianBlur(original, (7,7), 0)
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
                
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))
            clahe_img = clahe.apply(hsv[:, :, 2])
            
            gamma_img = adjust_gamma(clahe_img, 1.0)
            hsv[:, :, 2] = gamma_img
            
            processed_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            #showing the image
            cv2.imshow('improved_image', processed_img)
            if cv2.waitKey(30) & 0xFF == ord("q"): 
                break 
        else:
            break
        
    cap.release()
    cv2.destroyAllWindows() 

if __name__ == '__main__':
    main()