#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 18:47:11 2021

@author: bernard
"""

import cv2
import os
import numpy as np
from os.path import isfile, join
# #Camera Matrix
# K: 9.037596e+02 0.000000e+00 6.957519e+02 0.000000e+00 9.019653e+02 2.242509e+02 0.000000e+00 0.000000e+00 1.000000e+00
# #distortion coefficients
# D: -3.639558e-01 1.788651e-01 6.029694e-04 -3.922424e-04 -5.382460e-02
Red = (255,0,0)
Blue = (0,0,255)
Green = (0,255,0)
def convert_frames_to_video(pathIn, pathOut, fps):
    # Reference: https://www.life2coding.com/convert-image-frames-video-file-using-opencv-python/
    frame_array = []
    files = [f for f in os.listdir(pathIn) if f.endswith(".png")]
    #for sorting the file names properly
    files.sort(key = lambda x: int(x[5:-4]))
    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)

        #inserting the frames into an image array
        frame_array.append(img)
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
  
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
  
        # displaying the coordinates
        # on the image window
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(x) + ',' +
        #             str(y), (x,y), font,
        #             1, (255, 0, 0), 2)
        # cv2.imshow('f', img)
  
    # checking for right mouse clicks     
    if event==cv2.EVENT_RBUTTONDOWN:
  
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
  
        # displaying the coordinates
        # on the image window
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # b = img[y, x, 0]
        # g = img[y, x, 1]
        # r = img[y, x, 2]
        # cv2.putText(img, str(b) + ',' +
        #             str(g) + ',' + str(r),
        #             (x,y), font, 1,
        #             (255, 255, 0), 2)
        # cv2.imshow('f', img)

def undistort_image(img):
    # Camera parameter
    K = np.array([[9.037596e+02, 0.000000e+00, 6.957519e+02], 
                  [0.000000e+00, 9.019653e+02, 2.242509e+02], 
                  [0.000000e+00, 0.000000e+00, 1.000000e+00]])
    dist_coeff = np.array([[-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02]])
    img = cv2.undistort(img, K, dist_coeff, None, K)
    return img

def adjust_gamma(image, gamma=1.0):
    # Reference: https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)


def hist_lane_pixel(gray):
    # Apply sobel in x direction
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    # abs_sobelx = np.absolute(sobel_x)

    # Scale result to 0-255
    scaled_sobel = np.uint8(255*abs(sobel_x)/np.max(abs(sobel_x)))
    sx_binary = np.zeros_like(scaled_sobel)
    
    # Keep only derivative values that are in the margin of interest
    sx_binary[(scaled_sobel >= 30) & (scaled_sobel <= 255)] = 1*255
    # Detect pixels that are white in the grayscale image
    white_binary = np.zeros_like(gray)
    white_binary[(gray > 200) & (gray <= 255)] = 1*255
    
    binary = cv2.bitwise_or(sx_binary, white_binary)
    
    return binary

def main():
    file_dir = "./data_1/"
    data = [i for i in os.listdir(file_dir) if i.endswith('.png')]
    # fps = 25
    # convert_frames_to_video(file_dir, 'Lane Detection.avi', fps)
    
    #source points to be warped
    # pts_src = np.array([[353,402], [490,323], [742,323],[805,402]])
    pts_src = np.array([[175, 500], [540, 260], [720, 260],[880, 500]])
    #destination points to be warped towards
    pts_dst = np.array([[410,511],[410, 0],[780, 0],[780,511]])
    
    #for sorting the file names properly
    data.sort(key = lambda x: int(x[5:-4]))
    
    # for i in range(len(data)):
    for i in range(5): 
        filename = file_dir + data[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        # cv2.polylines(img, [pts_src], True, Red)
        
        h, _ = cv2.findHomography(pts_src, pts_dst)
        # warp = cv2.warpPerspective(img, h, (width, height))
        undistort_img = undistort_image(img)
        warp = cv2.warpPerspective(undistort_img, h, (width, height))
        # Edge Detection
        # warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # edge = cv2.Canny(blur, 200, 300)
        
        binary = hist_lane_pixel(blur)
        
        
        
        cv2.imshow('original', img)
        cv2.imshow('s', binary)
        cv2.imshow('warp', warp)
        # cv2.imshow('undistort', undistort_img)
        # cv2.imshow('edge', edge)
        # cv2.setMouseCallback('f', click_event)
        
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    # video = cv2.VideoWriter('Lane Detection.avi', 0, 1, ())
    # for i in data:
    #     img = cv2.imread(i)
if __name__ == '__main__':
    main()
 