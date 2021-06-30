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
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('Night_Drive_improved.avi', fourcc, 20.0, (1920, 1080))
    if not cap.isOpened():
        print("Error")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            height, width, _ = frame.shape
            blur = cv2.GaussianBlur(frame, (7, 7), 0)
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
            clahe_img = clahe.apply(hsv[:, :, 2])
            
            gamma_img = adjust_gamma(clahe_img, 1.0)
            hsv[:, :, 2] = gamma_img

            processed_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # hist,bins = np.histogram(hsv[:, :, 0].flatten(),256,[0,256])
            # cdf = hist.cumsum()
            # cdf_normalized = cdf * float(hist.max()) / cdf.max()
            
            # cdf_m = np.ma.masked_equal(cdf,0)
            # cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
            # cdf = np.ma.filled(cdf_m,0).astype('uint8')
            # processed_img = cdf[hsv]
            
            # # equalize the histogram of the Y channel
            # hsv[:,:,0] = cv2.equalizeHist(hsv[:,:,0])

            # # convert the YUV image back to RGB format
            # processed_img = cv2.cvtColor(hsv, cv2.COLOR_YUV2BGR)

            # showing the image
            cv2.imshow('improved_image', processed_img)
            # out.write(processed_img)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    # out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
