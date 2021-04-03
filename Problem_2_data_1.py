#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 13:37:59 2021

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
Blue = (255, 0, 0)
Red = (0, 0, 255)
Green = (0, 255, 0)


def convert_frames_to_video(pathIn, pathOut, fps):
    # Reference: https://www.life2coding.com/convert-image-frames-video-file-using-opencv-python/
    frame_array = []
    files = [f for f in os.listdir(pathIn) if f.endswith(".png")]
    # for sorting the file names properly
    files.sort(key=lambda x: int(x[5:-4]))
    for i in range(len(files)):
        filename = pathIn + files[i]
        # reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)

        # inserting the frames into an image array
        frame_array.append(img)
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
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
    if event == cv2.EVENT_RBUTTONDOWN:
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


def apply_sobel_x(gray):
    # Apply sobel in x direction
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    # abs_sobelx = np.absolute(sobel_x)

    # Scale result to 0-255
    scaled_sobel = np.uint8(255 * abs(sobel_x) / np.max(abs(sobel_x)))
    sx_binary = np.zeros_like(scaled_sobel)

    # Keep only derivative values that are in the margin of interest
    sx_binary[(scaled_sobel >= 100) & (
                scaled_sobel <= 255)] = 1 * 255  # Remember to multiply by 255, if you want to show the img
    
    # # Detect pixels that are white in the grayscale image
    # white_binary = np.zeros_like(gray)
    # white_binary[(gray > 200) & (gray <= 255)] = 1*255

    # binary = cv2.bitwise_or(sx_binary, white_binary)
    # cv2.imshow("s_x", sx_binary)

    return sx_binary


def hist_lane_pixel(binary):
    # Calculating the histogram to get max value
    hist = np.sum(binary, axis=0)

    mid_point = int(hist.shape[0] // 2)
    left_x_ind = np.argmax(hist[:mid_point])
    right_x_ind = np.argmax(hist[mid_point:]) + mid_point  # Add mid_point to avoid bias for 0

    img_center = int(binary.shape[1] / 2)
    turn = turn_prediction(left_x_ind, right_x_ind, img_center)

    # Use the sliding window method to extract pixels
    num_window = 10
    # the margin of the width of the window
    margin = 50
    window_height = int(binary.shape[0] // num_window)
    # x, y positions of none zero pixels
    nonzero_pts = binary.nonzero()
    nonzero_x = np.array(nonzero_pts[1])
    nonzero_y = np.array(nonzero_pts[0])

    left_x_cur = left_x_ind
    right_x_cur = right_x_ind

    left_lane = []
    right_lane = []

    # Set min_num_pixels to recenter the window
    min_num_pixels = 50

    for i in range(num_window):
        # Set the window boundary in x and y, left and right
        window_y_low = binary.shape[0] - (i + 1) * window_height
        window_y_high = binary.shape[0] - i * window_height
        window_left_low = left_x_cur - margin
        window_left_high = left_x_cur + margin
        window_right_low = right_x_cur - margin
        window_right_high = right_x_cur + margin

        # Identify none zero pixels in the window
        left_lane_pixels = ((nonzero_y >= window_y_low)
                            & (nonzero_y < window_y_high)
                            & (nonzero_x >= window_left_low)
                            & (nonzero_x < window_left_high)).nonzero()[0]  # &: binary operator

        right_lane_pixels = ((nonzero_y >= window_y_low)
                             & (nonzero_y < window_y_high)
                             & (nonzero_x >= window_right_low)
                             & (nonzero_x < window_right_high)).nonzero()[0]

        left_lane.append(left_lane_pixels)
        right_lane.append(right_lane_pixels)

        # Shift to the next mean position if the length of none zero pixels is greater than min_num_pixels
        if len(left_lane) > min_num_pixels:
            left_x_cur = int(np.mean(nonzero_x[left_lane_pixels]))
        if len(right_lane) > min_num_pixels:
            right_x_cur = int(np.mean(nonzero_x[right_lane_pixels]))

    # Concatenate the arrays of left and right lane
    left_lane = np.concatenate(left_lane)
    right_lane = np.concatenate(right_lane)

    # Extract the left and right lane pixel positions
    left_x = nonzero_x[left_lane]
    left_y = nonzero_y[left_lane]
    right_x = nonzero_x[right_lane]
    right_y = nonzero_y[right_lane]

    return left_x, left_y, right_x, right_y, turn


def poly_fit(sx_binary, left_x, left_y, right_x, right_y):
    # Use np.polyfit() to fit a second order polynomial
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)

    plot_y = np.linspace(0, sx_binary.shape[0] - 1, sx_binary.shape[0])

    left_fit_x = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
    right_fit_x = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]

    # Plot the result in sx_binary image
    img = np.dstack((sx_binary, sx_binary, sx_binary)) * 255
    # Extract points from line fitting
    left_pts = np.array([(np.vstack([left_fit_x, plot_y])).T])
    right_pts = np.array([np.flipud((np.vstack([right_fit_x, plot_y])).T)])

    pts = np.hstack((left_pts, right_pts))
    pts = np.array(pts, dtype='int32')

    cv2.fillPoly(img, pts, Green)
    cv2.polylines(img, np.int32([left_pts]), isClosed=False, color=Blue, thickness=10)
    cv2.polylines(img, np.int32([right_pts]), isClosed=False, color=Blue, thickness=10)

    return img


def turn_prediction(left_x, right_x, center_img):
    mean_distance_x = left_x + (right_x - left_x) / 2

    center_offset = center_img - mean_distance_x
    if (center_offset > 0):
        return ("Right")
    elif (center_offset < 0):
        return ("left")
    elif ((center_offset > 8)):
        return ("Straight")


def main():
    file_dir = "./data_1/"
    data = [i for i in os.listdir(file_dir) if i.endswith('.png')]
    # fps = 25
    # convert_frames_to_video(file_dir, 'Lane Detection.avi', fps)

    # source points to be warped (points are determined through try and error)
    # pts_src = np.array([[353,402], [490,323], [742,323],[805,402]])
    pts_src = np.array([[250, 500], [540, 300], [720, 300], [850, 500]])
    # destination points to be warped towards
    pts_dst = np.array([[410, 511], [410, 0], [780, 0], [780, 511]])

    # for sorting the file names properly
    data.sort(key=lambda x: int(x[5:-4]))

    for i in range(len(data)):
        filename = file_dir + data[i]
        # reading each files
        img = cv2.imread(filename)
        height, width, _ = img.shape
        # cv2.polylines(img, [pts_src], True, Red)

        blur = cv2.GaussianBlur(img, (7, 7), 0)
        
        # Gamma correction to adjust lighting condition
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
        clahe_img = clahe.apply(hsv[:, :, 2])

        gamma_img = adjust_gamma(clahe_img, 1.0)
        hsv[:, :, 2] = gamma_img

        processed_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        h, _ = cv2.findHomography(pts_src, pts_dst)
        # warp = cv2.warpPerspective(img, h, (width, height))
        undistort_img = undistort_image(processed_img)
        warp = cv2.warpPerspective(undistort_img, h, (width, height))
        # Edge Detection
        # warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # edge = cv2.Canny(blur, 200, 300)
        
        
        sx_binary = apply_sobel_x(blur)
        left_x, left_y, right_x, right_y, turn = hist_lane_pixel(sx_binary)
        lane_detect_img = poly_fit(sx_binary, left_x, left_y, right_x, right_y)

        # Unwarp the image
        h_inv = np.linalg.inv(h)
        lane_detect_img = cv2.warpPerspective(lane_detect_img, h_inv, (width, height))

        final_img = cv2.addWeighted(np.uint8(img), 1, np.uint8(lane_detect_img), 0.5, 0)

        cv2.putText(final_img, turn, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, Red, 2, cv2.LINE_AA)
        
        # cv2.imshow('original', img)
        cv2.imshow('s', sx_binary)
        cv2.imshow('lane_detect', lane_detect_img)
        
        # cv2.imshow('warp', warp)
        # cv2.imshow('undistort', undistort_img)
        # cv2.imshow('edge', edge)
        # cv2.setMouseCallback('f', click_event)
        # cv2.imshow('final', final_img)
        directory_name = './data_1_output_png/' + data[i]
        cv2.imwrite(directory_name, final_img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # file_dir = "./data_1_output_png/"
    # data = [i for i in os.listdir(file_dir) if i.endswith('.png')]
    # fps = 10
    # convert_frames_to_video(file_dir, 'Lane_Detection_data_1.avi', fps)
    
    # video = cv2.VideoWriter('Lane Detection.avi', 0, 1, ())
    # for i in data:
    #     img = cv2.imread(i)


if __name__ == '__main__':
    main()