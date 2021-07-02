#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 13:37:59 2021

@author: bernard
"""

import cv2
import os
import numpy as np

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
    # Take the derivative in x direction
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)

    # Scale result to 0-255
    scaled_sobel = np.uint8(255 * abs(sobel_x) / np.max(abs(sobel_x)))
    # Create an array with the size of "scaled_sobel"
    sx_binary = np.zeros_like(scaled_sobel)

    # Keep only derivative values that are in the margin of interest (MARK THE LANE AS WHITE)
    sx_binary[(scaled_sobel >= 100) & 
            (scaled_sobel <= 255)] = 1 * 255  # Remember to multiply by 255, if you want to show the img

    return sx_binary


def hist_lane_pixel(binary):
    # Calculate the histogram to get the sum pixel value of each row (axis=0)
    hist = np.sum(binary, axis=0)
    # Instantiate the mid point
    mid_point = int(hist.shape[0] // 2)
    # Find the largest pixel value of each side
    left_x_idx = np.argmax(hist[:mid_point])
    right_x_idx = np.argmax(hist[mid_point:]) + mid_point  # Add mid_point to avoid bias for 0
    
    img_center = int(binary.shape[1] / 2)

    # Make a turn prediction based on the left and right index
    turn = turn_prediction(left_x_idx, right_x_idx, img_center)

    # Use the sliding window method to extract pixel values
    num_window = 10
    # the margin of the width of the window
    margin = 50
    window_height = int(binary.shape[0] // num_window)
    
    # x, y positions of none zero pixels
    # "binary" is an numpy array so we can use numpy.nonzero() to find nonzero elements
    nonzero_pts = binary.nonzero()
    # Keep in mind that the image's x coordinate and y coordinate are opposite
    nonzero_x = np.array(nonzero_pts[1])
    nonzero_y = np.array(nonzero_pts[0])

    left_x_cur = left_x_idx
    right_x_cur = right_x_idx

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
    # img = np.dstack((binary, binary, binary))*255
    # img[nonzero_y[left_lane], nonzero_x[left_lane]] = Blue
    # img[nonzero_y[right_lane], nonzero_x[right_lane]] = Red
    # cv2.imshow('img', img)
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


def turn_prediction(left_lane_pts, right_lane_pts, image_center):
    center_lane = left_lane_pts + (right_lane_pts - left_lane_pts) / 2

    if abs(center_lane - image_center) < 40:
        return "Straight"
    
    elif center_lane - image_center < 0:
        return "Turning Left"

    else:
        return "Turning Right"

def main():
    # load the image from the file directory
    file_dir = "./data_1/"
    data = [i for i in os.listdir(file_dir) if i.endswith('.png')]

    # source points to be warped (points are determined through try and error)
    pts_src = np.array([[250, 500], [540, 300], [720, 300], [850, 500]])
    # destination points to be warped towards
    pts_dst = np.array([[410, 511], [410, 0], [780, 0], [780, 511]])

    # Sort the image name properly
    data.sort(key=lambda x: int(x[8:-4]))
    # Get the homography matrix
    h, _ = cv2.findHomography(pts_src, pts_dst)
    
    for i in range(len(data)):
        filename = file_dir + data[i]
        # read each image 
        img = cv2.imread(filename)
        height, width, _ = img.shape
        # Undistort the image using the camera parameter
        undistort_img = undistort_image(img)
        # Creates a bird-eye view of the warp image
        warp = cv2.warpPerspective(undistort_img, h, (width, height))
        # Preprocess the image
        gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Detect the lane using sobel x detector
        sx_binary = apply_sobel_x(blur)
        # Use the histogram and sliding window technique to find the lane
        left_x, left_y, right_x, right_y, turn = hist_lane_pixel(sx_binary)

        # If hist_lane_pixel() cannot find any lane, perform gamma correction to find lane
        if np.sum(left_x) == 0 or np.sum(left_y) == 0 or np.sum(right_x) == 0 or np.sum(right_y) == 0:

            hsv_img = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
            clahe_img = clahe.apply(hsv_img[:, :, 2])

            gamma_img = adjust_gamma(clahe_img, 1.0)
            hsv_img[:, :, 2] = gamma_img

            processed_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
            
            gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            sx_binary = apply_sobel_x(blur)
            img, left_x, left_y, right_x, right_y, turn = hist_lane_pixel(sx_binary)
            # lane_detect_img = poly_fit(img, left_x, left_y, right_x, right_y)

        lane_detect_img = poly_fit(sx_binary, left_x, left_y, right_x, right_y)
        
        # Unwarp the image
        h_inv = np.linalg.inv(h)
        lane_detect_img = cv2.warpPerspective(lane_detect_img, h_inv, (width, height))
        # Combine the processed image to the original image
        final_img = cv2.addWeighted(np.uint8(img), 1, np.uint8(lane_detect_img), 0.5, 0)
        cv2.putText(final_img, turn, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, Red, 2, cv2.LINE_AA)
        
        # cv2.imshow('final', final_img)
        # print("Press q or esc to continue to the next frame !!!")
        directory_name = './data_1_output_png/' + data[i]
        cv2.imwrite(directory_name, final_img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    file_dir = "./data_1_output_png/"
    data = [i for i in os.listdir(file_dir) if i.endswith('.png')]
    fps = 10
    convert_frames_to_video(file_dir, 'Lane_Detection_data_1.avi', fps)
    

if __name__ == '__main__':
    main()