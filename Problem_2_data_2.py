#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 01:05:47 2021

@author: bernard
"""

import cv2
import numpy as np

Blue = (255, 0, 0)
Red = (0, 0, 255)
Green = (0, 255, 0)


def undistort_image(img):
    # Camera parameter
    K = np.array([[1.15422732e+03, 0.00000000e+00, 6.71627794e+02],
                  [0.00000000e+00, 1.14818221e+03, 3.86046312e+02],
                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dist_coeff = np.array([[-2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02]])
    img = cv2.undistort(img, K, dist_coeff, None, K)
    return img


def extract_lane(img):
    hsl_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    # Search for the yellow lane at left
    # lower_mask_yellow = np.array([20, 120, 80], dtype='uint8')
    # upper_mask_yellow = np.array([45, 200, 255], dtype='uint8')
    lower_mask_yellow = np.array([20, 90, 90], dtype='uint8')
    upper_mask_yellow = np.array([45, 200, 255], dtype='uint8')
    mask_yellow = cv2.inRange(hsl_img, lower_mask_yellow, upper_mask_yellow)

    yellow_lane = cv2.bitwise_and(hsl_img, hsl_img, mask=mask_yellow).astype(np.uint8)

    # Search for the white lane at right
    # lower_mask_white = np.array([0, 200, 0], dtype='uint8')
    # upper_mask_white = np.array([255, 255, 255], dtype='uint8')
    lower_mask_white = np.array([0, 190, 0], dtype='uint8')
    upper_mask_white = np.array([255, 255, 255], dtype='uint8')
    mask_white = cv2.inRange(hsl_img, lower_mask_white, upper_mask_white)

    white_lane = cv2.bitwise_and(hsl_img, hsl_img, mask=mask_white).astype(np.uint8)

    # Combine both
    yellow_white_lane = cv2.bitwise_or(yellow_lane, white_lane)

    img = cv2.cvtColor(yellow_white_lane, cv2.COLOR_HLS2BGR)

    return img


def hist_lane_pixel(warp):
    # Calculating the histogram to get max value
    hist = np.sum(warp, axis=0)
    img = np.dstack((warp, warp, warp))*255
    # cv2.imshow('im', img)
    mid_point = int(hist.shape[0] // 2)
    left_x_ind = np.argmax(hist[:mid_point])
    right_x_ind = np.argmax(hist[mid_point:]) + mid_point  # Add mid_point to avoid bias for 0

    img_center = int(warp.shape[1] / 2)

    turn = turn_prediction(left_x_ind, right_x_ind, img_center)

    # Use the sliding window method to extract pixels
    num_window = 10
    # the margin of the width of the window
    margin = 50
    window_height = int(warp.shape[0] // num_window)
    # x, y positions of none zero pixels
    nonzero_pts = warp.nonzero()
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
        window_y_low = warp.shape[0] - (i + 1) * window_height
        window_y_high = warp.shape[0] - i * window_height
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

    img[nonzero_y[left_lane], nonzero_x[left_lane]] = Blue
    img[nonzero_y[right_lane], nonzero_x[right_lane]] = Red
    # cv2.imshow('img', img)
    return img, left_x, left_y, right_x, right_y, turn


def poly_fit(img, left_x, left_y, right_x, right_y):
    # Use np.polyfit() to fit a second order polynomial
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)

    plot_y = np.linspace(0, img.shape[0] - 1, img.shape[0])

    left_fit_x = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
    right_fit_x = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]

    # Extract points from line fitting
    left_pts = np.array([(np.vstack([left_fit_x, plot_y])).T])
    # flip the array upside down in order to cv2.fillPoly()
    right_pts = np.array([np.flipud((np.vstack([right_fit_x, plot_y])).T)])

    pts = np.hstack((left_pts, right_pts))
    pts = np.array(pts, dtype='int32')

    # turn_prediction(left_pts, right_pts)

    img = np.zeros_like(img).astype(np.uint8)
    cv2.fillPoly(img, pts, Green)
    cv2.polylines(img, np.int32([left_pts]), isClosed=False, color=Blue, thickness=10)
    cv2.polylines(img, np.int32([right_pts]), isClosed=False, color=Blue, thickness=10)

    return img


def turn_prediction(left_lane_pts, right_lane_pts, image_center):
    center_lane = left_lane_pts + (right_lane_pts - left_lane_pts) / 2

    if abs(center_lane - image_center) < 10:
        return "Straight"

    elif center_lane - image_center < 0:
        return "Turning Left"

    else:
        return "Turning Right"


def adjust_gamma(image, gamma=1.0):
    # Reference: https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def main():
    file_dir = "./data_2/challenge_video.mp4"
    cap = cv2.VideoCapture(file_dir)
    # source points to be warped (points are determined through try and error)
    pts_src = np.array([[330, 700], [600, 500], [720, 500], [1040, 700]])
    # destination points to be warped towards
    pts_dst = np.array([[400, 680], [400, 0], [880, 0], [880, 680]])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('Lane_Detection_data_2.avi',fourcc, 20.0, (1280, 720))
    if not cap.isOpened():
        print("Error")
    while cap.isOpened():

        ret, frame = cap.read()
        
        
        if ret:
            height, width, _ = frame.shape
            undistort_img = undistort_image(frame)

            extract_lane_img = extract_lane(undistort_img)

            h, _ = cv2.findHomography(pts_src, pts_dst)

            gray = cv2.cvtColor(extract_lane_img, cv2.COLOR_BGR2GRAY)
            # Filter noise
            img_blur = cv2.bilateralFilter(gray, 9, 120, 100)

            # Apply edge detection
            img_edge = cv2.Canny(img_blur, 100, 200)

            warp = cv2.warpPerspective(img_edge, h, (width, height))

            img, left_x, left_y, right_x, right_y, turn = hist_lane_pixel(warp)
            # If hist_lane_pixel() cannot find any lane, perform gamma correction to find lane
            if np.sum(left_x) == 0 or np.sum(left_y) == 0 or np.sum(right_x) == 0 or np.sum(right_y) == 0:

                hsv_img = cv2.cvtColor(undistort_img, cv2.COLOR_BGR2HSV)
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
                clahe_img = clahe.apply(hsv_img[:, :, 2])

                gamma_img = adjust_gamma(clahe_img, 1.0)
                hsv_img[:, :, 2] = gamma_img

                processed_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
                warp = cv2.warpPerspective(processed_img, h, (width, height))
                gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                img, left_x, left_y, right_x, right_y, turn = hist_lane_pixel(blur)
                lane_detect_img = poly_fit(img, left_x, left_y, right_x, right_y)
            else:
                lane_detect_img = poly_fit(img, left_x, left_y, right_x, right_y)

            # Unwarp the image
            h_inv = np.linalg.inv(h)
            lane_detect_img = cv2.warpPerspective(lane_detect_img, h_inv, (width, height))

            final_img = cv2.addWeighted(np.uint8(frame), 1, np.uint8(lane_detect_img), 0.5, 0)

            cv2.putText(final_img, turn, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, Red, 2, cv2.LINE_AA)

            cv2.imshow('final', final_img)
            out.write(final_img)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
