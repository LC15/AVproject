# ADDITIONAL INFO for the remainder of the code (this section is not done):
# keeping track of encoder to keep track of distance
# function: wait_for_edge(user_gpio, edge, wait_timeout)
#
# ** the camera needs to take pictures every 50 miliseconds (bc encoders check 20
# times every second)

import cv2
import numpy as np
import matplotlib.pyplot as plt
#import pigpio

#if __name__ == '__main__':
    #pi = pigpio.pi()

def get_image():
    retval, img = camera.read()
    return img

def filter_bw(image, lower, upper, kernel):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    filter_color = cv2.inRange(hsv_image, lower, upper)
    opening_image = cv2.morphologyEx(filter_color, cv2.MORPH_OPEN, kernel)
    closing_image = cv2.morphologyEx(opening_image, cv2.MORPH_CLOSE, kernel)
    return closing_image

def sort_contours(contours, position):
    if len(contours) != 0:
        if len(contours) == 1:
            position = 0
        sorted_contours = sorted(contours, key = lambda contour: cv2.contourArea(contour))
        area_bounds = sorted_contours[position]
    return area_bounds

def order_contours(approx_contours):
    ordered_contours = np.array(approx_contours, dtype = "float32")
    s = np.array([first[0]+first[1], second[0]+second[1], third[0]+third[1], fourth[0]+fourth[1]])
    ordered_contours[0] = approx_contours[np.argmin(s)]
    ordered_contours[2] = approx_contours[np.argmax(s)]

    approx_contours = np.delete(approx_contours, np.argmin(s), axis = 0)
    if np.argmax(s) == 0:
        position = 0
    else:
        position = np.argmax(s) - 1
    approx_contours = np.delete(approx_contours, position, axis = 0)
    (one, two) = approx_contours
    one = list(one[0])
    two = list(two[0])

    d = np.array([one[1]-one[0], two[1]-two[0]])
    ordered_contours[1] = approx_contours[np.argmax(d)]
    ordered_contours[3] = approx_contours[np.argmin(d)]
    return ordered_contours

def find_center(contour):
    moment = cv2.moments(contour)
    c_x = int(moment["m10"] / moment["m00"])
    c_y = int(moment["m01"] / moment["m00"])
    return c_x, c_y

def conversion(x, y, image_width, image_height, area_width, area_height):
    x_feet = (x * area_width) / image_width
    y_feet = (y * area_height) / image_height
    print("x_feet: ", x_feet)
    print("y_feet: ", y_feet)

upper_orange = np.array([10, 255, 255])
lower_orange = np.array([0, 128, 64])
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# to take an image from the webcam
# camera = cv2.VideoCapture(0)
# print("Hold up homie, I'm taking images rn... Say cheese :)")
# for i in range(30):
#     temp = get_image()
# original_image = get_image()
original_image = cv2.imread('slantview.png')

image2 = filter_bw(original_image, lower_orange, upper_orange, kernel)
im2, contours, hierarchy = cv2.findContours(image2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# get the four corners of the rectangle
rect_area_bounds = sort_contours(contours, -2)
epsilon = 0.05 * cv2.arcLength(rect_area_bounds, True)
approx_contours = cv2.approxPolyDP(rect_area_bounds, epsilon, True)

# to find desired transformed coordinates
approx_contours = np.array(approx_contours, np.float32)
(first, second, third, fourth) = approx_contours
first = list(first[0])
second = list(second[0])
third = list(third[0])
fourth = list(fourth[0])

ordered_contours = order_contours(approx_contours)

width_a = np.sqrt(((fourth[0] - first[0]) ** 2) + ((fourth[1] - first[1]) ** 2))
width_b = np.sqrt(((third[0] - second[0]) ** 2) + ((third[1] - second[1]) ** 2))
max_width = max(int(width_a), int(width_b))

height_a = np.sqrt(((second[0] - first[0]) ** 2) + ((second[1] - first[1]) ** 2))
height_b = np.sqrt(((fourth[0] - third[0]) ** 2) + ((fourth[1] - third[1]) ** 2))
max_height = max(int(height_a), int(height_b))

desired_contours = np.array([[0, 0], [0, max_height - 1], [max_width - 1, max_height - 1], [max_width - 1, 0]], np.float32)

# transform the image
transformation = cv2.getPerspectiveTransform(ordered_contours, desired_contours)
warped_image = cv2.warpPerspective(original_image, transformation, (max_width, max_height))

# find the center of the car in terms of feet
orange_filter_warped = filter_bw(warped_image, lower_orange, upper_orange, kernel)
orange_filer2, contours, hierarchy = cv2.findContours(orange_filter_warped, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
orange_light_contours = sort_contours(contours, -1) # might have to change this to detect small light
orange_center_x, orange_center_y = find_center(orange_light_contours)

upper_green = np.array([70, 255, 255])
lower_green = np.array([50, 100, 100])
filter_green_bw = filter_bw(warped_image, lower_green, upper_green, kernel)
green_im, green_contours, hierarchy = cv2.findContours(filter_green_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
green_light_contours = sort_contours(green_contours, 0)
green_center_x, green_center_y = find_center(green_light_contours)

center_of_car_x = int((orange_center_x + green_center_x) / 2)
center_of_car_y = int((orange_center_y + green_center_y) / 2)

conversion(center_of_car_x, center_of_car_y, 900, 600, 9, 6)

# just to view start/end results
# resized_original_image = cv2.resize(original_image, (900, 600))
# cv2.imshow('original', resized_original_image)
resized_warped_image = cv2.resize(warped_image, (900, 600))
# cv2.circle(resized_warped_image, (center_of_car_x, center_of_car_y), 7, (255, 255, 255), -1)
cv2.imshow('warped image', resized_warped_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
