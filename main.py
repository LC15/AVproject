import cv2
import numpy as np
from warp import Warp

# getting image from the camera
# camera = cv2.VideoCapture(0)
# def get_image():
#     retval, img = camera.read()
#     return img
# original_image = get_image()
original_image = cv2.imread('slantview1.png')

# create object and filter image to get contours
rectangle = Warp(np.array([10, 255, 255]), np.array([0, 128, 64]))
rectangle_filtered = rectangle.filter_bw(original_image)

# obtain, sort, and approximate contours to find desired contours
rectangle_bounds = rectangle.findsort_contours(rectangle_filtered, -2)
epsilon = 0.05 * cv2.arcLength(rectangle_bounds, True)
rectangle_approx_contours = cv2.approxPolyDP(rectangle_bounds, epsilon, True)

rectangle_ordered_contours, rectangle_desired_contours, max_width, max_height = rectangle.orderdesire_contours(rectangle_approx_contours)

# transform the image
transformation = cv2.getPerspectiveTransform(rectangle_ordered_contours, rectangle_desired_contours)
warped_image = cv2.warpPerspective(original_image, transformation, (max_width, max_height))

# find the center of the car
blue_light = Warp(np.array([130, 255, 255]), np.array([110, 50, 50]))
blue_light_filtered = blue_light.filter_bw(warped_image)
im2, blue_light_contours, hierarchy = cv2.findContours(blue_light_filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
blue_light_contours = blue_light_contours[0]
blue_center_x, blue_center_y = blue_light.find_center(blue_light_contours)

green_light = Warp(np.array([70, 255, 255]), np.array([50, 100, 100]))
green_light_filtered = green_light.filter_bw(warped_image)
im2, green_light_contours, hierarchy = cv2.findContours(green_light_filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
green_light_contours = green_light_contours[0]
green_center_x, green_center_y = green_light.find_center(green_light_contours)

center_of_car_x = float((blue_center_x + green_center_x) / 2)
center_of_car_y = float((blue_center_y + green_center_y) / 2)

rectangle.conversion(center_of_car_x, center_of_car_y, max_width, max_height, 9, 6)

# just to view start/end results
cv2.drawContours(original_image, rectangle_approx_contours, -1, (255, 0, 0), 3)
resized_original_image = cv2.resize(original_image, (900, 600))
cv2.imshow('original image', resized_original_image)

cv2.circle(warped_image, (int(center_of_car_x), int(center_of_car_y)), 4, (255, 255, 255), -1)
resized_warped_image = cv2.resize(warped_image, (900, 600))
cv2.imshow('warped image', resized_warped_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
