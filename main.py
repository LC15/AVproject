# GOAL: have the robot go to set a DESTINATION

# Before beginning, make sure that you have the libraries listed at the beginning
# of the code !!!

# INITIAL POSITION
# * the camera is only good for getting the initial position.

# STEP 1: perspective change (camera isn't going to be right above)
# openCV functions:
# 1. cv2.VideoCapture() - your webcam should be cv2.VideoCapture(0) (knowing what
#                         your webcam is is just for testing)
# 2. filter by color/hsv - cv2.cvtColor() only see things that are a certain color.
#                                         * car has red and green light in the front
#                                         and back
# 3. cv2.findContours() - get coordinates and dump into cv2.getPerspectiveTransform()
# 4. cv2.getPerspectiveTransform() - get info from this function and dump it
#                                   into cv2.warpPerspective()
# 5. cv2.warpPerspective()
# 6. use the final square photo to determine where the robot is in terms
# feet. figure out where the robot is in terms of pixels, find out how many pixels
# equal how many feet, then use that to convert the pixels back to feet to send
# the initial position to the computer.

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

upper = np.array([10, 255, 255])
lower = np.array([0, 128, 64])
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# to take an image from the webcam
useless_frames = 30
camera = cv2.VideoCapture(0)

def get_image():
    retval, img = camera.read()
    return img
print("Hold up homie, I'm taking images rn... Say cheese :)")
for i in range(useless_frames):
    temp = get_image()
image = get_image()
# image = cv2.imread('rec4.png')

# filtering the video/photo
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
filter_color = cv2.inRange(hsv_image, lower, upper)
opening_image = cv2.morphologyEx(filter_color, cv2.MORPH_OPEN, kernel)
closing_image = cv2.morphologyEx(opening_image, cv2.MORPH_CLOSE, kernel)
filtered_hsv = cv2.bitwise_and(image, image, mask = closing_image)

# contouring
im2, contours, hierarchy = cv2.findContours(closing_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

if len(contours) != 0:
    # finding the largest contour
    sorted_contours = sorted(contours, key = lambda contour: cv2.contourArea(contour))
    area_bounds = sorted_contours[-1]

# to get the four corners
epsilon = 0.05 * cv2.arcLength(area_bounds, True)
approx_contours = cv2.approxPolyDP(area_bounds, epsilon, True)

# to find desired coordinates
approx_contours = np.array(approx_contours, np.float32)
(tr, tl, bl, br) = approx_contours
tr = list(tr[0])
tl = list(tl[0])
bl = list(bl[0])
br = list(br[0])

width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
max_width = max(int(width_a), int(width_b))

height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
max_height = max(int(height_a), int(height_b))

desired_contours = np.array([[max_width - 1, 0], [0, 0], [0, max_height - 1], [max_width - 1, max_height - 1]], np.float32)
transformation = cv2.getPerspectiveTransform(approx_contours, desired_contours)
final_image = cv2.warpPerspective(image, transformation, (max_width, max_height))

# just to view what happens throughout the code
cv2.imshow('original', image)
cv2.imshow('final image', final_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
