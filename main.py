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

# capture a photo
useless_frames = 10
camera = cv2.VideoCapture(0)
def get_image():
    retval, img = camera.read()
    return img
print("Hold up homie, I'm taking images rn... Say cheese :)")
for i in range(useless_frames):
    temp = get_image()
camera_capture = get_image()
cv2.imwrite('testimage.jpg', camera_capture)
del(camera)

# line detection
camera_capture_grey = cv2.imread('testimage.jpg', 0)
ret, thresh = cv2.threshold(camera_capture_grey, 127, 255, 0) # light needs to be
                                                              # pretty good. no shadow. fix?
camera_capture_grey2, contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# this is just to see the contour lines
camera_capture_copy = camera_capture.copy()
cv2.drawContours(camera_capture_copy,contours,-1,(0,255,0))
cv2.imshow('draw contours', camera_capture_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
