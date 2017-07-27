# GOAL: have the robot go to set a DESTINATION

# Before beginning, make sure that you have the libraries listed at the beginning
# of the code !!!

# INITIAL POSITION
# * the camera is only good for getting the initial position. By this, we need
# a pixel to feet conversion. i'll explain more below.

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

useless_frames = 10
camera = cv2.VideoCapture(0)
def get_image():
    retval, img = camera.read()
    return img
for i in range(useless_frames):
    temp = get_image()
    print("Hold up homie, I'm taking images rn...")
camera_capture = get_image()
cv2.imwrite('/Users/SirenaSarena/Desktop/AVproject/testimage.jpg', camera_capture)
            # ^ this is the path in which you want to save you photo.
            #your path will be different depending on what
            # folder you are working in
del(camera)
