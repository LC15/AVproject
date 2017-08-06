import cv2
import numpy as np

class Warp():
    def __init__(self, upper_bound, lower_bound, kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))):
        self.upper = upper_bound
        self.lower = lower_bound
        self.kernel = kernel

    #filter and return a clean black and white image
    def filter_bw(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        filter_color = cv2.inRange(hsv_image, self.lower, self.upper)
        opening_image = cv2.morphologyEx(filter_color, cv2.MORPH_OPEN, self.kernel)
        closing_image = cv2.morphologyEx(opening_image, cv2.MORPH_CLOSE, self.kernel)
        return closing_image

    # find, sort and return desired contour (either smallest or largest)
    def findsort_contours(self, filtered_image, position):
        im2, contours, hierarchy = cv2.findContours(filtered_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        area_bounds = 0
        if len(contours) != 0:
            if len(contours) == 1:
                position = 0
            sorted_contours = sorted(contours, key = lambda contour: cv2.contourArea(contour))
            area_bounds = sorted_contours[position]
        return area_bounds

    # order the (4) contour points: top left, bottom left, bottom right, top right, find the
    # desired contours, max width and max height
    def orderdesire_contours(self, approx_contours):
        (first, second, third, fourth) = approx_contours
        first = list(first[0])
        second = list(second[0])
        third = list(third[0])
        fourth = list(fourth[0])

        ordered_contours = np.array(approx_contours, dtype = "float32")
        s = np.array([first[0]+first[1], second[0]+second[1], third[0]+third[1], fourth[0]+fourth[1]])
        ordered_contours[0] = approx_contours[np.argmin(s)]
        ordered_contours[2] = approx_contours[np.argmax(s)]

        approx_contours = np.delete(approx_contours, np.argmin(s), axis = 0)

        if np.argmax(s) < np.argmin(s):
            position = np.argmax(s)
        else:
            position = np.argmax(s) - 1
        approx_contours = np.delete(approx_contours, position, axis = 0)

        (one, two) = approx_contours
        one = list(one[0])
        two = list(two[0])

        d = np.array([one[1]-one[0], two[1]-two[0]])
        ordered_contours[1] = approx_contours[np.argmax(d)]
        ordered_contours[3] = approx_contours[np.argmin(d)]

        width_a = np.sqrt(((fourth[0] - first[0]) ** 2) + ((fourth[1] - first[1]) ** 2))
        width_b = np.sqrt(((third[0] - second[0]) ** 2) + ((third[1] - second[1]) ** 2))
        max_width = max(int(width_a), int(width_b))

        height_a = np.sqrt(((second[0] - first[0]) ** 2) + ((second[1] - first[1]) ** 2))
        height_b = np.sqrt(((fourth[0] - third[0]) ** 2) + ((fourth[1] - third[1]) ** 2))
        max_height = max(int(height_a), int(height_b))

        desired_contours = np.array([[0, 0], [0, max_height - 1], [max_width - 1, max_height - 1], [max_width - 1, 0]], np.float32)
        return ordered_contours, desired_contours, max_width, max_height

    # find the center of the lights
    def find_center(self, contour):
        moment = cv2.moments(contour)
        c_x = int(moment["m10"] / moment["m00"])
        c_y = int(moment["m01"] / moment["m00"])
        return c_x, c_y

    def conversion(self, x, y, max_width, max_height, area_width, area_height):
        x_feet = (x * area_width) / max_width
        y_feet = (y * area_height) / max_height
        print("x_feet: ", x_feet)
        print("y_feet: ", y_feet)
