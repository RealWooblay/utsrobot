import numpy as np 
import cv2 as cv
from gpiozero import Servo
from time import sleep

# Setting up the GPIO pin for the servo
from gpiozero.pins.pigpio import PiGPIOFactory

# Use the PiGPIOFactory for more precise control
factory = PiGPIOFactory()

# Setup the servo on GPIO 17
servo = Servo(17, pin_factory=factory)  

# Initialize Video Capture
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

#Steering geometry

wheelbase = 0.02 # meters, example value
track_width = 0.01 # meters, example value

# Add steering geometry to control servo 
def calculate_ackermann_angles(servo_pos, wheelbase, track_width):
    """ Calculate the Ackermann steering angles based on servo position. """
    master_angle = np.interp(servo_pos, [-1, 1], [-30, 30])  # servo position ranges from -1 to 1
    if master_angle == 0:
        return (0, 0)
    radius = wheelbase / np.tan(np.radians(master_angle))
    left_angle = np.degrees(np.arctan(wheelbase / (radius - track_width / 2)))
    right_angle = np.degrees(np.arctan(wheelbase / (radius + track_width / 2)))
    return (left_angle, right_angle)

def set_servo_angle(angle):
    """ Set servo position based on angle (-30 to 30 degrees). """
    position = np.interp(angle, [-30, 30], [-1, 1])
    servo.value = position
    sleep(0.1)  # add delay to allow servo to reach position

# Initialize trackbar names
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
high_H_name = 'High H'
low_S_name = 'Low S'
high_S_name = 'High S'
low_V_name = 'Low V'
high_V_name = 'High V'

# Trackbar callback functions
def on_low_H_thresh_trackbar(val):
    global low_H
    low_H = val
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)

def on_high_H_thresh_trackbar(val):
    global high_H
    high_H = val
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)

def on_low_S_thresh_trackbar(val):
    global low_S
    low_S = val
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)

def on_high_S_thresh_trackbar(val):
    global high_S
    high_S = val
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)

def on_low_V_thresh_trackbar(val):
    global low_V
    low_V = val
    cv.setTrackbarPos(low_V_name, window_detection_name, low_V)

def on_high_V_thresh_trackbar(val):
    global high_V
    high_V = val
    cv.setTrackbarPos(high_V_name, window_detection_name, high_V)

# Create trackbars for color selection
cv.namedWindow(window_detection_name)
cv.createTrackbar(low_H_name, window_detection_name, 0, 179, on_low_H_thresh_trackbar)
cv.createTrackbar(high_H_name, window_detection_name, 179, 179, on_high_H_thresh_trackbar)
cv.createTrackbar(low_S_name, window_detection_name, 0, 255, on_low_S_thresh_trackbar)
cv.createTrackbar(high_S_name, window_detection_name, 255, 255, on_high_S_thresh_trackbar)
cv.createTrackbar(low_V_name, window_detection_name, 0, 255, on_low_V_thresh_trackbar)
cv.createTrackbar(high_V_name, window_detection_name, 255, 255, on_high_V_thresh_trackbar)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert frame to HSV color space
    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Threshold the HSV image to get only yellow and blue colors
    # (You can modify these thresholds according to your specific yellow and blue colors)
    low_yellow = np.array([20, 100, 100])
    high_yellow = np.array([30, 255, 255])
    mask_yellow = cv.inRange(frame_HSV, low_yellow, high_yellow)

    cv.imshow("window yellowimg", mask_yellow)

    low_blue = np.array([110, 50, 50])
    high_blue = np.array([130, 255, 255])
    mask_blue = cv.inRange(frame_HSV, low_blue, high_blue)

    cv.imshow("window blueimg", mask_blue)
    # Bitwise-AND mask and original image
    res_yellow = cv.bitwise_and(frame, frame, mask=mask_yellow)
    res_blue = cv.bitwise_and(frame, frame, mask=mask_blue)

    # Combine the yellow and blue images
    result = cv.add(res_yellow, res_blue)

    # Display the resulting frame
    cv.imshow(window_detection_name, result)

    if cv.waitKey(1) == ord('q'):
        break


    

# Release the capture
cap.release()
cv.destroyAllWindows()
