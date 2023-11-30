import cv2
import numpy as np
from pyfirmata import Arduino, util

board = Arduino('COM4') # Change 'COM4' to your Arduino's serial port

red_pin = board.get_pin('d:13:o')     # Digital output pin 13
green_pin = board.get_pin('d:12:o')   # Digital output pin 12
blue_pin = board.get_pin('d:11:o')    # Digital output pin 11

def get_dominant_color(frame, threshold=500):
    # Convert the BGR image to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges in HSV
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([140, 255, 255])

    # Threshold the image to get only specified colors
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Get the area of each color
    red_area = cv2.countNonZero(mask_red)
    green_area = cv2.countNonZero(mask_green)
    blue_area = cv2.countNonZero(mask_blue)

    # Determine the dominant color
    dominant_color = "Unknown"
    max_area = max(red_area, green_area, blue_area)
    
    if max_area > threshold:
        if max_area == red_area:
            dominant_color = "Red"
        elif max_area == green_area:
            dominant_color = "Green"
        elif max_area == blue_area:
            dominant_color = "Blue"

    return dominant_color

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Error reading frame")
        break

    # Get the dominant color in the frame
    dominant_color = get_dominant_color(frame)

    # Draw the dominant color on the frame
    cv2.putText(frame, f"Dominant Color: {dominant_color}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    if dominant_color == "Red":
        red_pin.write(1)

    elif dominant_color == "Green":
        green_pin.write(1)

    elif dominant_color == "Blue":
        blue_pin.write(1)
    else:
        red_pin.write(0)
        green_pin.write(0)
        blue_pin.write(0)


    # Display the annotated frame
    cv2.imshow('Dominant Color Detection', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
