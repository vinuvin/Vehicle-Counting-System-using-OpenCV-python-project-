import cv2
import numpy as np

# Web camera
cap = cv2.VideoCapture('video.mp4')

min_rectangle_width = 80
min_rectangle_height = 80
count_line_position = 550
offset = 6
vehicle_counter = 0

# Initialize Background Subtractor
algo = cv2.createBackgroundSubtractorMOG2()


def center_handle(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


detect = []

while True:
    ret, frame1 = cap.read()
    if not ret:
        break

    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)

    # applying on each frame
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)

    counterShape, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)

    for (i, c) in enumerate(counterShape):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_counter = (w >= min_rectangle_width) and (h >= min_rectangle_height)
        if not validate_counter:
            continue

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame1, "Vehicle " + str(vehicle_counter), (x, y - 20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 244, 0),
                    2)

        counter = center_handle(x, y, w, h)
        detect.append(counter)
        cv2.circle(frame1, counter, 4, (0, 0, 255), -1)

    for (x, y) in detect:
        if (count_line_position - offset) <= y <= (count_line_position + offset):
            vehicle_counter += 1
            detect.remove((x, y))
            print("Vehicle Counter:", vehicle_counter)
            cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (0, 127, 255), 3)
            cv2.putText(frame1, "VEHICLE COUNTER: " + str(vehicle_counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 0, 255), 5)

    cv2.imshow('Video Original', frame1)

    if cv2.waitKey(1) == 13:  # Press Enter key to exit
        break

cv2.destroyAllWindows()
cap.release()
