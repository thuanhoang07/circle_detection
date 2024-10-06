import cv2
import numpy as np
import math


class CircleDetector:
    def __init__(self):
        self.lower_blue = np.array([100, 150, 100])
        self.upper_blue = np.array([130, 255, 255])
        self.radius_real_blue = 35

        self.lower_green = np.array([40, 100, 100])
        self.upper_green = np.array([70, 255, 255])
        self.radius_real_green = 50

        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([30, 255, 255])
        self.radius_real_yellow = 25

        self.threshold_distance = 3

        self.cap = cv2.VideoCapture(0)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        frame_height, frame_width = frame.shape[:2]
        center_camera = (frame_width // 2, frame_height // 2)

        cv2.circle(frame, center_camera, 5, (0, 255, 0), -1)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Detect blue, green, and yellow circles
        frame = self.detect_color(frame, hsv, self.lower_blue, self.upper_blue, center_camera,
                                  self.radius_real_blue, (255, 0, 0), "Blue Circle", (0, 0, 255), 30, frame_height - 20)
        frame = self.detect_color(frame, hsv, self.lower_green, self.upper_green, center_camera,
                                  self.radius_real_green, (0, 255, 0), "Green Circle", (0, 0, 255), 60, frame_height - 60)
        frame = self.detect_color(frame, hsv, self.lower_yellow, self.upper_yellow, center_camera,
                                  self.radius_real_yellow, (0, 255, 255), "Yellow Circle", (0, 0, 255), 90, frame_height - 100)

        return frame, ret

    def detect_color(self, frame, hsv, lower_color, upper_color, center_camera, radius_real, circle_color, text_label,
                     dot_color, y_offset, distance_y_offset):
        mask = cv2.inRange(hsv, lower_color, upper_color)
        objects = cv2.bitwise_and(frame, frame, mask=mask)
        gray = cv2.cvtColor(objects, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            gray_blurred,
            cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=35, param2=30, minRadius=5, maxRadius=100
        )

        frame_height, frame_width = frame.shape[:2]

        if circles is not None:
            circles = np.uint16(np.around(circles))
            min_distance = float('inf')
            closest_circle = None

            for (x, y, radius) in circles[0, :]:
                distance = math.sqrt((center_camera[0] - x) ** 2 + (center_camera[1] - y) ** 2)

                if distance < min_distance:
                    min_distance = distance
                    closest_circle = (x, y, radius)

            if closest_circle is not None:
                x, y, radius = closest_circle

                cv2.circle(frame, (x, y), radius, circle_color, 2)
                cv2.circle(frame, (x, y), 1, dot_color, -1)
                cv2.line(frame, center_camera, (x, y), circle_color, 1)

                # Display circle information at different vertical positions
                text = f'{text_label}: Center=({x}, {y}), Radius={radius}'
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

                # Display distance information, including the color label
                distance_text = f'{text_label} Distance to center: {int(min_distance)} px'
                cv2.putText(frame, distance_text, (10, distance_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)
                cv2.putText(frame, distance_text, (10, distance_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

                if radius > 0:
                    distance_real = (radius_real * min_distance) / radius
                    distance_real_text = f'{text_label} Real distance: {int(distance_real)} cm'
                    cv2.putText(frame, distance_real_text, (10, distance_y_offset - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 2)
                    cv2.putText(frame, distance_real_text, (10, distance_y_offset - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 0), 1)

                    if distance_real <= self.threshold_distance:
                        cv2.putText(frame, f"{text_label} Match_real", (frame_width // 2 - 50, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    circle_color, 3)

        return frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    detector = CircleDetector()

    while True:
        frame, ret = detector.process_frame()
        if not ret:
            break

        cv2.imshow("Circle Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
            break

    detector.release()


if __name__ == "__main__":
    main()
