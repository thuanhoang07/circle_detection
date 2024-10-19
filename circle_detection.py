import cv2
import numpy as np
import math


class ShapeDetector:
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

        # Lưu trữ trạng thái của hình elip/hình tròn
        self.last_detected_shapes = {'blue': None, 'green': None, 'yellow': None}
        self.flag = False  # Biến cờ flag

        self.cap = cv2.VideoCapture(0)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        frame_height, frame_width = frame.shape[:2]
        center_camera = (frame_width // 2, frame_height // 2)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Reset flag trước khi kiểm tra các hình
        self.flag = False  # Chỉ đặt lại cờ khi không phát hiện màu vàng

        # Detect blue và green, nhưng không ảnh hưởng tới flag
        frame_blue = self.detect_color(frame.copy(), hsv, self.lower_blue, self.upper_blue, center_camera,
                                       self.radius_real_blue, (255, 0, 0), "Blue Ellipse", (0, 0, 255), 30,
                                       frame_height - 20, 'blue', "Blue", update_flag=False)
        frame_green = self.detect_color(frame_blue, hsv, self.lower_green, self.upper_green, center_camera,
                                        self.radius_real_green, (0, 255, 0), "Green Ellipse", (0, 0, 255), 60,
                                        frame_height - 60, 'green', "Green", update_flag=False)
        # Phát hiện màu vàng và cập nhật flag
        frame_combined = self.detect_color(frame_green, hsv, self.lower_yellow, self.upper_yellow, center_camera,
                                           self.radius_real_yellow, (0, 255, 255), "Yellow Ellipse", (0, 0, 255), 90,
                                           frame_height - 100, 'yellow', "Yellow", update_flag=True)

        # Hiển thị flag lên màn hình
        flag_text = f"Flag: {self.flag}"
        cv2.putText(frame_combined, flag_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame_combined, flag_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

        print(flag_text)  # In trạng thái flag ra màn hình

        return frame_combined, ret

    def detect_color(self, frame, hsv, lower_color, upper_color, center_camera, radius_real, shape_color, text_label,
                     dot_color, y_offset, distance_y_offset, color_name, color_window_name, update_flag):
        mask = cv2.inRange(hsv, lower_color, upper_color)
        objects = cv2.bitwise_and(frame, frame, mask=mask)
        gray = cv2.cvtColor(objects, cv2.COLOR_BGR2GRAY)

        # Morphological operations: Closing to fill gaps in detected shapes
        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        frame_height, frame_width = frame.shape[:2]
        min_distance = float('inf')
        closest_ellipse = None

        for contour in contours:
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)

                if np.isnan(ellipse[0][0]) or np.isnan(ellipse[0][1]):
                    continue

                center = (int(ellipse[0][0]), int(ellipse[0][1]))
                distance = math.sqrt((center_camera[0] - center[0]) ** 2 + (center_camera[1] - center[1]) ** 2)

                min_distance = distance
                closest_ellipse = ellipse

        if closest_ellipse is not None:
            if closest_ellipse[1][0] > 0 and closest_ellipse[1][1] > 0:
                cv2.ellipse(frame, closest_ellipse, shape_color, 2)
                center = (int(closest_ellipse[0][0]), int(closest_ellipse[0][1]))
                cv2.line(frame, center_camera, center, (255, 255, 255), 2)

                self.last_detected_shapes[color_name] = closest_ellipse

                text = f'{text_label}'
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

                distance_text = f'{text_label} Distance to center: {int(min_distance)} px'
                cv2.putText(frame, distance_text, (10, distance_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)
                cv2.putText(frame, distance_text, (10, distance_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

                radius_real = max(closest_ellipse[1]) / 2
                if radius_real > 0:
                    distance_real = (radius_real * min_distance) / radius_real

                    if not (np.isnan(distance_real) or distance_real <= 0):
                        distance_real_text = f'{text_label} Real distance: {int(distance_real)} cm'
                        cv2.putText(frame, distance_real_text, (10, distance_y_offset - 20), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (255, 255, 255), 2)
                        cv2.putText(frame, distance_real_text, (10, distance_y_offset - 20), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 0), 1)

                        if distance_real <= self.threshold_distance:
                            cv2.putText(frame, f"{text_label} Match_real", (frame_width // 2 - 50, y_offset),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, shape_color, 3)

                # Đặt cờ flag thành True khi phát hiện hình elip nếu màu vàng
                if update_flag and color_name == 'yellow':
                    self.flag = True
        else:
            self.last_detected_shapes[color_name] = None

        return frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    detector = ShapeDetector()

    while True:
        frame, ret = detector.process_frame()
        if not ret:
            break

        cv2.imshow("Ellipse Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
            break

    detector.release()


if __name__ == "__main__":
    main()
