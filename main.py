import cv2
from color_detection import CircleDetector

def main():
    detector = CircleDetector()

    while True:
        frame, ret = detector.process_frame()

        if frame is None:
            break

        cv2.imshow("Circle Tracking", frame)

        # Điều khiển bằng bàn phím
        key = cv2.waitKey(1) & 0xFF
        if key == ord('x'):
            detector.switch_to_green()

        if key == ord('z'):
            detector.switch_to_blue()

        if key == ord('c'):
            detector.switch_to_yellow()

        if key == 27:
            break

    detector.release()

if __name__ == "__main__":
    main()
