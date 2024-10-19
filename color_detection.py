import cv2
import numpy as np

# Mở webcam
cap = cv2.VideoCapture(0)

# Kiểm tra nếu camera không mở được
if not cap.isOpened():
    print("Không thể mở webcam")
    exit()

# Hàm chính để xử lý từng khung hình từ webcam
while True:
    # Đọc từng khung hình
    ret, frame = cap.read()

    # Nếu không đọc được khung hình thì thoát
    if not ret:
        print("Không thể nhận khung hình")
        break

    # Chuyển đổi khung hình sang không gian màu HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Định nghĩa khoảng màu cho green, blue, và yellow
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])

    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Tạo mặt nạ cho từng màu
    mask_green = cv2.inRange(hsv_frame, lower_green, upper_green)
    mask_blue = cv2.inRange(hsv_frame, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

    # Áp dụng mặt nạ lên khung hình ban đầu
    result_green = cv2.bitwise_and(frame, frame, mask=mask_green)
    result_blue = cv2.bitwise_and(frame, frame, mask=mask_blue)
    result_yellow = cv2.bitwise_and(frame, frame, mask=mask_yellow)

    # Hiển thị kết quả
    cv2.imshow('Webcam', frame)
    cv2.imshow('Green Color', result_green)
    cv2.imshow('Blue Color', result_blue)
    cv2.imshow('Yellow Color', result_yellow)

    # Thoát chương trình khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
