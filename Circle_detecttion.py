import torch
import cv2

# Load mô hình YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # Thay bằng đường dẫn tới model của bạn

# Hàm để vẽ khung phát hiện lên khung hình camera
def draw_boxes(frame, detections, confidence_threshold=0.80):
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection[:6]
        if confidence >= confidence_threshold:  # Chỉ nhận dự đoán với xác suất >= 80%
            label = f'{model.names[int(class_id)]} {confidence:.2f}'
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return frame

# Mở camera
cap = cv2.VideoCapture(0)  # Sử dụng camera mặc định

if not cap.isOpened():
    print("Không thể mở camera")
    exit()

# Chạy vòng lặp để lấy khung hình từ camera
while True:
    ret, frame = cap.read()
    if not ret:
        print("Không nhận được khung hình từ camera")
        break

    # Dự đoán đối tượng trong khung hình
    results = model(frame)

    # Lấy kết quả dự đoán
    predictions = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, confidence, class]

    # Vẽ khung hình và nhãn lên khung hình camera (với độ tin cậy >= 80%)
    output_frame = draw_boxes(frame, predictions, confidence_threshold=0.80)

    # Hiển thị khung hình với các phát hiện
    cv2.imshow("YOLOv5 Detection", output_frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
