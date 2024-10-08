import cv2
import torch
import math
import pandas as pd

# Load mô hình YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')

# Lớp theo dõi đối tượng (Tracker)
class Tracker:
    def __init__(self):
        self.center_points = {}  # Lưu tọa độ trung tâm của từng đối tượng
        self.id_count = 0  # Đếm số lượng ID đã được cấp
        self.status = {}  # Trạng thái theo dõi từng đối tượng
        self.max_disappeared = 50  # Ngưỡng tối đa đối tượng mất dấu
        self.disappeared = {}  # Số lần đối tượng mất dấu

    def update(self, objects_rect):
        objects_bbs_ids = []

        # Danh sách các tọa độ trung tâm mới của các đối tượng
        new_center_points = {}

        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2  # Tính tọa độ trung tâm (cx, cy)
            cy = (y + y + h) // 2
            new_center_points[(cx, cy)] = (x, y, w, h)

        # Nếu chưa có đối tượng nào, gán ID cho đối tượng mới
        if len(self.center_points) == 0:
            for (cx, cy), (x, y, w, h) in new_center_points.items():
                self.center_points[self.id_count] = (cx, cy)
                self.status[self.id_count] = {'line1_up': False, 'line2_up': False, 'line1_down': False, 'line2_down': False, 'direction': None, 'counted': False}
                self.disappeared[self.id_count] = 0
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1
        else:
            # Tìm đối tượng tương ứng qua khoảng cách Euclidean
            used_ids = set()
            for (cx, cy), (x, y, w, h) in new_center_points.items():
                min_dist = float('inf')
                object_id = None

                # Tìm ID đối tượng có khoảng cách gần nhất
                for id, (prev_cx, prev_cy) in self.center_points.items():
                    if id in used_ids:
                        continue
                    dist = math.hypot(cx - prev_cx, cy - prev_cy)

                    if dist < min_dist and dist < 50:  # Điều kiện khoảng cách nhỏ hơn ngưỡng
                        min_dist = dist
                        object_id = id

                # Nếu tìm thấy đối tượng, cập nhật vị trí
                if object_id is not None:
                    self.center_points[object_id] = (cx, cy)
                    self.disappeared[object_id] = 0
                    objects_bbs_ids.append([x, y, w, h, object_id])
                    used_ids.add(object_id)
                else:
                    # Nếu không tìm thấy đối tượng, cấp ID mới
                    self.center_points[self.id_count] = (cx, cy)
                    self.status[self.id_count] = {'line1_up': False, 'line2_up': False, 'line1_down': False, 'line2_down': False, 'direction': None, 'counted': False}
                    self.disappeared[self.id_count] = 0
                    objects_bbs_ids.append([x, y, w, h, self.id_count])
                    self.id_count += 1

            # Xóa các đối tượng mất dấu
            disappeared_ids = set(self.center_points.keys()) - used_ids
            for object_id in disappeared_ids:
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    del self.center_points[object_id]
                    del self.status[object_id]
                    del self.disappeared[object_id]

        return objects_bbs_ids


# Khởi tạo video, tracker và các biến đếm
cap = cv2.VideoCapture("b.mp4")
tracker = Tracker()
count = 0
counter_up = 0
counter_down = 0

# Lưu trữ danh sách các ID đã đi lên và đi xuống
id_up = []
id_down = []

# Đường giới hạn
cy1 = 150  # Đường "line 1" (trên)
cy2 = 350  # Đường "line 2" (dưới)

# Thiết lập đối tượng ghi video với kích thước 640x480
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_resized.avi', fourcc, 30.0, (640, 480))

# Đọc danh sách lớp từ COCO
with open("coco.txt", "r") as f:
    class_list = f.read().split("\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue

    # Resize khung hình về kích thước 640x480
    frame = cv2.resize(frame, (640, 480))

    # Dự đoán đối tượng bằng YOLOv5
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()

    # Lưu danh sách bounding box của các đối tượng người
    list_boxes = []
    for det in detections:
        x1, y1, x2, y2, confidence, class_id = det
        if class_list[int(class_id)] == 'person':
            list_boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])

    # Cập nhật tracker với các đối tượng mới
    bbox_id = tracker.update(list_boxes)

    for bbox in bbox_id:
        x3, y3, w3, h3, obj_id = bbox
        cx = (x3 + x3 + w3) // 2
        cy = (y3 + y3 + h3) // 2
        cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)

        # Hiển thị ID trên màn hình
        cv2.putText(frame, f'ID: {obj_id}', (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Kiểm tra nếu người qua line 1 trước, sau đó qua line 2 (đi xuống)
        if cy < cy1 and not tracker.status[obj_id]['line1_down']:  # Qua line 1 trước (đi xuống)
            tracker.status[obj_id]['line1_down'] = True  # Đánh dấu đã qua line 1
        if tracker.status[obj_id]['line1_down'] and cy > cy2 and not tracker.status[obj_id]['line2_down'] and not tracker.status[obj_id]['counted']:
            counter_down += 1  # Đếm người đi xuống
            tracker.status[obj_id]['line2_down'] = True  # Đánh dấu đã qua line 2
            tracker.status[obj_id]['direction'] = 'down'  # Ghi lại hướng
            tracker.status[obj_id]['counted'] = True  # Đã được đếm
            id_down.append(obj_id)  # Thêm ID vào danh sách đi xuống

        # Kiểm tra nếu người qua line 2 trước, sau đó qua line 1 (đi lên)
        if cy > cy2 and not tracker.status[obj_id]['line2_up']:  # Qua line 2 trước (đi lên)
            tracker.status[obj_id]['line2_up'] = True  # Đánh dấu đã qua line 2
        if tracker.status[obj_id]['line2_up'] and cy < cy1 and not tracker.status[obj_id]['line1_up'] and not tracker.status[obj_id]['counted']:
            counter_up += 1  # Đếm người đi lên
            tracker.status[obj_id]['line1_up'] = True  # Đánh dấu đã qua line 1
            tracker.status[obj_id]['direction'] = 'up'  # Ghi lại hướng
            tracker.status[obj_id]['counted'] = True  # Đã được đếm
            id_up.append(obj_id)  # Thêm ID vào danh sách đi lên

    # Vẽ các đường giới hạn "lên" và "xuống"
    cv2.line(frame, (3, cy1), (637, cy1), (0, 0, 255), 2)  # Đường line 1 (trên)
    cv2.line(frame, (3, cy2), (637, cy2), (0, 255, 255), 2)  # Đường line 2 (dưới)

    # Hiển thị số lượng người đếm được
    cv2.putText(frame, f'Up: {counter_up}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Down: {counter_down}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Hiển thị danh sách các ID đã đi lên và đi xuống
    cv2.putText(frame, f'IDs Up: {id_up}', (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f'IDs Down: {id_down}', (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Ghi khung hình vào video đầu ra
    out.write(frame)

    # Hiển thị frame
    cv2.imshow('People Counter', frame)

    # Thoát nếu nhấn phím ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Giải phóng tài nguyên
cap.release()
out.release()  # Đóng file video đầu ra
cv2.destroyAllWindows()
