# 執行: 在powershell中輸入 python main.py
import cv2
from ultralytics import YOLO

# ========== 1. 載入模型 ==========

# 偵測椅子
det_model = YOLO("yolov8n.pt")

# 偵測人體姿勢
pose_model = YOLO("yolov8n-pose.pt")


# ========== 2. 開啟 Iriun Webcam ==========

CAMERA_ID = 0  # 若不是 Iriun，嘗試 1、2、3

cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("無法開啟攝影機")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# ========== 3. 即時辨識 ==========

while True:
    ret, frame = cap.read()

    if not ret:
        print("無法讀取畫面")
        break

    # YOLO 偵測椅子（COCO class 56）
    det_result = det_model(
        frame,
        classes=[56],
        conf=0.35,
        verbose=False
    )[0]

    # YOLO Pose 偵測人體
    pose_result = pose_model(
        frame,
        conf=0.4,
        verbose=False
    )[0]

    # 取得椅子座標
    chair_boxes = []

    if det_result.boxes is not None:
        chair_boxes = (
            det_result.boxes.xyxy.cpu().numpy()
        )

    # 每張椅子一個占用狀態
    occupied = [False] * len(chair_boxes)

    # ========== 4. 人體姿勢分析 ==========

    if (
        pose_result.keypoints is not None
        and pose_result.keypoints.xy is not None
    ):
        keypoints = (
            pose_result.keypoints.xy.cpu().numpy()
        )

        keypoint_conf = pose_result.keypoints.conf

        if keypoint_conf is not None:
            keypoint_conf = keypoint_conf.cpu().numpy()

        for person_idx, person_kpts in enumerate(keypoints):

            # 關鍵點可信度檢查
            if keypoint_conf is not None:
                required = [11, 12, 13, 14]

                if any(
                    keypoint_conf[person_idx][i] < 0.4
                    for i in required
                ):
                    continue

            # 左右髖關節
            left_hip = person_kpts[11]
            right_hip = person_kpts[12]

            # 左右膝蓋
            left_knee = person_kpts[13]
            right_knee = person_kpts[14]

            # 使用中間位置
            hip_x = (left_hip[0] + right_hip[0]) / 2
            hip_y = (left_hip[1] + right_hip[1]) / 2

            knee_y = (left_knee[1] + right_knee[1]) / 2

            # 確認膝蓋低於髖關節
            is_knee_lower = knee_y > hip_y

            if not is_knee_lower:
                continue

            # 檢查髖關節是否落在椅子區域
            for i, chair in enumerate(chair_boxes):

                cx1, cy1, cx2, cy2 = chair

                in_x_range = cx1 <= hip_x <= cx2
                in_y_range = cy1 <= hip_y <= cy2

                if in_x_range and in_y_range:
                    occupied[i] = True

                    # 繪製髖關節中心
                    cv2.circle(
                        frame,
                        (int(hip_x), int(hip_y)),
                        5,
                        (255, 0, 0),
                        -1
                    )

                    break

    # ========== 5. 畫出辨識結果 ==========

    available_count = 0

    for i, chair in enumerate(chair_boxes):

        x1, y1, x2, y2 = map(int, chair)

        if occupied[i]:
            color = (0, 0, 255)  # 紅色
            status = "Occupied"
        else:
            color = (0, 255, 0)  # 綠色
            status = "Available"
            available_count += 1

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            frame,
            f"Chair {i+1}: {status}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    # 顯示空位統計
    total = len(chair_boxes)

    cv2.putText(
        frame,
        f"Available: {available_count}/{total}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # ========== 6. 顯示畫面 ==========

    cv2.imshow("Restaurant Seat Detection", frame)

    # 按 Q 離開
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ========== 7. 釋放資源 ==========

cap.release()
cv2.destroyAllWindows()
