"""YOLO pose keypoint helpers and the standalone camera demo."""

from dataclasses import dataclass

import numpy as np


KPT_CONF_TH = 0.4
STAND_KNEE_ANGLE = 160
STAND_THIGH_RATIO = 0.8
# 脖子到膝蓋距離 / 脖子到髖部距離 >= 此值視為站姿（坐姿約 1.0~1.3，站姿約 1.8~2.0）
STAND_NECK_KNEE_RATIO = 1.8
CHEST_KEYPOINT_CONF = 0.4
IOU_THRESHOLD = 0.25
# 人體框與座位框的交集面積 / 座位面積 >= 此值才算坐在該座位
SEAT_OVERLAP_THRESHOLD = 0.25


def compute_iou(box_a, box_b):
    """計算兩個 xyxy 框的 IoU。"""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


def seat_overlap_ratio(person_box, seat_box_):
    """回傳人體框與座位框的交集面積 / 座位面積（0~1）。"""
    ix1 = max(person_box[0], seat_box_[0])
    iy1 = max(person_box[1], seat_box_[1])
    ix2 = min(person_box[2], seat_box_[2])
    iy2 = min(person_box[3], seat_box_[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    seat_area = (seat_box_[2] - seat_box_[0]) * (seat_box_[3] - seat_box_[1])
    if inter == 0 or seat_area <= 0:
        return 0.0
    return inter / seat_area


def joint_angle(a, b, c):
    """回傳以 b 為頂點的 a-b-c 夾角（度）。"""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0.0
    cos = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def is_standing_by_neck_knee(kpts, conf, ratio_th=STAND_NECK_KNEE_RATIO):
    """以脖子到膝蓋的距離判斷站姿。

    脖子取左右肩中點。坐姿時膝蓋與髖部約同高，脖子到膝蓋的距離
    約等於軀幹長；站姿時則約為軀幹長加大腿長。
    以「脖子到膝蓋距離 / 脖子到髖部距離」正規化，避免受遠近影響。
    任一側膝蓋達門檻即視為站姿；關節點不可信時回傳 False。
    """

    def ok(index):
        return conf is None or conf[index] >= KPT_CONF_TH

    if len(kpts) <= 14 or not (ok(5) and ok(6)):
        return False
    neck = (np.array(kpts[5], dtype=float) + np.array(kpts[6], dtype=float)) / 2

    visible_hips = [np.array(kpts[h], dtype=float) for h in (11, 12) if ok(h)]
    if not visible_hips:
        return False
    torso = np.linalg.norm(np.mean(visible_hips, axis=0) - neck)
    if torso <= 0:
        return False

    for knee in (13, 14):
        if not ok(knee):
            continue
        neck_knee = np.linalg.norm(np.array(kpts[knee], dtype=float) - neck)
        if neck_knee / torso >= ratio_th:
            return True
    return False


def is_standing(kpts, conf, person_box=None, chair_box=None, chest_outside=False):
    """依膝蓋角度與距離比、胸口位置、大腿與軀幹比例判斷站姿。"""

    def ok(index):
        return conf is None or conf[index] >= KPT_CONF_TH

    sides = ((11, 13, 15, 5), (12, 14, 16, 6))

    # 1. 膝蓋角度(hip, knee, ankle)：任一側腿成一直線才進入第 2 步確認
    #    （坐著把腿伸直時角度也會成一直線，所以不能單憑角度判定站姿）
    leg_straight = any(
        ok(hip) and ok(knee) and ok(ankle)
        and joint_angle(kpts[hip], kpts[knee], kpts[ankle]) >= STAND_KNEE_ANGLE
        for hip, knee, ankle, _ in sides
    )

    # 2. 脖子到膝蓋 / 脖子到髖部距離比
    if leg_straight and is_standing_by_neck_knee(kpts, conf):
        return True

    # 3. 胸口位置(胸口是否不在對應的座位 ROI 內，由呼叫端算好傳入)
    if chest_outside:
        return True

    # 4. 大腿與軀幹比例
    for hip, knee, _, shoulder in sides:
        if not (ok(hip) and ok(knee)):
            continue
        if ok(shoulder):
            torso = abs(kpts[hip][1] - kpts[shoulder][1])
            thigh_v = kpts[knee][1] - kpts[hip][1]
            if torso > 0 and thigh_v / torso >= STAND_THIGH_RATIO:
                return True

    return False


def chest_points(keypoints, confidences, confidence_threshold=CHEST_KEYPOINT_CONF):
    """以左右肩關節的中點代表胸口；肩膀不可信時填 None，順序與人體框相同。"""
    chests = []
    for index, person_keypoints in enumerate(keypoints):
        if len(person_keypoints) <= 6:
            chests.append(None)
            continue
        if confidences is not None and (
            index >= len(confidences)
            or len(confidences[index]) <= 6
            or confidences[index][5] < confidence_threshold
            or confidences[index][6] < confidence_threshold
        ):
            chests.append(None)
            continue
        left_shoulder = person_keypoints[5]
        right_shoulder = person_keypoints[6]
        chests.append((
            (float(left_shoulder[0]) + float(right_shoulder[0])) / 2,
            (float(left_shoulder[1]) + float(right_shoulder[1])) / 2,
        ))
    return chests


def seat_box(points):
    """四邊形座位 ROI 的外接矩形 (x1, y1, x2, y2)。"""
    x1, y1 = points[0]
    x2, y2 = points[2]
    return (x1, y1, x2, y2)


def chest_outside_seat(chest, box):
    """胸口不在對應座位 ROI 內時回傳 True（站姿）。胸口不可信時回傳 False。

    box 為 None 代表人體框與所有座位 ROI 都沒有重疊，胸口必然在 ROI 外。
    """
    if chest is None:
        return False
    if box is None:
        return True
    x, y = chest
    return not (box[0] <= x <= box[2] and box[1] <= y <= box[3])


def evaluate_seat_occupancy(seats, boxes, keypoints, confidences):
    """1. 辨識出所有人 2. 排除站姿 3. 坐姿的人以 IoU 決定坐哪張椅子。

    回傳 (chests, detected, standing):
    chests 每個人的胸口點或 None, detected 每個座位是否有人坐,
    standing 每個人是否判定為站姿。
    """
    seat_boxes = [seat_box(seat["points"]) for seat in seats]
    chests = chest_points(keypoints, confidences)

    # 2. 由胸口高度及姿勢關節判斷站姿, 先排除站姿的人
    standing = []
    for index, person_box in enumerate(boxes):
        related_seat = max(
            seat_boxes, key=lambda box: compute_iou(person_box, box), default=None
        )
        if related_seat is not None and compute_iou(person_box, related_seat) == 0:
            related_seat = None
        chest = chests[index] if index < len(chests) else None
        kpts = keypoints[index] if index < len(keypoints) else []
        conf = (
            confidences[index]
            if confidences is not None and index < len(confidences) else None
        )
        chest_out = chest_outside_seat(chest, related_seat)
        if len(kpts) > 16:
            standing.append(
                is_standing(kpts, conf, person_box, related_seat, chest_out)
            )
        else:
            # 關節點不足時只能靠胸口位置判斷
            standing.append(chest_out)

    # 3. 坐姿的人與座位 ROI 計算「交集面積 / 座位面積」, 取最大且超過門檻的座位
    detected = [False] * len(seats)
    for index, person_box in enumerate(boxes):
        if standing[index]:
            continue
        best_overlap = 0.0
        best_seat = -1
        for seat_index, box in enumerate(seat_boxes):
            overlap = seat_overlap_ratio(person_box, box)
            if overlap > best_overlap:
                best_overlap = overlap
                best_seat = seat_index
        if best_seat >= 0 and best_overlap >= SEAT_OVERLAP_THRESHOLD:
            detected[best_seat] = True
    return chests, detected, standing


@dataclass
class SeatState:
    """追蹤座位占用狀態，並以持續時間避免短暫誤判。"""

    occupied: bool | None = None
    candidate: bool | None = None
    since: float = 0.0

    def update(self, detected, now, occupy_seconds, empty_seconds):
        if self.candidate != detected:
            self.candidate = detected
            self.since = now
        required = occupy_seconds if detected else empty_seconds
        if now - self.since >= required:
            self.occupied = detected


def run_standalone_demo():
    """保留此檔案原本的即時相機椅子占用示範。"""
    import cv2
    from ultralytics import YOLO

    det_model = YOLO("yolov8n.pt")
    pose_model = YOLO("yolov8n-pose.pt")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("無法開啟攝影機")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("無法讀取畫面")

            det_result = det_model(frame, classes=[0], conf=0.35, verbose=False)[0]
            pose_result = pose_model(frame, conf=0.4, verbose=False)[0]
            chair_boxes = (
                det_result.boxes.xyxy.cpu().numpy()
                if det_result.boxes is not None else []
            )
            occupied = [False] * len(chair_boxes)
            seated_boxes = []

            if pose_result.boxes is not None and len(pose_result.boxes) > 0:
                person_boxes = pose_result.boxes.xyxy.cpu().numpy()
                keypoints = (
                    pose_result.keypoints.xy.cpu().numpy()
                    if pose_result.keypoints is not None
                    and pose_result.keypoints.xy is not None else None
                )
                keypoint_conf = (
                    pose_result.keypoints.conf.cpu().numpy()
                    if pose_result.keypoints is not None
                    and pose_result.keypoints.conf is not None else None
                )

                for person_idx, person_box in enumerate(person_boxes):
                    if keypoints is not None:
                        conf = (
                            keypoint_conf[person_idx]
                            if keypoint_conf is not None else None
                        )
                        related_chair = max(
                            chair_boxes,
                            key=lambda chair: compute_iou(person_box, chair),
                            default=None,
                        )
                        if is_standing(
                            keypoints[person_idx], conf, person_box, related_chair
                        ):
                            x1, y1, x2, y2 = map(int, person_box)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 160, 160), 1)
                            continue
                    seated_boxes.append(person_box)

            for person_box in seated_boxes:
                best_iou = 0.0
                best_chair = -1
                for index, chair in enumerate(chair_boxes):
                    iou = compute_iou(person_box, chair)
                    if iou > best_iou:
                        best_iou = iou
                        best_chair = index
                x1, y1, x2, y2 = map(int, person_box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                if best_chair >= 0 and best_iou >= IOU_THRESHOLD:
                    occupied[best_chair] = True
                    cv2.putText(
                        frame, f"IoU {best_iou:.2f}",
                        (x1, min(frame.shape[0] - 5, y2 + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1,
                    )

            available_count = 0
            for index, chair in enumerate(chair_boxes):
                x1, y1, x2, y2 = map(int, chair)
                color = (0, 0, 255) if occupied[index] else (0, 255, 0)
                status = "Occupied" if occupied[index] else "Available"
                available_count += not occupied[index]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, f"Chair {index + 1}: {status}",
                    (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 2,
                )
            cv2.putText(
                frame, f"Available: {available_count}/{len(chair_boxes)}",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
            )
            cv2.imshow("Restaurant Seat Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_standalone_demo()
