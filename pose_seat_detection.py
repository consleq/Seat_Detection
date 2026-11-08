"""YOLO pose keypoint helpers and the standalone camera demo."""

from dataclasses import dataclass

import numpy as np


KPT_CONF_TH = 0.4
STAND_KNEE_ANGLE = 120
STAND_THIGH_RATIO = 0.8
STAND_PERSON_CHAIR_HEIGHT_RATIO = 1.5
STAND_HIP_CHAIR_HEIGHT_RATIO = 0.55
CHEST_KEYPOINT_CONF = 0.4
# 胸口高於座位 ROI 上緣超過「ROI 高度 x 此比例」時視為站姿
CHEST_ABOVE_SEAT_RATIO = 0.25
IOU_THRESHOLD = 0.25


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


def joint_angle(a, b, c):
    """回傳以 b 為頂點的 a-b-c 夾角（度）。"""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0.0
    cos = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def is_standing(kpts, conf, person_box=None, chair_box=None):
    """依關節點判斷站姿；腳踝不可見時可參考附近椅子的相對高度。"""

    def ok(index):
        return conf is None or conf[index] >= KPT_CONF_TH

    sides = ((11, 13, 15, 5), (12, 14, 16, 6))
    for hip, knee, ankle, shoulder in sides:
        if not (ok(hip) and ok(knee)):
            continue
        # 以關節角度(hip, knee, ankle)為第一判斷條件
        if ok(ankle):
            if joint_angle(kpts[hip], kpts[knee], kpts[ankle]) >= STAND_KNEE_ANGLE:
                return True
        # 若腳踝不可見，則以大腿與軀幹比例為第二判斷條件
        if ok(shoulder):
            torso = abs(kpts[hip][1] - kpts[shoulder][1])
            thigh_v = kpts[knee][1] - kpts[hip][1]
            if torso > 0 and thigh_v / torso >= STAND_THIGH_RATIO:
                return True

    # 若所有腳踝都不可見，則以大腿與軀幹比例為判斷條件
    if not any(ok(ankle) for _, _, ankle, _ in sides):
        thigh_ratios = []
        for hip, knee, _, shoulder in sides:
            if not (ok(hip) and ok(knee) and ok(shoulder)):
                break
            torso = kpts[hip][1] - kpts[shoulder][1]
            thigh_v = kpts[knee][1] - kpts[hip][1]
            if torso <= 0 or thigh_v <= 0:
                break
            thigh_ratios.append(thigh_v / torso)
        if len(thigh_ratios) == len(sides) and min(thigh_ratios) >= 0.65:
            return True

        if chair_box is not None and person_box is not None:
            chair_height = chair_box[3] - chair_box[1]
            person_height = person_box[3] - person_box[1]
            visible_hips = [kpts[hip][1] for hip, _, _, _ in sides if ok(hip)]
            if (
                chair_height > 0
                and visible_hips
                and person_height / chair_height >= STAND_PERSON_CHAIR_HEIGHT_RATIO
                and np.mean(visible_hips)
                <= chair_box[1] + chair_height * STAND_HIP_CHAIR_HEIGHT_RATIO
            ):
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


def chest_above_seat(chest, box):
    """胸口明顯高於座位 ROI 上緣時回傳 True（站姿）。"""
    if chest is None:
        return False
    return chest[1] < box[1] - (box[3] - box[1]) * CHEST_ABOVE_SEAT_RATIO


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
        standing.append(
            (related_seat is not None and chest_above_seat(chest, related_seat))
            or (len(kpts) > 16 and is_standing(kpts, conf, person_box, related_seat))
        )

    # 3. 坐姿的人與座位 ROI 計算 IoU, 取最大且超過門檻的座位
    detected = [False] * len(seats)
    for index, person_box in enumerate(boxes):
        if standing[index]:
            continue
        best_iou = 0.0
        best_seat = -1
        for seat_index, box in enumerate(seat_boxes):
            iou = compute_iou(person_box, box)
            if iou > best_iou:
                best_iou = iou
                best_seat = seat_index
        if best_seat >= 0 and best_iou >= IOU_THRESHOLD:
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

            det_result = det_model(frame, classes=[56], conf=0.35, verbose=False)[0]
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
