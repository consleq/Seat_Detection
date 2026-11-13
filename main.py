"""固定座位 ROI + YOLO 姿勢偵測；姿勢判斷集中在 pose_seat_detection.py

流程: 1. 辨識出所有人 2. 由胸口高度及姿勢關節排除站姿 3. 坐姿的人以 ROI 的 IoU 決定座位

先執行 python seat_selector.py 框選並儲存, 再執行 python main.py
"""

# argparse 讀取命令列參數
import argparse
import json
import math
from pathlib import Path
# monotonic 計時不受使用者調整系統時間影響
import time

import cv2

# 共用相機編號, 設定檔位置和座位驗證函式, 匯入時不會啟動框選視窗
from seat_selector import CAMERA_ID, OUTPUT_PATH, load_seats
from pose_seat_detection import KPT_CONF_TH, SeatState, evaluate_seat_occupancy


WINDOW = "Restaurant Seat Detection"
JOINT_MARKERS = (
    (11, "L Hip", (0, 255, 255)),
    (12, "R Hip", (0, 255, 255)),
    (13, "L Knee", (0, 165, 255)),
    (14, "R Knee", (0, 165, 255)),
    (15, "L Ankle", (0, 0, 255)),
    (16, "R Ankle", (0, 0, 255)),
)


def parse_args():
    # 直接執行時採用 default, 也能以命令列暫時覆寫辨識參數
    # 例如: python main.py --occupy-seconds 3
    parser = argparse.ArgumentParser(description="排除站姿後，以人體框與座位 ROI 的 IoU 偵測占用狀態。")
    parser.add_argument("--seats", type=Path,
                        default=OUTPUT_PATH)
    parser.add_argument("--model", default="yolov8n-pose.pt", help="YOLO 姿勢模型檔案")
    parser.add_argument("--conf", type=float, default=0.4, help="人體偵測信心門檻")
    parser.add_argument("--occupy-seconds", type=float, default=2.0)
    parser.add_argument("--empty-seconds", type=float, default=5.0)
    args = parser.parse_args()
    # conf 是模型對人體偵測的信心
    # 拒絕超出範圍的數值, 以及 NaN 或無限大
    if not math.isfinite(args.conf) or not 0 < args.conf <= 1:
        parser.error("--conf 必須大於 0 且不超過 1。")
    for name in ("occupy_seconds", "empty_seconds"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} 必須是非負秒數。")
    return args


def main():
    # 1. 先讀取並驗證座位設定, 有問題時不載入模型或開啟相機
    args = parse_args()
    if not args.seats.is_file():
        raise ValueError("找不到座位設定檔，請先執行 python seat_selector.py 並按 S 儲存。")
    config = json.loads(args.seats.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("version") != 1:
        raise ValueError("不支援的座位設定格式，請用 seat_selector.py 重新設定。")
    # 沿用框選時記錄的解析度, 讓像素座標能對上實際影像
    size = config["image_size"]
    width, height = size["width"], size["height"]
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError("座位設定中的解析度必須是正整數。")
    seats = load_seats(args.seats, width, height)
    if not seats:
        raise ValueError("尚未設定座位，請先執行 python seat_selector.py，框選後按 S 儲存。")

    # 2. 座位設定有效後才載入 YOLO 姿勢模型
    from ultralytics import YOLO

    print(f"載入模型 {args.model}（首次使用且本機沒有檔案時，可能需要下載）。")
    model = YOLO(args.model)
    if model.task != "pose" or model.names.get(0) != "person":
        raise ValueError("請使用 class 0 為 person 的 YOLO 姿勢模型，例如 yolov8n-pose.pt。")
    # 3. 透過 Windows DirectShow 開啟與框選工具相同編號的相機
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"無法開啟相機 {CAMERA_ID}，請修改 seat_selector.py 的 CAMERA_ID 後重試。")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # 每個座位各自持有狀態和計時器, 不共用同一個 SeatState 物件
        states = [SeatState() for _ in seats]
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        print(f"已載入 {len(seats)} 個座位；請確認機位與框選時一致。")
        print(f"胸口高度與關節排除站姿，坐姿以 IoU 對應座位；連續有人 {args.occupy_seconds:g} 秒占用，")
        print(f"連續無人 {args.empty_seconds:g} 秒空位。按 Q / Esc 或關閉視窗結束。")
        while True:
            # 4. 逐張讀取影格, 讀取失敗就停止, 不把斷線當作空位
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("無法讀取相機影格，已停止辨識；請確認 Iriun 連線。")
            if frame.shape[:2] != (height, width):
                raise ValueError("實際相機解析度與座位設定不符，請以相同解析度重新框選。")
            # 5. 在完整畫面偵測人體與姿勢, classes=[0] 限定 person 類別
            # 回傳結果是清單, 單張畫面取第 0 筆
            result = model(frame, classes=[0], conf=args.conf, verbose=False)[0]
            # xyxy 為 [左, 上, 右, 下], 轉到 CPU 後再轉成 Python 清單
            boxes = [] if result.boxes is None else result.boxes.xyxy.cpu().tolist()
            keypoints = ( # 關節點座標, 轉到 CPU 後再轉成 Python 清單
                []
                if result.keypoints is None or result.keypoints.xy is None
                else result.keypoints.xy.cpu().tolist()
            )
            confidences = (
                None
                if result.keypoints is None or result.keypoints.conf is None
                else result.keypoints.conf.cpu().tolist()
            )
            chests, seat_detections, standing = evaluate_seat_occupancy(
                seats, boxes, keypoints, confidences
            )
            now = time.monotonic()
            # 人體外框: 灰色細框為站姿(已排除), 藍框為坐姿, 僅供檢查偵測是否正確
            for box, is_stand in zip(boxes, standing):
                x1, y1, x2, y2 = map(int, box)
                color = (160, 160, 160) if is_stand else (255, 160, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1 if is_stand else 2)
                cv2.putText(frame, "Stand" if is_stand else "Sit", (x1, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # 6. 在畫面上標示胸口點與座位狀態, 僅供檢查偵測是否正確
            for chest in chests:
                if chest is None:
                    continue
                chest_x, chest_y = chest
                cv2.circle(frame, (round(chest_x), round(chest_y)), 4, (255, 0, 255), -1)

            for person_index, person_keypoints in enumerate(keypoints):
                person_confidences = (
                    confidences[person_index]
                    if confidences is not None and person_index < len(confidences)
                    else None
                )
                for joint_index, label, color in JOINT_MARKERS:
                    if joint_index >= len(person_keypoints):
                        continue
                    if person_confidences is not None and (
                        joint_index >= len(person_confidences)
                        or person_confidences[joint_index] < KPT_CONF_TH
                    ):
                        continue
                    x, y = person_keypoints[joint_index]
                    point = (round(float(x)), round(float(y)))
                    cv2.circle(frame, point, 5, color, -1)
                    cv2.putText(
                        frame, label, (point[0] + 5, point[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                    )

            for seat, state, detected in zip(seats, states, seat_detections):
                state.update(detected, now,
                             args.occupy_seconds, args.empty_seconds)
                # 顯示確認後的狀態: 黃色待確認, 紅色占用, 綠色空位
                if state.occupied is None:
                    label, color = "Checking", (0, 200, 255)
                elif state.occupied:
                    label, color = "Occupied", (0, 0, 255)
                else:
                    label, color = "Available", (0, 255, 0)
                x1, y1 = seat["points"][0]
                x2, y2 = seat["points"][2]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"{seat['id']}: {label} Sit {'in' if detected else 'out'}"
                # 若當前結果與確認狀態不同, 額外顯示目標狀態與已等待秒數
                if state.candidate != state.occupied:
                    target = "Occupied" if state.candidate else "Available"
                    text += f" -> {target} {now - state.since:.1f}s"
                cv2.putText(frame, text, (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # 7. 只把確認為 False 的座位計入空位, None 不算空位
            available = sum(state.occupied is False for state in states)
            occupied = sum(state.occupied is True for state in states)
            unknown = len(seats) - available - occupied
            cv2.putText(frame, f"Available: {available}/{len(seats)}  Occupied: {occupied}  Checking: {unknown}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # 8. 顯示結果並處理鍵盤和視窗事件, Q 或 Esc 結束
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        # 即使讀取或辨識發生例外, 也會釋放相機並關閉視窗
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 直接執行本檔案時才進入主流程, 匯入函式時不啟動辨識
    try:
        main()
    except (RuntimeError, ValueError, KeyError, TypeError, OSError, ImportError) as exc:
        print(f"錯誤：{exc}")
        raise SystemExit(1)