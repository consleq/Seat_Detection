"""以 YOLO 即時偵測椅子及人體，判斷座位占用。"""

# argparse 讀取命令列參數
import argparse
import json
import math
from pathlib import Path
# monotonic 計時不受使用者調整系統時間影響
import time
from uuid import uuid4

import cv2

from live_tracking import LiveSeats, PersonMotion
from stable_seats import StableSeats
from seat_selector import CAMERA_ID, load_seats
from pose_seat_detection import KPT_CONF_TH, SeatState, evaluate_seat_occupancy


WINDOW = "Restaurant Seat Detection"
TEMP_SEATS_PATH = Path(__file__).resolve().with_name("temp_seats.json")
LIVE_CHEST_MARGIN = 0.5
SNAPSHOT_INTERVAL = 0.5
FIREBASE_INTERVAL = 1.0
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
    parser = argparse.ArgumentParser(description="即時偵測椅子並判斷占用狀態。")
    parser.add_argument("--seats", type=Path, help="使用固定座位設定檔並關閉即時椅子偵測")
    parser.add_argument("--model", default="yolov8n-pose.pt", help="YOLO 姿勢模型檔案")
    parser.add_argument("--conf", type=float, default=0.4, help="人體偵測信心門檻")
    parser.add_argument("--chair-model", default="yolov8n.pt", help="YOLO 椅子物件偵測模型檔案")
    parser.add_argument("--chair-conf", type=float, default=0.35, help="椅子偵測信心門檻")
    parser.add_argument("--occupy-seconds", type=float, default=2.0)
    parser.add_argument("--empty-seconds", type=float, default=5.0)
    parser.add_argument("--movement-threshold", type=float, default=0.12,
                        help="判為移動所需的中心位移與人體框高度比（預設 0.12）")
    parser.add_argument("--no-firebase", action="store_true", help="只執行本機辨識")
    args = parser.parse_args()
    # conf 是模型對人體偵測的信心
    # 拒絕超出範圍的數值, 以及 NaN 或無限大
    if not math.isfinite(args.conf) or not 0 < args.conf <= 1:
        parser.error("--conf 必須大於 0 且不超過 1。")
    if not math.isfinite(args.chair_conf) or not 0 < args.chair_conf <= 1:
        parser.error("--chair-conf 必須大於 0 且不超過 1。")
    if not math.isfinite(args.movement_threshold) or args.movement_threshold <= 0:
        parser.error("--movement-threshold 必須是正數。")
    for name in ("occupy_seconds", "empty_seconds"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} 必須是非負秒數。")
    return args


def load_fixed_config(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("version") != 1:
        raise ValueError("不支援的座位設定格式，請用 seat_selector.py 重新設定。")
    # 沿用框選時記錄的解析度, 讓像素座標能對上實際影像
    size = config["image_size"]
    width, height = size["width"], size["height"]
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError("座位設定中的解析度必須是正整數。")
    return config, load_seats(path, width, height), width, height


def connect_firebase():
    try:
        import firebase_sync
    except ModuleNotFoundError as exc:
        if exc.name != "firebase_admin":
            raise
        print("未安裝 firebase-admin，使用本機辨識。")
        return None, None
    try:
        return firebase_sync, firebase_sync.init_firebase()
    except FileNotFoundError:
        print("找不到 Firebase 憑證，使用本機辨識。")
        return None, None


def write_temp_seats(snapshot):
    pending = TEMP_SEATS_PATH.with_suffix(".json.tmp")
    pending.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(TEMP_SEATS_PATH)


def main():
    args = parse_args()
    live = args.seats is None
    if live:
        live_seats = LiveSeats()
    else:
        config, fixed_seats, width, height = load_fixed_config(args.seats)
        fixed_states = [SeatState() for _ in fixed_seats]
    motion = PersonMotion(args.movement_threshold)

    from ultralytics import YOLO

    print(f"載入模型 {args.model}（首次使用且本機沒有檔案時，可能需要下載）。")
    model = YOLO(args.model)
    if model.task != "pose" or model.names.get(0) != "person":
        raise ValueError("請使用 class 0 為 person 的 YOLO 姿勢模型，例如 yolov8n-pose.pt。")
    if live:
        print(f"載入椅子偵測模型 {args.chair_model}（首次使用且本機沒有檔案時，可能需要下載）。")
        chair_model = YOLO(args.chair_model)
        if chair_model.task != "detect":
            raise ValueError("請使用支援 chair 類別的 YOLO 物件偵測模型，例如 yolov8n.pt。")
        chair_class = next(
            (class_id for class_id, name in chair_model.names.items() if name == "chair"), None
        )
        if chair_class is None:
            raise ValueError("椅子偵測模型缺少 chair 類別。")
    firebase, rest = (None, None) if args.no_firebase else connect_firebase()
    stable_seats = StableSeats() if live and firebase else None
    session_id = uuid4().hex
    published = {}
    fixed_published = False
    latest_snapshot = saved_snapshot = None
    last_snapshot_at = last_sync_at = -math.inf

    # 3. 透過 Windows DirectShow 開啟與框選工具相同編號的相機
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"無法開啟相機 {CAMERA_ID}，請修改 seat_selector.py 的 CAMERA_ID 後重試。")
        if not live:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if firebase:
                firebase.upload_seats(rest, config)
                fixed_published = True
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        print("即時椅子偵測" if live else f"已載入 {len(fixed_seats)} 個固定座位。")
        print(f"先檢查人體移動，再檢查姿勢；連續有人 {args.occupy_seconds:g} 秒占用，")
        print(f"連續無人 {args.empty_seconds:g} 秒空位。按 Q / Esc 或關閉視窗結束。")
        while True:
            # 4. 逐張讀取影格, 讀取失敗就停止, 不把斷線當作空位
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("無法讀取相機影格，已停止辨識；請確認 Iriun 連線。")
            if live:
                height, width = frame.shape[:2]
            elif frame.shape[:2] != (height, width):
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
            now = time.monotonic()
            if live:
                chair_result = chair_model(
                    frame, classes=[chair_class], conf=args.chair_conf, verbose=False
                )[0]
                chair_boxes = [] if chair_result.boxes is None else chair_result.boxes.xyxy.cpu().tolist()
                live_seats.update_boxes(chair_boxes, now, width, height)
                seats = [seat.as_seat() for seat in live_seats.seats]
            else:
                seats = fixed_seats
            moving = motion.update(boxes, now)
            chests, seat_detections, standing = evaluate_seat_occupancy(
                seats, boxes, keypoints, confidences, moving=moving,
                chest_margin=LIVE_CHEST_MARGIN if live else 0.0,
            )
            if live:
                active = live_seats.update_occupancy(
                    seat_detections, now, args.occupy_seconds, args.empty_seconds,
                )
                seat_rows = [(seat.as_seat(), seat.state, detected) for seat, detected in active]
                latest_snapshot = {
                    "version": 1,
                    "image_size": {"width": width, "height": height},
                    "seats": [
                        {**seat, "status": "checking" if state.occupied is None else state.occupied}
                        for seat, state, _ in seat_rows
                    ],
                }
                if latest_snapshot != saved_snapshot and now - last_snapshot_at >= SNAPSHOT_INTERVAL:
                    write_temp_seats(latest_snapshot)
                    saved_snapshot = latest_snapshot
                    last_snapshot_at = now
                if stable_seats is not None:
                    stable_seats.update(active, now)
                    if now - last_sync_at >= FIREBASE_INTERVAL:
                        published = firebase.sync_live_seats(
                            rest, session_id, stable_seats.snapshot(), published,
                        )
                        last_sync_at = now
            else:
                seat_rows = list(zip(seats, fixed_states, seat_detections))
                for seat, state, detected in seat_rows:
                    state.update(detected, now, args.occupy_seconds, args.empty_seconds)
                    if firebase:
                        firebase.update_status(seat["id"], state.occupied)

            # 人體外框: 灰色細框為站姿(已排除), 藍框為坐姿, 僅供檢查偵測是否正確
            for box, is_stand, is_moving in zip(boxes, standing, moving):
                x1, y1, x2, y2 = map(int, box)
                color = (160, 160, 160) if is_stand else (255, 160, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1 if is_stand else 2)
                label = "Moving" if is_moving else ("Stand" if is_stand else "Sit")
                cv2.putText(frame, label, (x1, max(12, y1 - 4)),
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

            for seat, state, detected in seat_rows:
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
                if state.candidate is not None and state.candidate != state.occupied:
                    target = "Occupied" if state.candidate else "Available"
                    text += f" -> {target} {now - state.since:.1f}s"
                cv2.putText(frame, text, (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # 7. 只把確認為 False 的座位計入空位, None 不算空位
            available = sum(state.occupied is False for _, state, _ in seat_rows)
            occupied = sum(state.occupied is True for _, state, _ in seat_rows)
            unknown = len(seat_rows) - available - occupied
            cv2.putText(frame, f"Available: {available}/{len(seat_rows)}  Occupied: {occupied}  Checking: {unknown}",
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
        if live:
            try:
                if latest_snapshot != saved_snapshot:
                    write_temp_seats(latest_snapshot)
            finally:
                if firebase and published:
                    firebase.sync_live_seats(rest, session_id, [], published)
        elif fixed_published:
            firebase.reset_all([seat["id"] for seat in fixed_seats])


if __name__ == "__main__":
    # 直接執行本檔案時才進入主流程, 匯入函式時不啟動辨識
    try:
        main()
    except (RuntimeError, ValueError, KeyError, TypeError, OSError, ImportError) as exc:
        print(f"錯誤：{exc}")
        raise SystemExit(1)
