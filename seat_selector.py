"""
用滑鼠設定座位 ROI
不執行人體或座位占用辨識.

執行: python seat_selector.py
相機編號, 解析度與儲存位置可在下方設定區修改.
"""

# json 負責讀寫座位設定, Path 負責組合檔案路徑
import json
from pathlib import Path

# OpenCV 負責讀取相機, 顯示畫面和接收滑鼠操作
import cv2


# ========== 設定區 ==========
CAMERA_ID = 1  # 相機編號, main.py 也會用這個設定
FRAME_WIDTH = 640  # 向相機要求的解析度, 實際值仍以讀到的影格為準
FRAME_HEIGHT = 480
# __file__ 是本程式路徑, seats.json固定放在與本程式同個資料夾
OUTPUT_PATH = Path(__file__).resolve().with_name("seats.json")

WINDOW = "Seat ROI Selector"
MIN_SIZE = 10  # 矩形的寬與高至少 10 像素, 避免誤點產生太小的 ROI


def load_seats(path, width, height):
    """載入座位並檢查格式, 此函式也會由 main.py 使用"""
    # 尚未建立設定檔時, 回傳空清單供使用者新增座位
    if not path.exists():
        return []
    # read_text 取得文字, json.loads 將 JSON 文字轉成 Python 資料
    data = json.loads(path.read_text(encoding="utf-8"))
    # 像素座標依賴畫面大小, 解析度不同就不能直接沿用
    if data.get("image_size") != {"width": width, "height": height}:
        raise ValueError("設定檔解析度與目前畫面不同，請使用相同解析度或修改 OUTPUT_PATH 後重新框選。")
    seats = data["seats"]
    if not isinstance(seats, list):
        raise ValueError("seats 必須是清單。")
    
    ids = set() # set 用來記錄出現過的編號, 避免重複
    for seat in seats:
        seat_id = seat["id"]
        if not isinstance(seat_id, str) or not seat_id or seat_id in ids:
            raise ValueError("座位編號必須是非空白且不重複的字串。")
        ids.add(seat_id)
        
        points = seat["points"] # 每個point包含 4 個 [x, y], x 向右增加, y 向下增加
        if not isinstance(points, list) or len(points) != 4:
            raise ValueError("每個座位必須包含四個角落。")
        for point in points:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError("角落座標格式錯誤。")
            x, y = point
            if type(x) is not int or type(y) is not int:
                raise ValueError("座標必須是整數。")
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError("座標超出目前畫面範圍。")
        # 角落順序固定為左上, 右上, 右下, 左下
        x1, y1 = points[0]
        x2, y2 = points[2]
        if points != [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]:
            raise ValueError("此工具只支援矩形座位區域。")
        if x2 - x1 < MIN_SIZE or y2 - y1 < MIN_SIZE:
            raise ValueError("座位矩形太小。")
    return seats


def main():
    # CAMERA_ID 選擇相機, CAP_DSHOW 指定 Windows 的 DirectShow 介面
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"無法開啟相機 {CAMERA_ID}，請修改設定區的 CAMERA_ID 後重試。")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        # ok 表示讀取是否成功, frame 是以 NumPy 陣列表示的影像
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("相機已開啟，但無法讀取影格。請確認 Iriun 連線。")
        # shape 的前兩項依序是高度與寬度, 不是寬度與高度
        height, width = frame.shape[:2]
        latest = frame.copy()
        seats = load_seats(OUTPUT_PATH, width, height)
        # 以 JSON 往返建立獨立副本, 避免修改 seats 時連儲存基準也一起改動
        saved_seats = json.loads(json.dumps(seats))
        # None 表示目前沒有拖曳, start 與 end 分別記錄拖曳起點和終點
        drag = {"start": None, "end": None}

        def on_mouse(event, x, y, flags, param):
            # OpenCV 在滑鼠事件發生時呼叫此函式, flags 和 param 目前未使用
            # 將滑鼠座標限制在有效影像範圍內
            point = (max(0, min(x, width - 1)), max(0, min(y, height - 1)))
            if event == cv2.EVENT_LBUTTONDOWN:
                drag["start"] = point
                drag["end"] = point
            elif event == cv2.EVENT_MOUSEMOVE and drag["start"] is not None:
                drag["end"] = point
            elif event == cv2.EVENT_LBUTTONUP and drag["start"] is not None:
                # sorted 讓不管怎麼拖也能得到左上與右下
                x1, x2 = sorted((drag["start"][0], point[0]))
                y1, y2 = sorted((drag["start"][1], point[1]))
                drag["start"] = drag["end"] = None
                if x2 - x1 < MIN_SIZE or y2 - y1 < MIN_SIZE:
                    print("框選太小，請重新拖曳（寬、高至少 10 像素）。")
                    return
                # 從 S01 起尋找未使用的編號, :02d 表示至少兩位數並補零
                used = {seat["id"] for seat in seats}
                number = 1
                while f"S{number:02d}" in used: # 字串格式化
                    number += 1
                seat = {"id": f"S{number:02d}",
                        "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]}
                # append 只改記憶體內的清單, 按 S 才會寫入檔案
                seats.append(seat)
                print(f"新增 {seat['id']}: {seat['points']}")

        # 建立視窗後綁定滑鼠回呼, 才能在此視窗接收框選操作
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, on_mouse)
        print(f"相機 {CAMERA_ID}，實際解析度 {width} x {height}；已載入 {len(seats)} 個座位。")
        print("畫面為靜態快照。滑鼠左鍵拖曳:新增矩形; U: 刪除最後一個; R: 更新快照。")
        print("S: 儲存; Q / Esc: 離開（不會自動儲存）。")
        print(f"設定檔：{OUTPUT_PATH.resolve()}")
        print("載入既有座標時，請確認相機位置與取景未變動。")

        while True:
            # 在快照副本上畫框, 避免舊框和文字永久留在原始快照上
            canvas = frame.copy()
            for seat in seats:
                x1, y1 = seat["points"][0]
                x2, y2 = seat["points"][2]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(canvas, seat["id"], (x1, max(16, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            if drag["start"] is not None:
                cv2.rectangle(canvas, drag["start"], drag["end"], (0, 255, 255), 2)
            # 比較目前資料與最後儲存的副本, 判斷是否有尚未儲存的變更
            dirty = seats != saved_seats
            # OpenCV 預設字型不支援中文, 視窗內使用英文
            # 顏色使用 BGR 順序, 例如 (0, 255, 0) 是綠色
            cv2.putText(canvas, f"Seats: {len(seats)} | {'UNSAVED' if dirty else 'SAVED'}",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
            cv2.putText(canvas, "Drag: Add | U: Undo | R: Refresh | S: Save | Q: Quit",
                        (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
            cv2.imshow(WINDOW, canvas)
            # waitKey 同時處理視窗事件與鍵盤輸入, 27 是 Esc 的鍵碼
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                if dirty:
                    print("已離開，未儲存的變更已捨棄。")
                break
            if key in (ord("u"), ord("U")):
                # pop 移除最後新增的座位, 但還沒有寫回設定檔
                drag["start"] = drag["end"] = None
                if seats:
                    print(f"刪除 {seats.pop()['id']}，按 S 才會儲存刪除結果。")
            elif key in (ord("r"), ord("R")):
                drag["start"] = drag["end"] = None
                # 持續讀取但只在 R 時替換快照, 以減少相機緩衝的舊畫面
                frame = latest.copy()
                print("已更新快照，座位座標保留。")
            elif key in (ord("s"), ord("S")):
                # 同時儲存格式版本, 實際解析度和座位座標
                data = {"version": 1, "image_size": {"width": width, "height": height},
                        "seats": seats}
                try:
                    # indent=2 讓檔案易讀, ensure_ascii=False 保留中文字元
                    OUTPUT_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
                except OSError as exc:
                    print(f"儲存失敗：{exc}")
                else:
                    saved_seats = json.loads(json.dumps(seats))
                    print(f"已儲存 {len(seats)} 個座位至 {OUTPUT_PATH.resolve()}")
            # 背景持續取得最新影格, 畫面仍維持快照直到按下 R
            ok, current = cap.read()
            if ok and current is not None:
                if current.shape[:2] != (height, width):
                    raise RuntimeError("相機解析度在執行中改變，請重新啟動並確認座標。")
                latest = current
    finally:
        # 正常離開或發生例外時都釋放相機, 並關閉 OpenCV 視窗
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 只有直接執行本檔案才啟動工具, 被 main.py 匯入時不會開啟相機
    try:
        main()
    except (RuntimeError, ValueError, KeyError, TypeError, OSError) as exc:
        # 將預期的操作錯誤顯示在 Terminal, 並以非零狀態結束程式
        print(f"錯誤:{exc}")
        raise SystemExit(1)
