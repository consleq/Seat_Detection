# 餐廳空位回報

使用固定相機拍攝座位，以 YOLO 偵測人體，根據人體外框與座位範圍的重疊程度顯示空位數。

## 環境版本

- Windows（相機透過 DirectShow 讀取）
- Python `3.14.3`
- OpenCV `5.0.0.93`（套件名稱為 `opencv-python`）
- Ultralytics `8.4.198`
- 姿勢模型 `yolov8n-pose.pt`

在專案資料夾開啟 PowerShell，建立虛擬環境並安裝套件。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install opencv-python==5.0.0.93 ultralytics==8.4.198
```

## 基本操作

1. 使用 Iriun Webcam 時，先連接手機與電腦，確認電腦端看得到手機畫面。需用到鏡頭時，電腦端應用程式也要開著，不能關。可執行 `python camera_test.py` 測試影格，按 `Q` 結束。
2. 執行 `python seat_selector.py`。在快照上用滑鼠左鍵框選各座位。`U` 刪除最後一框，`R` 更新快照，`S` 將座位存到 `seats.json`，`Q` 離開。
3. 關閉框選視窗後執行 `python main.py`。畫面上紅色表示占用、綠色表示空位、黃色表示確認中。按 `Q` 結束。

Iriun 相機預設使用編號 `1`。若要更換相機，修改 `seat_selector.py` 的 `CAMERA_ID`，`main.py` 會共用此設定。`camera_test.py` 的相機編號需在該檔案內另行修改。

`main.py` 載入 YOLO 姿勢模型； `pose_seat_detection.py` 的 `evaluate_seat_occupancy` 彙整多個判斷邏輯，包含：1. 辨識出所有人；2. 由胸口高度（左右肩中點，高於座位 ROI 上緣超過 ROI 高度 25% 視為站姿）及姿勢關節（`is_standing`）先排除站姿的人；3. 坐姿的人以人體框與座位 ROI 的 IoU 決定坐哪張椅子（IoU ≥ `IOU_THRESHOLD`）。座位有人持續 2 秒時標示占用，連續 5 秒無人時標示空位。畫面中灰色細框為站姿、藍框為坐姿，紫色點表示胸口位置。

檔案多和 main 分支差不多，較大的程式碼構造差別是 pose_seat_detection.py 及 main.py 被分成多個函式，且放入新增的判斷邏輯。
