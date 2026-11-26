# 餐廳空位回報

使用固定相機拍攝座位，以 YOLO 即時偵測椅子和人體，並顯示空位數。

## 環境版本

- Windows（相機透過 DirectShow 讀取）
- Python `3.14.3`
- OpenCV `5.0.0.93`（套件名稱為 `opencv-python`）
- Ultralytics `8.4.198`
- firebase-admin `7.8.0`
- 姿勢模型 `yolov8n-pose.pt`
- 椅子偵測模型 `yolov8n.pt`

在專案資料夾開啟 PowerShell，建立虛擬環境並安裝套件。

```powershell
python -m pip install -r requirements.txt
```

## 基本操作

1. 使用 Iriun Webcam 時，先連接手機與電腦，確認電腦端看得到手機畫面。需用到鏡頭時，電腦端應用程式也要開著，不能關。可執行 `python camera_test.py` 測試影格，按 `Q` 結束。
2. 直接執行 `python main.py`。辨識從零個座位開始，偵測到椅子後才加入統計。紅色表示占用、綠色表示空位、黃色表示確認中。按 `Q` 或 Esc 結束。

即時座位會寫入專案目錄的 `temp_seats.json`，包含影像尺寸、座位 ID、四角座標和狀態。短暫漏偵測時會沿用上一個椅框。連續 3 秒都看不到椅子、也沒有配對到坐姿的人時，移除該座位。

若要使用固定座位，執行 `python seat_selector.py` 手動框選，按 `S` 儲存到 `seats.json`，再執行 `python main.py --seats seats.json`。
固定座位模式會關閉即時椅子偵測，並繼續用人體辨識判斷固定座位是否占用。

Iriun 相機預設使用編號 `1`。若要更換相機，修改 `seat_selector.py` 的 `CAMERA_ID`，`main.py` 會共用此設定。`camera_test.py` 的相機編號需在該檔案內另行修改。

## 參數設定

| 參數 | 預設 | 用途 |
| --- | --- | --- |
| `--seats 路徑` | 即時椅子偵測 | 讀取指定的固定座位設定檔，關閉即時椅子偵測。 |
| `--model 路徑` | `yolov8n-pose.pt` | 指定 YOLO 人體姿勢模型。 |
| `--conf 數值` | `0.4` | 設定人體偵測信心門檻，範圍為大於 0 至 1。 |
| `--chair-model 路徑` | `yolov8n.pt` | 指定即時模式使用的 YOLO 椅子偵測模型。 |
| `--chair-conf 數值` | `0.35` | 設定椅子偵測信心門檻，範圍為大於 0 至 1。 |
| `--occupy-seconds 秒數` | `2` | 持續偵測有人達到指定秒數後，確認座位已占用。 |
| `--empty-seconds 秒數` | `5` | 持續偵測無人達到指定秒數後，確認座位為空位。 |
| `--movement-threshold 比例` | `0.12` | 設定判為移動所需的外框中心位移，以人體框平均高度為基準。 |
| `--no-firebase` | 有憑證時同步 | 關閉 Firestore 同步，只在本機辨識。 |

多個參數可以用空格接在同一條指令後。`--no-firebase` 是開關，直接加上即可。

```powershell
python main.py --chair-conf 0.45 --occupy-seconds 3 --no-firebase
python main.py --seats seats.json --empty-seconds 8 --no-firebase
```

## 占用判定

站姿的第一項檢查是人體是否移動。程式比較同一人在約半秒內的外框中心位移，達到移動門檻時標示 `Moving`，並排除占用判定。其餘人員再依關節和胸口位置判斷站姿。即時椅框的胸口檢查範圍會往上延伸半個椅框高度。程式依人體框覆蓋椅框的比例分配座位，覆蓋至少 25% 才會計入。

## Firestore 同步

有 Firebase 憑證時，即時模式會把本次執行的座位同步到 `restaurants/test/seats`，以 `live_` 開頭的文件 ID 識別，正常結束時會移除本次的即時文件。缺少憑證時會使用本機辨識。

Firestore 會等每張椅子的框連續穩定 3 秒後才新增文件。已上傳的座標會忽略小幅抖動，有明顯變化且新位置穩定 3 秒才更新。占用狀態仍在確認後同步。連續 15 秒看不到椅子、也沒有配對到坐姿的人，才在雲端移除該座位。這些條件只用於雲端同步，本機 `temp_seats.json` 仍記錄即時結果。
