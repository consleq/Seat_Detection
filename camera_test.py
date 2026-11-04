import cv2

# Windows 使用 DirectShow
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) # 不行的話把0改成1或2試試看

if not cap.isOpened():
    print("無法開啟攝影機")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("讀取畫面失敗")
        break

    cv2.imshow("Iriun Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
