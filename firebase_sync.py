import json
import firebase_admin
from firebase_admin import credentials, firestore

_seats_ref = None

def init_firebase(key_path="restaurant-seats-firebase-adminsdk-fbsvc-6476a323fd.json", restaurant_id="test"):
    """連線 Firebase，只能呼叫一次"""
    global _seats_ref
    if not firebase_admin._apps:          # 避免重複 initialize_app 報錯
        cred = credentials.Certificate(key_path)
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    rest = db.collection("restaurants").document(restaurant_id)
    _seats_ref = rest.collection("seats")
    return rest

def upload_seats(rest, config):
    """把 seats.json 的座位位置上傳，狀態設為 checking；並刪除 Firebase 上不在 seats.json 的舊座位"""
    current_ids = {seat["id"] for seat in config["seats"]}
    for doc in rest.collection("seats").stream():
        if doc.id not in current_ids:
            print(f"刪除 Firebase 上多餘的座位：{doc.id}")
            doc.reference.delete()
    for seat in config["seats"]:
        (x1, y1), _, (x2, y2), _ = seat["points"]
        rest.collection("seats").document(seat["id"]).set({
            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "status": "checking",
        }, merge=True)

def update_status(seat_id, status):
    """更新單一座位狀態"""
    _seats_ref.document(seat_id).update({
        "status": status,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    })

def reset_all(seat_ids):
    """程式結束時全部設回 checking"""
    for seat_id in seat_ids:
        update_status(seat_id, "checking")