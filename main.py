import cv2
import csv
import os
import winsound
from datetime import datetime
from ultralytics import YOLO

# 1. Initialize Folders & Settings
LOG_DIR = "pothole_logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, "detection_history.csv")
logged_potholes = set()

# 2. Setup Log File Header
if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Track_ID", "Confidence", "Distance", "Image_Path"])

# 3. Load Model
model = YOLO('best.pt')

# 4. Connect Phone Stream (Check your IP: 105)
url = "http://192.168.68.105:8080/video"
cap = cv2.VideoCapture(url)

print("\n" + "=" * 50)
print(f"{'TIME':<10} | {'ID':<5} | {'CONF':<6} | {'DIST':<6} | STATUS")
print("=" * 50)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    results = model.track(frame, conf=0.6, persist=True, tracker="bytetrack.yaml", stream=True)
    h, w, _ = frame.shape

    for r in results:
        if r.boxes is None or r.boxes.id is None: continue

        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        ids = r.boxes.id.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()

        for box, track_id, conf in zip(boxes, ids, confs):
            x1, y1, x2, y2 = box
            dist_val = max(0, (1.0 - (y2 / h)) * 8.0)
            label = f"ID:{track_id} CF:{conf:.2f} D:{dist_val:.1f}m"

            if (h * 0.45) < y2 < (h * 0.70):
                color = (0, 0, 255)  # Red
                if track_id not in logged_potholes:
                    winsound.Beep(1000, 300)

                    now = datetime.now()
                    timestamp_file = now.strftime("%Y%m%d_%H%M%S")

                    # --- THE FIX: ID FIRST WITH PADDING ---
                    # Using :04d makes it ID_0001, ID_0043 so sorting works perfectly
                    img_name = f"ID_{track_id:04d}_{timestamp_file}.jpg"
                    img_path = os.path.join(LOG_DIR, img_name)

                    cv2.imwrite(img_path, frame)

                    with open(log_file, mode='a', newline='') as f:
                        csv.writer(f).writerow(
                            [now.strftime("%H:%M:%S"), track_id, f"{conf:.2f}", f"{dist_val:.1f}m", img_name])

                    logged_potholes.add(track_id)
                    print(
                        f"{now.strftime('%H:%M:%S'):<10} | {track_id:<5} | {conf:.2f} | {dist_val:.1f}m | 🚨 LOGGED: {img_name}")
            else:
                color = (0, 255, 255)  # Yellow

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("ADAS Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()