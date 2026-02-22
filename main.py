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

# TRACKING STORAGE: Keeps track of IDs already logged to prevent duplicates
logged_potholes = set()

# 2. Setup Log File Header
if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Track_ID", "Image_Path", "Confidence"])

# 3. Load Model (YOLOv8/v11)
model = YOLO('best.pt')

# 4. Connect Phone/Camera Stream
# Change this IP to your current phone IP from IP Webcam
url = "http://192.168.68.105:8080/video"
cap = cv2.VideoCapture(url)

print("🚀 ADAS System Active with ByteTrack...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # --- BYTE TRACK ENABLED ---
    # persist=True keeps IDs across frames
    # tracker="bytetrack.yaml" uses the specific ByteTrack algorithm
    results = model.track(frame, conf=0.6, persist=True, tracker="bytetrack.yaml", stream=True)

    h, w, _ = frame.shape

    for r in results:
        # Check if any objects were actually tracked (have IDs)
        if r.boxes is None or r.boxes.id is None:
            continue

        # Extract data
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        ids = r.boxes.id.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()

        for box, track_id, conf in zip(boxes, ids, confs):
            x1, y1, x2, y2 = box

            # --- WARNING ZONE LOGIC ---
            # If the pothole bottom (y2) is in the middle section of the screen
            if (h * 0.45) < y2 < (h * 0.70):

                # Visual Alert (Red Box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, f"WARNING: ID {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # --- UNIQUE LOGGING LOGIC ---
                # Only log if this is a NEW ID we haven't recorded yet
                if track_id not in logged_potholes:
                    winsound.Beep(1000, 300)  # Audible Alert

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_name = f"pothole_id{track_id}_{timestamp}.jpg"
                    img_path = os.path.join(LOG_DIR, img_name)

                    # Save visual proof
                    cv2.imwrite(img_path, frame)

                    # Write to CSV
                    with open(log_file, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([timestamp, track_id, img_path, f"{conf:.2f}"])

                    # Remember this ID so we don't log it again in the next frame
                    logged_potholes.add(track_id)
                    print(f"✅ Logged New Pothole: ID {track_id}")

            else:
                # Normal Detection (Yellow Box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Display the ADAS Feed
    cv2.imshow("ADAS Detection & ByteTrack Logger", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"🏁 Session Ended. Total unique potholes detected: {len(logged_potholes)}")