# 🛣️ Real-Time Pothole ADAS System

A real-time Advanced Driver Assistance System (ADAS) that detects and logs potholes using **YOLOv11** and **ByteTrack**. This version uses a smartphone as a wireless camera feed and a laptop for processing.

## 🚀 Key Features
- **Object Tracking:** Uses ByteTrack to assign unique IDs to potholes, ensuring each is logged only once.
- **Warning Zone:** Visual and audible alerts trigger when a hazard enters the critical driving path.
- **Automated Logging:** Saves image evidence and CSV data for infrastructure analysis.

## 🛠️ Hardware Setup
1. **Camera:** Android Phone running "IP Webcam" app.
2. **Processing:** Laptop with Python 3.10+ and an active network connection.

## 💻 Installation & Usage
1. Clone the repo: `git clone https://github.com/YOUR_USERNAME/Pothole-Detection-ADAS.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Update the `url` in `main.py` with your IP Webcam address.
4. Run: `python scripts/main.py`
