# Face Recognition with OpenCV (Android IP Webcam)

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red.svg)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-orange.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A complete **face recognition system built from scratch using OpenCV**, leveraging an **Android phone as an IP webcam**.
This project demonstrates the full **computer vision pipeline**: dataset creation, model training, and real-time recognition.

---

# 🎯 Project Overview

This repository implements a **real-time face recognition pipeline** using the **LBPH (Local Binary Patterns Histogram)** algorithm.

The system works in **three main stages**:

| Stage                     | Description                                          |
| ------------------------- | ---------------------------------------------------- |
| **Dataset Creation**      | Capture face images using an Android IP webcam       |
| **Model Training**        | Train an LBPH face recognizer using collected images |
| **Real-Time Recognition** | Detect and identify faces from a live video stream   |

**Recognition Pipeline**

```
Camera Stream → Face Detection → Feature Extraction → LBPH Classifier → Identity Prediction
```

---

# 💼 Why This Matters

| Skill Demonstrated              | Business Value                                                   |
| ------------------------------- | ---------------------------------------------------------------- |
| **Computer Vision Engineering** | Builds complete vision pipelines using OpenCV                    |
| **Algorithm Understanding**     | Implements face recognition using LBPH instead of black-box APIs |
| **Data Pipeline Design**        | Handles dataset collection, preprocessing, and model training    |
| **Real-Time Systems**           | Processes live camera streams efficiently                        |
| **Edge AI Development**         | Uses lightweight algorithms suitable for embedded devices        |

---

# 🚀 Technical Highlights

### Core Computer Vision Stack

| Component          | Technology                 |
| ------------------ | -------------------------- |
| Face Detection     | Haar Cascade Classifier    |
| Feature Extraction | Local Binary Pattern (LBP) |
| Face Recognition   | LBPH Classifier            |
| Video Source       | Android IP Webcam          |
| Image Processing   | OpenCV + NumPy             |

---

# 🧠 Algorithm Explanation

The system uses **LBPH (Local Binary Pattern Histogram)** for face recognition.

### Local Binary Pattern

LBP encodes texture by comparing each pixel with its neighbors.

```
LBP(x, y) = Σ s(gi - gc) * 2^i
```

Where:

| Symbol | Meaning            |
| ------ | ------------------ |
| `gc`   | Center pixel       |
| `gi`   | Neighbor pixel     |
| `s(x)` | Threshold function |

LBPH builds histograms of these patterns and compares them across faces.

---

# 📂 Project Structure

```
face-recognition/
│
├── face_dataset.py
├── training.py
├── face_recognition.py
│
├── dataset/
│   └── User.ID.Sample.jpg
│
├── trainer/
│   └── trainer.yml
│
└── haarcascade_frontalface_default.xml
```

| File                  | Purpose                         |
| --------------------- | ------------------------------- |
| `face_dataset.py`     | Collect face images from camera |
| `training.py`         | Train LBPH recognition model    |
| `face_recognition.py` | Run real-time recognition       |
| `dataset/`            | Stored training images          |
| `trainer/`            | Saved trained model             |

---

# ⚙️ System Workflow

```
Start Android IP Webcam
        ↓
Collect Face Dataset
        ↓
Train Recognition Model
        ↓
Run Real-Time Face Recognition
```

---

# 📸 Dataset Collection

Run:

```
python face_dataset.py
```

The program will:

1. Connect to your **phone IP webcam**
2. Detect faces
3. Capture **30 samples per user**
4. Save images to the dataset folder

File naming format:

```
User.<ID>.<SampleNumber>.jpg
```

Example:

```
User.1.1.jpg
User.1.2.jpg
User.1.3.jpg
```

---

# 🧠 Model Training

Run:

```
python training.py
```

The script will:

1. Load images from `dataset/`
2. Extract face IDs from filenames
3. Detect faces
4. Train an **LBPH recognizer**
5. Save the trained model

Output:

```
trainer/trainer.yml
```

---

# 🎥 Real-Time Face Recognition

Run:

```
python face_recognition.py
```

The program will:

* Start the camera stream
* Detect faces
* Predict identities
* Display results on screen

Output example:

```
User1   89%
User2   92%
Unknown
```

---

# 📊 Recognition Output

| Output     | Meaning                      |
| ---------- | ---------------------------- |
| Name       | Predicted identity           |
| Confidence | Recognition confidence score |

Lower LBPH distance = **better match**.

---

# 📦 Installation

Clone the repository:

```
git clone https://github.com/N1N0u/facerecon.git
cd facerecon
```

Install dependencies:

```
pip install opencv-python numpy pillow
```

Download Haar Cascade:

```
https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
```

---

# 📱 Android Camera Setup

Install **IP Webcam** on your phone.

Start the server and obtain the stream URL:

```
http://PHONE_IP:8080/video
```

Update the script:

```
cv2.VideoCapture("http://PHONE_IP:8080/video")
```

---

# 🧪 Example Use Cases

This system can be adapted for:

| Application        | Description                        |
| ------------------ | ---------------------------------- |
| Smart Door Access  | Identify authorized users          |
| Attendance Systems | Detect registered employees        |
| Security Cameras   | Monitor known individuals          |
| AI Experiments     | Learn computer vision fundamentals |
| Smart Home         | Recognize household members        |

---

# 🔬 Learning Outcomes

This project demonstrates knowledge of:

* Computer Vision
* Image Processing
* Feature Extraction
* Machine Learning (LBPH)
* Real-Time Video Processing
* OpenCV Architecture

---

# 🚧 Possible Improvements

Future enhancements could include:

* Deep learning face recognition (**FaceNet / ArcFace**)
* GPU acceleration
* Multi-camera support
* REST API for remote recognition
* Web dashboard for monitoring
* Face database management

---

# 📝 License

MIT License — Free for educational and commercial use.

---

# 👨‍💻 Author

**N1N0u**
