# 🎭 Emotion Detection API - FastAPI Based Face Emotion Recognizer

This project is an **AI-based facial emotion detection API** built with **FastAPI**, leveraging **ViT (Vision Transformer)** from Hugging Face for emotion classification and **MTCNN** from `facenet-pytorch` for face detection.

It follows a clean **OOP-based modular structure**, inspired by production-ready MLOps patterns — so it's scalable, testable, and easy to maintain.

---

## 📸 What it Does

- Upload any image (with one or more human faces)
- Detect faces and extract bounding boxes
- Predict emotion for each face using a transformer model
- Return:
  - face index
  - bounding box
  - emotion label
  - emotion ID
  - confidence score
  - matching emoji 😄😢😠

---

## 🧠 Behind the Scenes

- **Face Detection:** MTCNN from `facenet-pytorch`
- **Emotion Classification Model:** `trpakov/vit-face-expression` from Hugging Face
- **Framework:** FastAPI
- **Core Logic:** Modular OOP structure just like [BG_addAndRemove repo](https://github.com/Arogya-2002/BG_addAndRemove)

---

## 📁 Project Structure

```bash
.
├── app.py                          # FastAPI app entrypoint with routes
├── requirements.txt               # All Python dependencies
├── .gitignore                     # Ignoring env/cache/etc.
├── README.md                      # This file 💁‍♂️
└── src/
    ├── components/                # Core logic modules
    │   ├── detect_face.py         # Detect faces using MTCNN
    │   ├── classify_emotion.py    # Predict emotions with ViT
    │   └── process.py             # Full image processing pipeline
    ├── entity/                    # Data models
    │   ├── artifact.py            # Result structures (face index, label, etc.)
    │   └── config.py              # Config for models and constants
    ├── exceptions.py              # Custom error handling
    ├── logger.py                  # Logging setup
    └── utils.py                   # Utility functions like image loader
```

---

## 🔧 Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/emotion-detector-api.git
cd emotion-detector-api
```

### 2. Create Virtual Environment (Optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Start the FastAPI server

```bash
uvicorn app:app --reload
```

- Visit `http://127.0.0.1:8000` to check if it’s up.
- Docs available at `http://127.0.0.1:8000/docs`

---

## 📡 API Endpoint

### `/detect-emotions`

- **Method:** POST
- **Payload:** Image file (form field name = `file`)
- **Response:** List of detected faces and emotion predictions

#### 🧪 Sample Request via cURL

```bash
curl -X POST "http://127.0.0.1:8000/detect-emotions" \
  -F "file=@your_image.jpg"
```

#### ✅ Example Response

```json
{
  "faces": [
    {
      "face_index": 1,
      "box": [45, 50, 160, 170],
      "emotion_label": "happy",
      "emotion_id": 3,
      "confidence": 97.5,
      "emoji": "😄"
    }
  ]
}
```

---

## 📦 Models & Libs Used

| Component         | Tech                                |
|------------------|--------------------------------------|
| Face Detection    | `facenet-pytorch` (MTCNN)            |
| Emotion Detection | `trpakov/vit-face-expression` (ViT) |
| Web Framework     | `FastAPI`                           |
| Image Processing  | `Pillow`, `torch`, `numpy`          |

---

## 🧱 Next Steps / Ideas

- Add frontend (Streamlit, Gradio, or HTML form)
- Add emotion bar graph per face
- Save logs or outputs to file
- Dockerize the app
- Deploy on Hugging Face Spaces or Render

---

## 🙌 Credits

Built by [Your Name]  
OOP layout inspired by: [BG_addAndRemove GitHub Repo](https://github.com/Arogya-2002/BG_addAndRemove)

---

## 📃 License

MIT License — free to use and build on
