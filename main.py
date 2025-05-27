from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
import shutil, os, requests

# مسیر مدل و لینک دانلود Drive (لینک تو)
MODEL_PATH = "model.pt"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1EpxYIq6HgMkuuu5ac0CNQQ4-2YUb-d2A"

# دانلود مدل اگر وجود نداشت
if not os.path.exists(MODEL_PATH):
    print("در حال دانلود مدل YOLO...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    print("مدل با موفقیت ذخیره شد ✅")

# لیست کلاس‌ها
CLASSES = [
    "Angry", "Calm", "Disgust", "Excited", "Fear", "Happy", "Sad", "Serious",
    "Sleepy", "Sly", "Sorry", "Surprised", "Thinking", "Worried", "Neutral"
]

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):
    input_path = "static/input.jpg"
    output_path = "static/output.jpg"

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    model = YOLO(MODEL_PATH)
    results = model(input_path)
    result = results[0]
    result.save(filename=output_path)

    labels = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        labels.append(CLASSES[cls_id])

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result_image": "/" + output_path,
        "labels": labels
    })
