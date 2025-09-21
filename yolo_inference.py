from ultralytics import YOLO

# Test basic player detection using yolov8
model = YOLO("models/best.pt")

results = model.predict(source="input/videoplayback.mp4", save=True)

print(results[0])
print("=========================")
for box in results[0].boxes:
    print(box)
