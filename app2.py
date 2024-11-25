import base64
import cv2
import torch
from PIL import Image
from pathlib import Path

from ultralytics import YOLO

from io import BytesIO
from flask import Flask, request, jsonify


app = Flask(__name__)
model = YOLO('yolov8_1120.pt')

@app.route('/upload', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()
    img = Image.open(BytesIO(image_bytes))

    results = model(img)
    class_names = extract_class(results)
    filtered_images = draw_boxes(results)

    if len(filtered_images) > 0:
        annotated_frame = filtered_images[0]
        rendered_image = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

        buffered = BytesIO()
        rendered_image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        response = {
            "names": class_names,
            "image": image_base64
        }
        return jsonify(response)
    else:
        return jsonify({"error": "No desired classes detected."})


def draw_boxes(results):
    output_dir = 'results'
    desired_classes = [0, 1, 3, 4, 5, 6, 7, 9, 10]
    filtered_images = []

    # 결과 이미지 저장
    for i, result in enumerate(results):
        filtered_boxes = []
        annotated_frame = result.orig_img.copy()  # 원본 이미지 복사

        for box in result.boxes:
            class_id = int(box.cls[0])  # 클래스 ID

            if class_id in desired_classes:  # 원하는 클래스만 필터링
                filtered_boxes.append(box)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # 좌표
                confidence = box.conf[0].item()  # 신뢰도
                label = f"{model.names[class_id]} {confidence:.2f}"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        filtered_images.append(annotated_frame)

    return filtered_images


def extract_class(results):
    food_classes = ['rice_ball', 'kimbap', 'ramyeon', 'black_noodle', 'topokki', 'fried_chicken']

    filtered_boxes = []
    class_names = []

    for box in results[0].boxes.data:  # YOLOv8의 바운딩 박스 데이터
        x1, y1, x2, y2, confidence, class_id = box.tolist()
        class_name = model.names[int(class_id)]
        if class_name in food_classes:  # 원하는 클래스만 필터링
            filtered_boxes.append((x1, y1, x2, y2, confidence, class_name))
            class_names.append(class_name)

    return class_names



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6000)