from io import BytesIO

from flask import Flask, request, jsonify
import torch
from PIL import Image
from pathlib import Path
from yolov5 import detect

app = Flask(__name__)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', trust_repo=True, force_reload=True)

@app.route('/upload', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    # 파일 데이터를 바이너리로 읽고 BytesIO 객체로 변환
    image_file = request.files['image']
    image_bytes = image_file.read()
    img = Image.open(BytesIO(image_bytes))

    results = model(img)
    class_names = extract_class(results)

    return jsonify({"names": class_names})


def extract_class(results):
    class_ids = results.pred[0][:, -1].cpu().numpy()

    class_names = [results.names[int(class_id)] for class_id in class_ids]

    food_classes = ['rice_ball', 'kimbap', 'ramyeon', 'black_noodle', 'topokki', 'fried_chicken']

    sorted_class_name = []
    for class_name in class_names:
        if class_name in food_classes:
            sorted_class_name.append(class_name)

    return sorted_class_name



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6000)