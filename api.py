import io
from flask import Flask, jsonify, request
import torch
from PIL import Image
from torchvision import transforms
import os

device = "cpu"

class_names = ['0', '1', '2', '3', '4']
milk_list = {'0': '딸기우유', '1': '바나나우유', '2': '초코우유', '3': '커피우유', '4': '흰우유'}

transforms_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
model = torch.load('milk_epoch250_cpu.pt')

# 이미지를 읽어 결과를 반환하는 함수
def get_prediction(image_bytes):
    return '1'
    # image = Image.open(io.BytesIO(image_bytes))
    # image = transforms_test(image).unsqueeze(0).to(device)
    #
    # with torch.no_grad():
    #     outputs = model(image)
    #     _, preds = torch.max(outputs, 1)
    #     print(f'[예측 결과: {milk_list[class_names[preds[0]]]}]')

    # return class_names[preds[0]]


app = Flask(__name__)


@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        # 이미지 바이트 데이터 받아오기
        file = request.files['file']
        image_bytes = file.read()

        # 분류 결과 확인 및 클라이언트에게 결과 반환
        class_name = get_prediction(image_bytes=image_bytes)
        print("결과:", {'class_name': class_name})
        return jsonify({'class_name': class_name})

if __name__ == '__main__':

    # port = int(os.environ.get('PORT', 8000))

    # app.run(host='0.0.0.0', port=port, debug=True)
    app.run()