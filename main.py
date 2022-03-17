import json
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, render_template

# 학습 모델 불러오기
loaded_model = load_model("./model/best_model_CNN.h5")

# 허용 확장자
ALLOWED_EXTENSIONS = set(["png"])

# Flask 앱 설정
app = Flask(__name__, template_folder="templates", static_folder="static")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1] in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    if request.method == "GET":
        return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if request.method == "POST":
        file = request.files["file"]
        if file and allowed_file(file.filename):
            # 파일을 저장해서 불러올 경우
            # filename = secure_filename(file.filename)
            # file.save("upload\\" + filename)
            # img = Image.open("upload\\" + filename)

            # 예측하기
            img = Image.open(file)
            img = img.convert('L')                  # grayscale로 변환
            img = img.resize((28, 28))              # 이미지 폭(width), 높이(height)를 28 x 28 로 변환
            img = np.array(img)                     # 이미지를 numpy 타입으로 변환
            img = img.reshape((28, 28, 1))          # numpy 배열을 1차원으로 변환
            test = np.array([img])
            test = 255 - test                       # 배경(흰색)과 글씨(검정)를 뒤집기 => 배경(검정),글씨(흰색)
            test = test / 255.0                     # 정규화(normalization)
            results = loaded_model.predict(test)    # 예측하기
            results_max = str(np.argmax(results[0]))
            results_encoded = json.dumps(results, cls=NumpyEncoder)
            return jsonify(results=results_encoded, predict=results_max)
        return jsonify()


if __name__ == "__main__":
    app.run(debug=True)
