from flask import Flask
from classification_config import conf_pt as default_conf
from classification_pt import ClassificationPt
from prediction import Classify
from flask import jsonify, request
import json

app = Flask(__name__)

classifying_pytorch_model = Classify(default_conf)

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/predict', methods=['POST'])
def predict():
    json_data_string = request.data.decode('utf-8')

    if len(json_data_string) == 0:
        return jsonify(result="Передана пустая строка")

    post_data_obj = json.loads(json_data_string, strict=False)

    message = post_data_obj.get('message')

    global classifying_pytorch_model

    class_pred = classifying_pytorch_model.predict(message)

    return jsonify(f'С вероятностью {class_pred[1]}% это "{class_pred[0]}"')


@app.route('/train', methods=['GET'])
def train():
    classification = ClassificationPt()

    classification.train()

    return jsonify("Train finished")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
