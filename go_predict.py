from prediction import Classify
from classification_config import conf_pt as default_conf

classifying_pytorch_model = Classify(default_conf)

while True:
    user_input = input(">>> ")
    class_pred = classifying_pytorch_model.predict(user_input)


    print(f'С вероятностью {class_pred[1]}% это "{class_pred[0]}"')