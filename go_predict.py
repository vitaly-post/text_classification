from prediction import Classify
from classification_config import conf_pt as default_conf

classifying_pytorch_model = Classify(default_conf)

class_pred = classifying_pytorch_model.predict("Кабулов прокомментировал возможность признания движения Талибан")

print(f'С вероятностью {class_pred[1]}% это "{class_pred[0]}"')