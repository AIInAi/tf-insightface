import cv2
import numpy as np
from models import base_server
from configs import configs

test_img = cv2.imread(configs.test_img_fp)
test_img = cv2.resize(test_img, (112, 112))
dropout_rate = 0.5
input_data = [np.expand_dims(test_img, axis=0), dropout_rate]

srv = base_server.BaseServer(model_fp=configs.model_fp,
                             input_tensor_names=configs.input_tensor_names,
                             output_tensor_names=configs.output_tensor_names,
                             device=configs.device)
prediction = srv.inference(data=input_data)