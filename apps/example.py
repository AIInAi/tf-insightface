import cv2
import numpy as np
from models import base_server
from configs import configs

# Read example image
test_img = cv2.imread(configs.test_img_fp)
test_img = cv2.resize(test_img, (112, 112))

# Define input tensors feed to session graph
dropout_rate = 0.5
input_data = [np.expand_dims(test_img, axis=0), dropout_rate]

# Define a Base Server
srv = base_server.BaseServer(model_fp=configs.model_fp,
                             input_tensor_names=configs.input_tensor_names,
                             output_tensor_names=configs.output_tensor_names,
                             device=configs.device)
# Run prediction
prediction = srv.inference(data=input_data)

# Print results
print('512-D Features are \n{}'.format(prediction))
