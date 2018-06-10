import os

BASE_PATH = '/'.join(os.getcwd().split('/')[:-1])

input_tensor_names = ['img_inputs:0', 'dropout_rate:0']
output_tensor_names = ['resnet_v1_50/E_BN2/Identity:0']
model_fp = '{}/pretrained/insightface.pb'.format(BASE_PATH)
test_img_fp = '{}/tests/1.png'.format(BASE_PATH)
device = '/cpu:0'
