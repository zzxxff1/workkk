import numpy as np
import paddle
from PIL import Image
from resnet import ResNet

net = ResNet(2)
params_path = 'palm.pdparams'
state_dict = paddle.load(params_path)
net.set_state_dict(state_dict)

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 127.5 - 1.
    image = np.transpose(image, (2, 0, 1))
    return image

def predict(file_path):
    img = Image.open(file_path)
    #img2 = Image.open(file2_path)

    img_processed = preprocess_image(img)
    #img2_processed = preprocess_image(img2)

    img_tensor = paddle.to_tensor(img_processed, dtype='float32')
    #img2_tensor = paddle.to_tensor(img2_processed, dtype='float32')
    img_tensor = paddle.unsqueeze(img_tensor, axis=0)
    #img2_tensor = paddle.unsqueeze(img2_tensor, axis=0)

    pred = net(img_tensor)
    #pred2 = net(img2_tensor)

    prediction = '病变' if pred[0][1] > pred[0][0] else '正常'
    #prediction2 = '病变' if pred2[0][1] > pred2[0][0] else '正常'
    
    return prediction
