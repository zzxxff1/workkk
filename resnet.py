#!/usr/bin/env python
# coding: utf-8

# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 


# In[2]:


import os 
import numpy as np 
import paddle
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image


# In[3]:


plt.figure(figsize=(10,6))
for i, t in enumerate('PHN'):
    x = Image.open(f'PALM-Training400/{t}0001.jpg')
    x = np.array( x.resize((200,200)) )
    plt.subplot(1,3,i+1)
    plt.axis('off')
    plt.imshow(x)


# In[4]:


x_p, x_np = [], []
for i in range(1, 400):
    try:
        x = Image.open('PALM-Training400/H%04d.jpg'%i)
        x = np.array( x.resize((224,224)) )
        x_np.append(x)
    except:
        break 
for i in range(1, 400):
    try:
        x = Image.open('PALM-Training400/P%04d.jpg'%i)
        x = np.array( x.resize((224,224)) )
        x_p.append(x)
    except:
        break 
for i in range(1, 400):
    try:
        x = Image.open('PALM-Training400/N%04d.jpg'%i)
        x = np.array( x.resize((224,224)) )
        x_np.append(x)
    except:
        break 
        
x_p = np.array(x_p)
x_np = np.array(x_np)
print(x_p.shape, x_np.shape)


# In[5]:


if x_p.dtype == 'uint8':
    x_p = x_p / 127.5 - 1.
    x_np = x_np / 127.5 - 1.


# In[6]:


# H: 26,  P: 213,  N:161,   Total: 400
def sample(n): # return shape (2n, 3, 224, 224)
    s_p = x_p[np.random.randint(1, x_p.shape[0], size=(n,))]
    s_np = x_np[np.random.randint(1, x_np.shape[0], size=(n,))]
    return np.transpose(np.concatenate((s_p, s_np),axis=0), (0,3,1,2))
print(sample(5).shape)


# ### References
# 
# * [ResNet Paper](https://arxiv.org/abs/1512.03385)
# 
# * [AIStudio Resnet](https://aistudio.baidu.com/aistudio/projectdetail/3513506)
# 

# In[8]:


class ConvBNLayer(paddle.nn.Layer):
    # convolution layer with batch normalization and relu
    def __init__(self, num_channels, num_filters, filter_size, stride=1):
        super().__init__()

        self._conv = paddle.nn.Conv2D(num_channels, num_filters, filter_size, 
                                        stride=stride, padding=(filter_size-1)//2, bias_attr=False)
        self._bn = paddle.nn.BatchNorm2D(num_filters)
        self._act = paddle.nn.ReLU()

    def forward(self, x):
        return self._act( self._bn( self._conv(x) ) )

class ResBlock(paddle.nn.Layer):
    # residual layer in ResNet50, also called bottleneck
    def __init__(self, num_channels, num_filters, stride=1):
        super().__init__()

        self._conv1 = ConvBNLayer(num_channels,  num_filters, 1, 1)
        self._conv2 = ConvBNLayer(num_filters,   num_filters, 3, stride)
        self._conv3 = ConvBNLayer(num_filters, num_filters*4, 1, 1)

        if num_channels != (num_filters << 2):
            # a conv layer that match the input to the filters
            self.short = ConvBNLayer(num_channels,  num_filters*4, 1, stride)
        else:
            self.short = None

        self._act = paddle.nn.ReLU()
            
    def forward(self, x):
        y = self._conv3(self._conv2(self._conv1(x)))
        if self.short is not None:
            # a conv layer that match the input to the filters
            x = self.short(x)
        y += x
        return self._act(y)


# In[9]:


class ResNet(paddle.nn.Layer):
    # ResNet50
    def __init__(self, output):
        super().__init__()
        self._conv1 = ConvBNLayer(3, 64, 7, 2)
        self._pool1 = paddle.nn.MaxPool2D(kernel_size=3,stride=2,padding=1)
        self.blocklist = []
        for part, channels in zip([3,4,6,3],[64,128,256,512]):
            # upzoom the channels
            self.blocklist.append( ResBlock(num_channels = max(64,channels//2), 
                                            num_filters = channels//4,
                                            stride = 2 if channels != 64 else 1 ) )
            # followed with several normal residual blocks
            for i in range(1,part):
                self.blocklist.append( ResBlock(num_channels = channels, 
                                                num_filters  = channels//4,
                                                stride = 1) )

        self._pool2 = paddle.nn.AdaptiveAvgPool2D(output_size=1)

        self._dense = paddle.nn.Linear(512, output)
        self._sig = paddle.nn.Sigmoid()

    def forward(self,x):
        x = self._conv1(x)
        x = self._pool1(x)
        for i , block in enumerate(self.blocklist):
            x = block(x)
        x = self._pool2(x)
        x = paddle.reshape(x, (x.shape[0], 512))
        x = self._dense(x)
        x = self._sig(x)
        return x

net = ResNet(2)
#opt = paddle.optimizer.Momentum(parameters = net.parameters(), learning_rate = 1e-3, momentum = 0.9,
#                                weight_decay = 1e-3)
opt = paddle.optimizer.Adam(parameters=net.parameters())
losses = []
paddle.save(net.state_dict(), 'palm.pdparams')


# In[10]:


# input_data = paddle.to_tensor(sample(4), dtype='float32')
# output = net(input_data)
# print(output)


# In[124]:


#take only 5 minutes or so on GPU
# epochs = 1000
# batch_size = 32
# # P label: 1, H/N label: 0
# label = paddle.to_tensor([1]*batch_size+[0]*batch_size ,dtype='int64')
# for i in tqdm(range(1+len(losses),1+len(losses)+epochs)):
#     data = paddle.to_tensor(sample(batch_size), dtype='float32')
#     pred = net(data)
#     loss = paddle.nn.CrossEntropyLoss()(pred,label)
#     opt.clear_grad()
#     loss.backward() 
#     opt.step()
#     losses.append(loss.item())


# In[125]:


# print("预测结果：", pred)


# In[131]:


plt.figure(figsize=(11,6))
plt.plot(losses)


# In[72]:


# download the validation file and copy the labels 
# P: 1,   H/N: 0
# validlabel = '0110000100111101101011100111110011110001001101111001101000000110111100100110111101101100011011101111010110111110011010011000000001110011010110101001010010001101111001000101100101111101000010100111010010111000101110010111110001000100110010010110011101011110001010010011011010101011101111110000110101110111100100100111111101010001101011000100111000100010100101101101100001100111000010011111110000010010'
# print(len(validlabel))


# In[127]:
# acc = 0
# valid_preds = []
# # split into 400 = 20 * 20
# for i in tqdm(range(20)):
#     start = i * 20 + 1
#     valid_xs = []
#     for j in range(start, start+20):        
#         x = Image.open('PALM-Validation400/V%04d.jpg' % j)
#         x = np.array(x.resize((224, 224)))
#         valid_xs.append(x)
#     valid_xs = np.array(valid_xs) / 127.5 - 1.
#     valid_xs = np.transpose(valid_xs, (0, 3, 1, 2))

#     valid_pred = net(paddle.to_tensor(valid_xs, dtype='float32'))
   
#     valid_preds.append(valid_pred[:, 1].numpy())

#     for j in range(20):
#         img_num = start + j
#         prediction = '病变' if valid_pred[j][1] > valid_pred[j][0] else '正常'
#         actual = '病变' if int(validlabel[start+j-1]) else '正常'
#         print(f"第 {img_num} 张图片预测结果：{prediction}, 真实标签：{actual}, 预测值：{valid_pred[j]}")
#         if (float(valid_pred[j][1] - valid_pred[j][0]) < 0) ^ int(validlabel[start+j-1]):
#             acc += 1

# accuracy = acc / 400
# print(f'Acc = {acc}/400 = {accuracy * 100}%')


# valid


# %%
