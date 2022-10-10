import torch
from dataloaders.dataloader1 import *
from networks.AlexNet import AlexNet
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json


# 预处理
data_transforms = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

# load image

img = Image.open('../data/test/test/1223.jpg')
plt.imshow(img)
# [N, C, H, W]
img = data_transforms(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)

# read class_indict
try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# create model
model = AlexNet()
# load model weights
model_weight_path = "../model/AlexNet_128_0.002_100.pth"
model.load_state_dict(torch.load(model_weight_path))

# 关闭 Dropout
model.eval()
with torch.no_grad():
    # predict class
    output = torch.squeeze(model(img))     # 将输出压缩，即压缩掉 batch 这个维度
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()
print(class_indict[str(predict_cla)], predict[predict_cla].item())
plt.show()
