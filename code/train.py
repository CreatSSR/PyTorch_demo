import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import os
import json
import time
import sys
sys.path.append(r'./dataloaders/')
from dataloaders.dataloader1 import *
from networks.AlexNet import AlexNet
from torch.utils.tensorboard import SummaryWriter



train_data_dir = '../data/train'
val_data_dir = '../data/train'


device = torch.device("cuda"if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
LR = 0.0002
Epoch = 100
loss_function = nn.CrossEntropyLoss()


Loss_list = []
Accuracy_list = []
learning_rate = []


def train():
    net = AlexNet()
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr = LR, momentum=1)

    best_acc = 0.0

    train_dataset = train_dataloader(train_data_dir, BATCH_SIZE)
    val_dataset = val_dataloader(val_data_dir, BATCH_SIZE)
    val_num = len(val_dataset)

    for epoch in range(Epoch):
        #writer = SummaryWriter('./logs')
        #开启dropout
        net.train()

        running_loss = 0.0
        acc = 0.0

        time_start = time.perf_counter()

        for step, data in enumerate(train_dataset, start = 0):
            img, label = data
            #writer.add_images("train_image", img, global_step=None, walltime=None, dataformats='NCHW')
            #writer.add_scalar("train_label", label, global_step=None, walltime=None)
            optimizer.zero_grad()

            outputs = net(img.to(device))
            loss = loss_function(outputs, label.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # 打印训练进度（使训练过程可视化）
            rate = (step + 1) / len(train_dataset)  # 当前进度 = 当前step / 训练一轮epoch所需总step
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")

        print()
        print('%f s' % (time.perf_counter() - time_start))

        Lr = optimizer.state_dict()['param_groups'][0]['lr']

        net.eval()  # 验证过程中关闭 Dropout

        with torch.no_grad():
            for val_data in val_dataset:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1] # 以output中值最大位置对应的索引（标签）作为预测输出

                #writer.add_images("val_image", img, global_step=None, walltime=None, dataformats='NCHW')
                #writer.add_scalar("val_label", label, global_step=None, walltime=None)
                #writer.add_scalar("predict_label", predict_y, global_step=None, walltime=None)

                acc += (predict_y == val_labels.to(device)).sum().item()
            accuracy = acc / val_num

                #acc += (predict_y == val_labels.to(device)).sum().item()
                #val_accurate = acc / val_num

            # 保存准确率最高的那次网络参数
            save_path = '../model/AlexNet_128_0.002_100.pth'
            if accuracy > best_acc:
                best_acc = accuracy
                torch.save(net.state_dict(), save_path)

            print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f  lr: %3f \n' %
                  (epoch + 1, running_loss / step, accuracy, Lr))

            Loss_list.append(running_loss / step)
            Accuracy_list.append(accuracy)
            learning_rate.append(Lr)

        #writer.add_scalar("loss", loss, epoch)
        #writer.add_scalar("acc", accuracy, epoch)

    #writer.close()

    print('Finished Training')

if __name__=="__main__":
    train()

    #print(Loss_list, Accuracy_list)

    x1 = range(0, Epoch)
    x2 = range(0, Epoch)
    x3 = range(0, Epoch)
    y1 = Accuracy_list
    y2 = Loss_list
    y3 = learning_rate
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('Test accuracy vs. epoches')
    plt.ylabel('Test accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('Test loss vs. epoches')
    plt.ylabel('Test loss')
    plt.show()
    plt.savefig("accuracy_loss.jpg")





