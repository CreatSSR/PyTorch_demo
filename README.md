# README

## Step 1

将normal,smoke,phone分别按比例（高斯随机划分  7：3）划分为训练集与验证集

batch_size = 32

epoch = 500

## Step 2

数据预处理：

data augmentating： MixUp 保持人物的不变形图像的等比填充缩放裁剪，水平翻转、由于数据清晰度差增加高斯模糊

crop： 224 * 224 * 3

normalize？

## Step 3

模型构建：

AlexNet

为防止overfitting 增加dropout

loss：交叉熵

优化器：SGD

## Step 4

模型训练：

learn_rate: 0.01



