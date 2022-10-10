from dataset import myDataset


root = '.../.../data/train'
label_1 = 'calling_images'
label_2 = 'normal_images'
label_3 = 'smoking_images'

calling_dataset = myDataset(root, label_1)
normal_dataset = myDataset(root, label_2)
smoking_dataset = myDataset(root, label_3)


train_data = calling_dataset + normal_dataset + smoking_dataset
#image,label=train_data[index]



