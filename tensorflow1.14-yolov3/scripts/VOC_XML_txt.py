import os
import random

trainval_percent = 1  # 训练验证集占整个数据集的比例，根据需要自己修改
train_percent = 0.9  # 训练集占训练验证集的比例，根据需要自己修改
xmlfilepath = '/raid/my_data/Annotations/'
txtsavepath = '/raid/my_data/ImageSets/Main/'
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrainval = open(txtsavepath + 'trainval.txt', 'w')
ftest = open(txtsavepath + 'test.txt', 'w')
ftrain = open(txtsavepath + 'train.txt', 'w')
fval = open(txtsavepath + 'val.txt', 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()