# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 15:15:26 2020

@author: 99147
"""
import time
import torch
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms

# 指定gpu使用
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('USE GPU', device)
else:
    print('USE CPU') 

test_mean =  [0.492, 0.441, 0.400]
test_std =   [0.277, 0.249, 0.273]

print(test_mean)
print(test_std)

# 数据处理
transform_test = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(), 
    transforms.Normalize(mean=test_mean, std=test_std)
    ]) 

# 读取数据 
test_dataset =torchvision.datasets.ImageFolder(root='./data/test', transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
print(int(len(test_dataset)))
print(test_dataset.classes)
print(test_dataset.class_to_idx)

Net = torch.load('./ESE.pth')

# 主函数
if __name__=='__main__':

    total_time = 0
    
    TP = 0
    TN = 0
    FN = 0
    FP = 0    
    
    Net.eval()
    start_time = time.time()
    for j, (images_test, labels_test) in enumerate(test_loader):

        images_test = Variable(images_test.to(device))
        labels_test = Variable(labels_test.to(device))
        out_test = Net(images_test)
                 
        _, pre_test = torch.max(out_test.data, 1) # 行最大值的索引
        if   pre_test==0 and labels_test==0:
            TP += 1
        elif pre_test==1 and labels_test==0:
            FN += 1
        elif pre_test==1 and labels_test==1:   
            TN += 1
        elif pre_test==0 and labels_test==1:   
            FP += 1 
           
    total_time = time.time() - start_time   
    print(total_time)
    
    SE  = TP/(TP+FN)
    SP  = TN/(TN+FP)
    AC  = (TP+TN)/(TP+TN+FP+FN)
    PRE = TP/(TP+FP)
    REC = TP/(TP+FN)
    FM  = (2*PRE*REC)/(PRE+REC)
        
    print('Test_SE={}'.format(SE))
    print('Test_SP={}'.format(SP))
    print('Test_ACC={}'.format(AC))
    print('Test_Precision={}'.format(PRE))
    print('Test_Recall={}'.format(REC))
    print('Test_F-measure={}'.format(FM))
    
  
    

