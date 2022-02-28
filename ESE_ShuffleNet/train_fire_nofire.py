# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 14:46:58 2019

@author: pc
"""
import time
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR

from SEshufflenet import seshufflenet

# 参数 
BARCH_SIZE = 32
LR = 0.001
EPOCH = 100

# 指定gpu使用
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('USE GPU', device)
else:
    print('USE CPU') 


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def get_mean_std(dataset, ratio=1):
    """Get mean and std by sample ratio
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(len(dataset)*ratio), shuffle=False, num_workers=0)
    train = iter(dataloader).next()[0]   # 一个batch的数据
    mean = np.mean(train.numpy(), axis=(0,2,3))
    std = np.std(train.numpy(), axis=(0,2,3))
    return mean, std
    

trans_pre = transforms.Compose([ transforms.Resize(size=(224, 224)), transforms.ToTensor() ])
train_pre = torchvision.datasets.ImageFolder(root='./data/train', transform=trans_pre) # 根据文件夹分类
test_pre =torchvision.datasets.ImageFolder(root='./data/test', transform=trans_pre)

#train_mean, train_std = get_mean_std(train_pre)
#test_mean, test_std = get_mean_std(test_pre)

train_mean = [0.421, 0.409, 0.407]
train_std =  [0.269, 0.241, 0.270]
test_mean =  [0.492, 0.441, 0.400]
test_std =   [0.277, 0.249, 0.273]

print(train_mean)
print(train_std)
print(test_mean)
print(test_std)

# 数据处理
transform_train = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(), 
    transforms.Normalize(mean=train_mean, std=train_std)
    ])
transform_test = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(), 
    transforms.Normalize(mean=test_mean, std=test_std)
    ]) 

# 读取数据 
train_dataset = torchvision.datasets.ImageFolder(root='./data/train', transform=transform_train) # 根据文件夹分类
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BARCH_SIZE, shuffle=True) 
test_dataset =torchvision.datasets.ImageFolder(root='./data/test', transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
print(int(len(train_dataset)))
print(int(len(test_dataset)))
print(train_dataset.classes)
print(train_dataset.class_to_idx)

# 开始训练 

Net = seshufflenet().to(device)

criterion = nn.CrossEntropyLoss().to(device)
opti = torch.optim.Adam(Net.parameters(), lr=LR)
scheduler = MultiStepLR(opti, milestones=[50, 80], gamma=0.2)
 
# 主函数
if __name__=='__main__':
    
    print(get_parameter_number(Net))
    
    Acc_list_train = []
    Acc_list_test = []
    Loss_list = []
    
    SE_list_test = []
    SP_list_test = []
    AC_list_test = []
    PRE_list_test = []
    REC_list_test = []
    FM_list_test = []
    
    best_test_acc = 0
    total_time = 0
    
    Net.train()
    
    for epoch in range(EPOCH):
        print('Epoch %d'  % (epoch + 1))
        start_time = time.time()
        sum_loss = 0.0
        iter_loss = 0.0
        correct1 = 0
        total1 = 0
        correct2 = 0
        total2 = 0
        TP = 0
        TN = 0
        FN = 0
        FP = 0
        for i, (images,labels) in enumerate(train_loader):

            images = Variable(images.to(device))
            labels = Variable(labels.to(device))
 
            opti.zero_grad()                      # 梯度清零，清空过往梯度
            out = Net(images)
            _, predicted = torch.max(out.data, 1) # 获取输出的预测类别
 
            total1 += labels.size(0)
            correct1 += (predicted == labels).sum().item()
            
            loss = criterion(out, labels)                  # 计算交叉熵
            iter_loss += loss.item()        
            sum_loss += loss.item()*labels.size(0)         # 总损失函数
            
            loss.backward()   # 反向传播，计算当前梯度
            opti.step()       # 根据梯度更新网络参数
            
            if i % 5 == 4:
                print('[%d, %d] loss: %.03f' % (epoch+1, i+1, iter_loss))
                iter_loss = 0.0
                
        train_acc = 100.0*correct1/total1
        
        total_time += time.time() - start_time
        
        scheduler.step()
        
        Net.eval()
        for j, (images_test, labels_test) in enumerate(test_loader):
            
            images_test = Variable(images_test.to(device))
            labels_test = Variable(labels_test.to(device))
            out_test = Net(images_test)
            _, pre_test = torch.max(out_test.data, 1)     # 行最大值的索引
            total2 += labels_test.size(0)
            
            if   pre_test==0 and labels_test==0:
                TP += 1
            elif pre_test==1 and labels_test==0:
                FN += 1
            elif pre_test==1 and labels_test==1:   
                TN += 1
            elif pre_test==0 and labels_test==1:   
                FP += 1
                
            correct2 += (pre_test == labels_test).sum().item()
            
        test_acc = 100.0*correct2/total2
        
        if test_acc > best_test_acc:
            torch.save(Net, './ESE.pth')
            best_test_acc = test_acc
            print('Best_test_accurary={}'.format(best_test_acc))  
        else:
            print('Best_test_accurary={}'.format(best_test_acc)) 
        
        Net.train()         
        
        SE  = TP/(TP+FN+1e-20)
        SP  = TN/(TN+FP+1e-20)
        AC  = (TP+TN)/(TP+TN+FP+FN+1e-20)
        PRE = TP/(TP+FP+1e-20)
        REC = TP/(TP+FN+1e-20)
        FM  = (2*PRE*REC)/(PRE+REC)
        print(' ')
        SE_list_test.append(SE)
        print('Test_SE={}'.format(SE))
        SP_list_test.append(SP)
        print('Test_SP={}'.format(SP))
        AC_list_test.append(AC)
        print('Test_ACC={}'.format(AC))
        PRE_list_test.append(PRE)
        print('Test_Precision={}'.format(PRE))
        REC_list_test.append(REC)
        print('Test_Recall={}'.format(REC))
        FM_list_test.append(FM)
        print('Test_F-measure={}'.format(FM))
        print(' ')
        Acc_list_train.append(train_acc)
        print('Train_accurary={}'.format(train_acc))
        Acc_list_test.append(test_acc)
        print('Test_accurary={}'.format(test_acc))
        Loss_list.append(sum_loss)
        print('Epoch_loss={}'.format(sum_loss))
    
    print('Total_time={}'.format(total_time))
    
#    print('Loss_list={}'.format(Loss_list))
#    print('Acc_list={}'.format(Acc_list_test))
    print()
    print('Best_Train_accurary={}'.format(max(Acc_list_train)))
    print('Best_Test_accurary={}'.format(max(Acc_list_test)))
    print()
    print('Best_Test_SE={}'.format(max(SE_list_test)))
    print('Best_Test_SP={}'.format(max(SP_list_test)))
    print('Best_Test_ACC={}'.format(max(AC_list_test)))
    print('Best_Test_Precision={}'.format(max(PRE_list_test)))
    print('Best_Test_Recall={}'.format(max(REC_list_test)))
    print('Best_Test_F-measure={}'.format(max(FM_list_test)))
    
    x1 = range(0, EPOCH)
    x2 = range(0, EPOCH)
    x3 = range(0, EPOCH)
    y1 = Acc_list_train
    y2 = Acc_list_test
    y3 = Loss_list
    
    plt.subplot(3, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('Train accuracy vs. epoches')
    plt.ylabel('Train accuracy')
    
    plt.subplot(3, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('Test accuracy vs. epoches')
    plt.ylabel('Test accuracy')
    
    plt.subplot(3, 1, 3)
    plt.plot(x3, y3, '.-')
    plt.xlabel('Train loss vs. epoches')
    plt.ylabel('Train loss')
    
    plt.savefig("Accuracy_Epoch" + (str)(EPOCH) + ".png")
    plt.show()



