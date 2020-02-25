#模型微调-热狗识别
#微调是在迁移技术中常用的一种技术
#主要用于将已经训练好的模型（针对源数据集）迁移到目标数据集上
#认为源数据集上的知识对于目标数据集有用，例如可以识别物体的边缘，纹理，形状和物体组成等的知识
'''
模型微调步骤：
1.获取源模型
2.创建目标模型，复制源模型除了输出层以外的所有参数，这里假设了源数据集上的知识在目标数据集上仍有效，且输出层与标签密切相关，故不予采用
3.定义，初始化输出层
4.训练输出层，由于输出层是未经训练的，所以学习率设置较大，其他层的参数由于已经带有一定的知识，所以需要调整的参数数值较小，故学习率设置为一个较小的非零值
'''
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models
import os
import sys

sys.path.append("/home/kesci/input/")
import d2lzh1981 as d2l

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#获取数据集
os.listdir('/home/kesci/input/resnet185352')
data_dir = '/home/kesci/input/hotdog4014'
os.listdir(os.path.join(data_dir, "hotdog"))
train_imgs = ImageFolder(os.path.join(data_dir, 'hotdog/train'))
test_imgs = ImageFolder(os.path.join(data_dir, 'hotdog/test'))
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4);

#预处理-图像裁剪与数据标准化
#由于torchvision的models要求数据格式形状相同并且符合输入要求
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_augs = transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

test_augs = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        normalize
    ])

#源模型获取
pretrained_net = models.resnet18(pretrained=False)#pretrained设置为True可以自动下载并加载模型参数
pretrained_net.load_state_dict(torch.load('/home/kesci/input/resnet185352/resnet18-5c106cde.pth'))

pretrained_net.fc = nn.Linear(512, 2)

output_params = list(map(id, pretrained_net.fc.parameters()))
feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())

lr = 0.01
optimizer = optim.SGD([{'params': feature_params},
                       {'params': pretrained_net.fc.parameters(), 'lr': lr * 10}],
                       lr=lr, weight_decay=0.001)

#模型微调
def train_fine_tuning(net, optimizer, batch_size=128, num_epochs=5):
    train_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'hotdog/train'), transform=train_augs),
                            batch_size, shuffle=True)
    test_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'hotdog/test'), transform=test_augs),
                           batch_size)
    loss = torch.nn.CrossEntropyLoss()
    d2l.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)

train_fine_tuning(pretrained_net, optimizer)

scratch_net = models.resnet18(pretrained=False, num_classes=2)
lr = 0.1
optimizer = optim.SGD(scratch_net.parameters(), lr=lr, weight_decay=0.001)
train_fine_tuning(scratch_net, optimizer)