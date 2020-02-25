#批量归一化位置在全连接层中的仿射变换和激活函数之间
import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
import sys
sys.path.append("/home/kesci/input/") 
import d2lzh1981 as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#batch_norm计算方法定义
def batch_norm(is_training, X, gamma, beta, moving_mean, moving_var, eps, momentum):#eps为一个防止分母为零的极小量，在比值正常时几乎不影响计算
    if not is_training:
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)#在非训练状态下使用传入参数直接处理
    else:
        assert len(X.shape) in (2, 4)#通过传入参数表示归一化对象，如果不是合法格式则弹出报错
        if len(X.shape) == 2:#全连接
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:#卷积
            mean = X.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            var = ((X - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var#由该批数据更新传入参数
    Y = gamma * X_hat + beta  
    # gamma和beta是两个可学习参数，用于拉伸和偏移，批量归一化如果反而使模型效果变差则学习到gamma与beta的值使批量归一化无效化
    return Y, moving_mean, moving_var

#batch_norm模型定义
class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super(BatchNorm, self).__init__()#调用父类方法，这里是将batchnorm看作模型的一个子类，引入父类参数
        if num_dims == 2:
            shape = (1, num_features) #对全连接层所有数据做归一化
        else:
            shape = (1, num_features, 1, 1)  #对于卷积层可能各个通道下的分布本就均值不同，所以按照通道进行归一化比较合理
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)

    def forward(self, X):
        if self.moving_mean.device != X.device:#调整计算使用的硬件
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        Y, self.moving_mean, self.moving_var = batch_norm(self.training, 
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
            #这里eps被设为一个10^-5的数值，这是一个较为常见的设置数值，在torch中的
        return Y
    
#在lenet模型中使用位置与方法
net = nn.Sequential(
            nn.Conv2d(1, 6, 5), #例：用于计算的层
            BatchNorm(6, num_dims=4),#例：批量归一化层放于计算与激活函数中间
            nn.Sigmoid(),#例：激活函数
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5),
            BatchNorm(16, num_dims=4),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            d2l.FlattenLayer(),
            nn.Linear(16*4*4, 120),
            BatchNorm(120, num_dims=2),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            BatchNorm(84, num_dims=2),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )
print(net)

#依据计算能力选择
batch_size = 0
#batch_size = 256  
batch_size=16
assert batch_size > 0, '未选择batch_size'

#读取数据，使用其他数据集时更改该部分
def load_data_fashion_mnist(batch_size, resize=None, root='/home/kesci/input/FashionMNIST2065'):
    """Download the fashion mnist dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())
    
    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=2)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_iter, test_iter
train_iter, test_iter = load_data_fashion_mnist(batch_size)

if __name__ == '__main__':
    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)