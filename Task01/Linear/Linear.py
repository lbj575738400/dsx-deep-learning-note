import torch
import numpy as np
import sys
import random
import torch.optim as optim
print('torch version: ' +str(torch.__version__))


def read_data(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # random read samples
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # min -> the last time may be not enough for a batch
        yield  features.index_select(0, j), labels.index_select(0, j)

#true model define
def true_model_generate(nums_features=3):
    numbers_features = 3 # numbers of features
    true_weight = np.random.normal(0, 5.0, numbers_features)
    true_bias = np.random.normal(0, 10.0)
    #print(true_weight,true_bias)
    return true_weight,true_bias

#generate test data and train data
def generate_data(true_weight,true_bias,numbers_train=100,numbers_test=100,numbers_features=3):
    feature = torch.randn((numbers_test + numbers_train, numbers_features))
    labels = []
    for i in range(len(feature)):
        labels.append(0.0)
        for j in range(len(feature[i])):
            labels[i] += true_weight[j] * feature[i][j]
        labels[i] += true_bias
    labels = torch.tensor(labels) #labels = w[0]*f[0]+w[1]*f[1]+....+w[n]*f[n]
    labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype= torch.float)# unknown bias

    train_features = feature[:numbers_train]
    train_labels = labels[:numbers_train]
    test_features = feature[numbers_train:]
    test_labels = labels[numbers_train:]
    return train_features,train_labels,test_features,test_labels

class Linear:
    def __init__(self, feature, labels, lr = 0.03, nums_epochs = 5, batch_size = 10):
        self.feature = feature
        self.labels = labels
        self.nums_epochs = nums_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.w = torch.tensor(np.random.normal(0, 0.01, (len(feature[0]), 1)), dtype=torch.float32)
        self.b = torch.zeros(1, dtype=torch.float32)
        self.w.requires_grad_(requires_grad=True)
        self.b.requires_grad_(requires_grad=True)
    
    #linear regression 
    def linreg(self, feature, weight, bias):
        return torch.mm(feature, weight) + bias
        
    #optimization function 
    def sgd(self, params, lr, batch_size): 
        for param in params:
            param.data -= lr * param.grad / batch_size # ues .data to operate param without gradient track

    #loss function
    def sq_loss(self, y_hat,y):
        return ((y_hat-y.view(y_hat.size()))**2 / 2) # loss function is MSE

    #train function
    def train(self,loss_show=True):
        for eps in range(self.nums_epochs):
            for x,y in read_data(self.batch_size, self.feature, self.labels):
                ls = self.sq_loss(self.linreg(x,self.w,self.b), y).sum()
                ls.backward()
                self.sgd([self.w, self.b], self.lr, self.batch_size)
                self.w.grad.data.zero_()
                self.b.grad.data.zero_()
            if loss_show:
                train_ls = self.sq_loss(self.linreg(self.feature, self.w, self.b), self.labels)
                print('epoch %d loss: %f' % (eps+1, train_ls.mean().item()))

    def calculate(self, feature, labels):
        ls = self.sq_loss(self.linreg(feature, self.w, self.b), labels).sum()
        print('test loss: %f' % ls)
    
    def model_show(self):
        print('w: ')
        print(self.w)
        print('b: %f' % self.b)



if __name__ == "__main__":
    epochs = 5
    lr = 0.03
    true_weight,true_bias = true_model_generate()
    train_features,train_labels,test_features,test_labels = generate_data(true_weight,true_bias)
    model = Linear(train_features,train_labels,nums_epochs=100)
    model.train()
    model.model_show()
    print('true w: ')
    print(true_weight)
    print('true b: %f' % true_bias)
    model.calculate(feature=test_features,labels=test_labels)