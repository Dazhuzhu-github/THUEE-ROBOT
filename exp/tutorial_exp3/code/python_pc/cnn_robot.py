import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils import data
from torch.optim.lr_scheduler import StepLR
import os.path
# import matplotlib.pyplot as plt
import re
import numpy as np
from PIL import Image

import argparse

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# data_transformer = transforms.Compose([
#     transforms.Resize((28,28)),
#     transforms.ToTensor(),
#     normalize
# ])

def my_transform(img):
    img = img.resize((28,28))
    im_data = np.array(img).astype(np.float32)
    im_data = im_data.transpose(2,0,1)
    for i in range(im_data.shape[0]):
        im_data[i,:,:] = (im_data[i,:,:] - np.mean(im_data[i,:,:])) / np.std(im_data[i,:,:])
    return im_data

class myDataSet(data.Dataset):
    def __init__(self, is_test):
        self.data_path = 'dataset/'
        self.data_path2 = 'testdata/'
        self.image = []
        self.label = []
        # self.transform = transform
        self.class_name = ['duck', 'others']
        if is_test:
            # self.image.append(np.array(Image.open(self.data_path+str(i)+'.jpeg').convert('RGB')))
                # if i is 6:
                #     for j in range(5):
                #         self.image.append(Image.open(self.data_path+self.class_name[i]+'/'+str(j)+'.jpg').convert('RGB'))
                #         self.label.append(i)
                # else:
                #     for j in range(1):
                #         self.image.append(Image.open(self.data_path+self.class_name[i]+'/'+str(j)+'.jpg').convert('RGB'))
                #         self.label.append(i)
            for j in range(34):
                self.image.append(Image.open(self.data_path2+str(j)+'.jpg').convert('RGB'))
            self.label.extend([0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1])
            print("label",self.label)
        else:
            for i in range(len(self.class_name)):
                if i is 0:
                    for j in range(7):
                        self.image.append(Image.open(self.data_path+self.class_name[i]+'/'+str(j)+'.jpg').convert('RGB'))
                        self.label.append(i)
                else:
                    for j in range(11):
                        self.image.append(Image.open(self.data_path+self.class_name[i]+'/'+str(j)+'.jpg').convert('RGB'))
                        self.label.append(i)                
        # self.image = np.array(self.image)
        self.label = torch.LongTensor(np.array(self.label))
        # print('len(image)', self.eeg.shape)
        print('label.size', self.label.size())
 
    def __getitem__(self, index):
        sample = {'img': torch.Tensor(my_transform(self.image[index])),
                      'label': self.label[index]}            
        return sample

    def __len__(self):
        return len(self.image)

trainset = myDataSet(is_test=False)
testset = myDataSet(is_test=True)


train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                           batch_size=6,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=testset,
                                          batch_size=6,
                                          shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 输入1通道，输出10通道，kernel 5*5
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, 2)

    def forward(self, x):
        in_size = x.size(0) # one batch
        x = F.relu(self.mp(self.conv1(x)))
        # print('relu1 out for debug: ', x)
        x = F.relu(self.mp(self.conv2(x)))
        # print('relu2 out for debug: ', x)
        x = x.view(in_size, -1) # flatten the tensor
        x = self.fc(x)
        print('fc out for debug: ', x)
        return F.log_softmax(x)

model = Net()
device = 'cpu'

def my_train(istest, batch_size, lr, epochs):
    best_acc = 0
    model_path = 'model_param'
    if os.path.exists(os.path.join(model_path, 'model.pkl')):
        model.load_state_dict(torch.load(os.path.join(model_path, 'model.pkl')))
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    if not istest:
        model.train()
        for epoch in range(epochs):
            epoch_accuracy = 0
            running_loss = 0.0
            print('epoch', epoch)
            for i, img_label in enumerate(train_loader):
                feature = Variable(img_label['img']).to(device)
                label = Variable(img_label['label']).to(device)
                optimizer.zero_grad()
                output = model(feature)
                #print('output.shape', output.shape)
                #print('label.shape', label.shape)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                acc = (output.argmax(dim=1) == label).float().mean()
                epoch_accuracy += acc / len(train_loader)
            print('train loss:', running_loss / len(train_loader))
            print('train acc:', epoch_accuracy.item())
            torch.save(model.state_dict(), os.path.join(model_path, 'model.pkl'))  
            #val
            # torch.no_grad()
            corr = 0
            val_running_loss = 0.0 
            val_epoch_accuracy = 0
            # for i, (feature, label) in enumerate(testLoader):
            for i, img_label in enumerate(test_loader):
                # feature = Variable(feature).to(device)
                # label = Variable(label).to(device)
                feature = Variable(img_label['img']).to(device)
                label = Variable(img_label['label']).to(device)
                output = model(feature)
                loss = criterion(output, label)
                val_running_loss += loss.item()
                acc = (output.argmax(dim=1) == label).float().mean()
                val_epoch_accuracy += acc / len(test_loader)
            print('val loss:', val_running_loss/len(test_loader))
            print('val acc:', val_epoch_accuracy.item())
            if val_epoch_accuracy > best_acc:
                best_acc = val_epoch_accuracy
                torch.save(model.state_dict(), os.path.join(model_path, 'model{}_{}.pkl'.format(epoch, val_epoch_accuracy)))
            
        print('Finished Training')
        print('best acc', best_acc)
        torch.save(model.state_dict(), os.path.join(model_path, 'model.pkl')) 
    if istest:
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for img_label in test_loader:
                #images = Image.open("./testdata"+'/'+str(i)+'.jpg').convert('RGB')
                #labels = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5]
                feature = Variable(img_label['img']).to(device)
                labels = Variable(img_label['label']).to(device)
                outputs = model(feature)
                _, predicted = torch.max(outputs.data,1)
                total +=len(labels)
                #print(outputs)
                print(torch.max(outputs.data,1))
                print("labels:",labels) 
                correct += (predicted == labels).sum().item()
                print("predicted:",predicted)
                print(predicted == labels)
                print("correct:",correct) 
        print("Accuracy of this net work is", (correct/total),"in tatal",total,"data")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input:BatchSize initial LR EPOCH')
    parser.add_argument('--test', type=bool, default=False, help='set test mode')
    parser.add_argument('--BATCH_SIZE', type=int, default=6, help='batch_size')
    parser.add_argument('--LR', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--EPOCH', type=int, default=200, help='epoch')

    args = parser.parse_args()

    print('batch_size:', args.BATCH_SIZE)
    print('initial LR:', args.LR)
    print('epoch:', args.EPOCH)

    my_train(istest=args.test, batch_size=args.BATCH_SIZE, lr=args.LR, epochs=args.EPOCH)

    