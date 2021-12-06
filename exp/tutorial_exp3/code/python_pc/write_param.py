import numpy as np
import torch
from cnn_robot import Net, myDataSet
from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
data_transformer = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    normalize
])

model = Net()
model_path = 'model_param/model.pkl'
model.load_state_dict(torch.load(model_path))

testset = myDataSet(is_test=True)

model(testset[0]['img'].reshape(1,3,28,28))
#存储网络参数
k=model.state_dict().keys()
f=open("./network.h","w+")
f.write('#ifndef NET_H\n')
f.write('#define NET_H\n')
for name, parameters in model.named_parameters():
    f.write("float "+name.replace(".","_"))
    if (parameters.ndim==1):
        f.write('[]={' + '\n')
    else :
        if (parameters.ndim==4):
            f.write('[]['+str(parameters.shape[1]*parameters.shape[2]*parameters.shape[3])+']={' + '\n')
        else:
            f.write('[][' + str(parameters.shape[1]) + ']={' + '\n')
    num=0
    for ker in parameters.detach().numpy():
        num+=1
        if parameters.ndim > 1:
            f.write('{')
        fl=ker.flat
        for it in fl:
            f.write(str(it))
            if (fl.index!=ker.size):
                f.write(',')
        if parameters.ndim > 1:
            f.write('}')
        if (num!=parameters.shape[0]):
            f.write(',' + '\n')
        else:
            f.write('\n')
    f.write('};'+'\n')
f.write('#endif\n')
f.close()

#存储图片
f=open("./picture.h","w+")
f.write('#ifndef PIC_H\n')
f.write('#define PIC_H\n')
f.write("float pokeman[]["+str(testset[0]['img'].shape[1])+"]["+str(testset[0]['img'].shape[2])+"]={\n")
num = 0
for row in (testset[0]['img'].numpy()):
    num += 1
    f.write('{')
    num_num = 0
    for row_row in row:
        num_num += 1
        # print(row_row.shape)
        fl = row_row.flat
        f.write('{')
        for it in fl:
            f.write(str(it))
            if (fl.index!=row_row.size):
                f.write(',')
        f.write('}')
        # print(testset[0]['img'].shape[0])
        if (num_num!=testset[0]['img'].shape[1]):
            f.write(',' + '\n')
        else:
            f.write('\n')
    f.write('}')
    if (num!=testset[0]['img'].shape[0]):
        f.write(',' + '\n')
    else:
        f.write('\n')
f.write('};'+'\n')
f.write('#endif\n')
f.close()