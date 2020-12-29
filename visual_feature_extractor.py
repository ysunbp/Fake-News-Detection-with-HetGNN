import torch.nn as nn
import torchvision.models as models
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        # 取掉model的最后层
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
    def forward(self, x):
        x = self.resnet_layer(x)
        return x

#resnet50 = models.resnet50(pretrained=True, progress=True)
resnet18 = models.resnet18(pretrained=True, progress=True)
#model = resnet18
model = Net(resnet18)
#print(model) #output size 16*512*1*1


class weiboDataset(Dataset):
    def __init__(self, root, resize):
        self.image_files = np.array([x.path for x in os.scandir(root) if x.name.endswith(".jpg") or x.name.endswith(".png") or x.name.endswith(".JPG")])
        self.transform = transforms.Compose([transforms.Resize(size=(resize, resize))])
        self.toTensor = transforms.ToTensor()
    def __getitem__(self, index):
        path = self.image_files[index]
        #print(path)
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        img = self.toTensor(img)
        return img
    def __len__(self):
        return len(self.image_files)


file_path = "F:\\weibo_img\\"
all_img = os.listdir(file_path)
print(all_img)
dataset = weiboDataset(file_path, 224)
train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
EPOCH = 1

for epoch in range(EPOCH):
    for step, data in enumerate(train_loader):
        txt_name = all_img[step][:-4]
        out = model(data)
        out_np = out.detach().numpy()
        out_np = np.reshape(out_np, (1,512))
        np.savetxt('F:\\visual_features\\'+txt_name+'.txt', out_np)
