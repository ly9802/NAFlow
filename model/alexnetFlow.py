# -- coding:utf-8 --
# Time:2025/12/25 19:11

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from .BackPropagationModule import ReverseReLU, ReverseMaxPool2d, ReverseComplex, ReverseConv

__all__ = ['AlexNet_Flow', 'alexnet_flow']

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

def check_path(path):
    if os.path.exists(path):
        pass;
    else:
        os.makedirs(path);
def transform_function(resolution=224, is_train=False):
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode
    trans_train = transforms.Compose([
        transforms.RandomResizedCrop((resolution, resolution), scale=(0.2, 1.)),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    to_pil = transforms.ToPILImage();
    trans_test = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if is_train:
        return trans_train
    else:
        return transform_test
def read_image(img_path):
    got_img = False
    import os
    from PIL import Image
    if not os.path.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    img = None
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
    return img

class AlexNet(nn.Module):
    def __init__(self,num_classes=1000):
        super(AlexNet,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self,x):
        x = self.features(x)
        x = x.contiguous().view(x.size(0), -1)
        score_vector = self.classifier(x)
        return score_vector

class AlexNet_Flow(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet_Flow, self).__init__()
        self.model=AlexNet(num_classes)

    def get_target_width_height(self, input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def scale_batch_map(self,bmap, target_size=None):
        result_list = []
        for img in bmap:
            img = img - np.min(img)
            img = img / (1e-10 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            img = np.uint8(255 * img)
            result_list.append(img)
        return result_list[0]

    def forward(self, x):
        print("---------------------Forward Propagation--------------")
        batchsize = x.size(dim=0)
        target_size = self.get_target_width_height(x)
        fpmap_list = []
        bpmap_list = []
        bpmodule_list = []
        x = torch.autograd.Variable(x, requires_grad=True)
        for idx, module in enumerate(self.model.features.children()):
            if isinstance(module, nn.MaxPool2d):
                y = module(x)
                bpmodule_list.append(ReverseMaxPool2d(module, x,y))
                x=y
            elif isinstance(module, nn.ReLU):
                y = module(x)
                bpmodule_list.append(ReverseReLU(module, x,y))
                x=y
            elif isinstance(module, nn.Conv2d):
                bs, dim_x, h_x, w_x=x.shape
                y=module(x)
                bs_y,dim_y, h_y, w_y=y.shape
                q=dim_y*h_y*w_y
                p=dim_x*h_x*w_x
                if q>=p:
                    bpmodule_list.append(ReverseConv(module, x, y))
                else:
                    current_length=len(bpmodule_list)
                    complex_list=[module]
                    for j in range(current_length-1,0,-1):
                        ej_module=bpmodule_list.pop(j)
                        complex_list.append(ej_module.module)
                        if isinstance(ej_module.module, nn.Conv2d):
                            if q>ej_module.num_x:
                                complex_list.reverse();
                                complex_module=nn.Sequential(*complex_list)
                                bpmodule_list.append(ReverseComplex(complex_module, ej_module.input, y))
                                break;
                            else:
                                continue;
                        else:
                            continue;
                x=y;
            else:
                x = module(x);
            fpmap_list.append(x)

        num_layers = len(bpmodule_list)

        feature_vector = x.contiguous().view(batchsize, -1)
        score_vector = self.model.classifier(feature_vector)
        score, predicted_label = torch.max(score_vector, dim=-1);
        predicted_label = predicted_label.detach().cpu().item()
        print("---------------------Backward Propagation--------------")
        self.model.zero_grad()
        score.backward(gradient=None, retain_graph=True,create_graph=False,inputs=x)
        gradient=x.grad
        bpmap_list.append(F.relu(x*gradient))
        for ind in range(num_layers-1,0,-1):
            bp_module=bpmodule_list[ind]
            y, gradient=bp_module(x, gradient)

            bpmap_list.append(F.relu(y*gradient))
            x=y
        bpmap_list.reverse()
        print("---------------------Generate NAFlow--------------")
        flow_list=[]
        for no, map in enumerate(bpmap_list):
            attn_map=F.relu(torch.sum(map,dim=1, keepdim=False))
            scaled_map=self.scale_batch_map(attn_map.detach().cpu().numpy(), target_size)
            flow_list.append(scaled_map)
        return  predicted_label,flow_list

def alexnet_flow(pretrained=False, **kwargs):
    net = AlexNet_Flow(**kwargs)
    if pretrained:
        net.model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return net


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    print("well done!")