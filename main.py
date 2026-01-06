# -- coding:utf-8 --
# Time:2025/12/23 11:52
# File:main.py
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import argparse
import numpy as np
import torch

print("pytorch version:", torch.__version__)
print("cuda version:", torch.version.cuda)

np.random.seed(0)
torch.manual_seed(0)

from util.imagenet_index import index2class
from util.ImageNetValImageToClassDict import ImageName2Class_dict
from util.ImageNetClassIDToNameDict import ClassID2Name_dict
from util.ImageNetClassIDToNumDict import ClassID2Num_dict
from util.utils import pretrained_weights, load_pretrained_weights, check_path, transform_function, read_image,show_image

def model_define(args):
    if args.model_choice == "AlexNet":
        from model.alexnetFlow import AlexNet_Flow
        net = AlexNet_Flow()
        path = pretrained_weights(args.model_choice)
        load_pretrained_weights(net.model, path)
        net.to(args.device)
        net.eval()
        return net

    else:
        return None
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-set", default="ImageNet2012", type=str, help="Dataset Name")
    parser.add_argument('--image-path', default='./examples/ILSVRC2012_val_00013393.JPEG', type=str, help='test image path')
    parser.add_argument("--model-choice", default="AlexNet", choices=["AlexNet"])
    parser.add_argument('--pretrained_weights', default="./pretrainedweights/alexnet-owt-4df8aa71.pth", type=str, help="Path to The Pretrained Weights")
    parser.add_argument('--use-cuda', default=True, action='store_true')
    parser.add_argument("--save-dir", default="./examples/NAFlow/", type=str, help="Where to save the results");
    parser.add_argument("--device", default=None);
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    args.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')
    return args

if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device=device
    check_path(args.save_dir)
    trans = transform_function(resolution=224, is_train=False)
    net=model_define(args)

    image_name = args.image_path.split("/")[-1]
    prefix = image_name.split(".", 1)[0]
    print("Image:", args.image_path)
    print("Ground Truth Name:",ClassID2Name_dict[ImageName2Class_dict[image_name]])
    print("Ground Truth Label:",ClassID2Num_dict[ImageName2Class_dict[image_name]])
    print("Ground Truth ID:", ImageName2Class_dict[image_name])

    test_tensor=trans(read_image(args.image_path)).to(args.device).unsqueeze(dim=0)
    predicted_label,flow_list=net(test_tensor);
    print("Predicted Label:{}, Predicted Name:{}".format(predicted_label,index2class[predicted_label]))

    if predicted_label==ClassID2Num_dict[ImageName2Class_dict[image_name]]:
        print("Correct Prediction!")
        for item_no, cam_ndarray in enumerate(flow_list):
            new_name = prefix + "_" + str(item_no)
            show_image(cam_ndarray, img_path=args.image_path, name=new_name, save_dir=args.save_dir,original=False)
    else:
        print("Wrong Prediction!")

    print("Model:", args.model_choice)
    print("Finish!");
