'''
Code written by: Xiaoqing Liu
If you use significant portions of this code or the ideas from our paper, please cite it :)
'''
import torch
import argparse
import os
from torch.utils.data import DataLoader
from model.build_BiSeNet import BiSeNet
import numpy as np
from utils import reverse_one_hot, get_label_info, colour_code_segmentation, compute_global_accuracy

import matplotlib.pyplot as plt
from matplotlib.pylab import array
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage
from dataset.transform import Relabel, ToLabel, Colorize


def infer(model):
    print('start test!')
    label_np=array(Image.open('./data/test/Labels/672.png'))
    label = Variable(torch.from_numpy(label_np))
    img=Image.open('./data/test/Images/672.png')
    img_n=array(img)
    
    img_np=np.resize(img_n,(480,480,3))
    outputs = model(Variable(torch.from_numpy(img_np[np.newaxis, :].transpose(0,3,1,2)).float(),volatile = True).cuda())
       # print outputs.size()
    outputs = outputs.squeeze()
    print outputs
    predict = reverse_one_hot(outputs)
    print predict
    label = label.squeeze()
    print label

    fig,ax=plt.subplots(1,3)
    ax[0].imshow(img_np,cmap='gray')
    ax[1].imshow(predict)
    ax[2].imshow(label)
    plt.show()

    return 0

def main(params):

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The path to the pretrained weights of model')
    parser.add_argument('--crop_height', type=int, default=640, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=640, help='Width of cropped/resized input image to network')
    parser.add_argument('--data', type=str, default='/path/to/data', help='Path of training data')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101", help='The context path model you are using.')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to user gpu for training')
    parser.add_argument('--num_classes', type=int, default=2, help='num of object classes (with void)')
    args = parser.parse_args(params)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()
    # load pretrained model if exists
    print('load model from %s ...' % args.checkpoint_path)
    model.module.load_state_dict(torch.load(args.checkpoint_path))
    print('Done!')

    # test
    infer(model)


if __name__ == '__main__':
    params = [
        '--checkpoint_path', './checkpoints/epoch_66.pth',
        '--data', '/path/to/CamVid',
        '--cuda', '0'
    ]
    main(params)
