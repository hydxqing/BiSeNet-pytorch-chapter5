'''
Code written by: Xiaoqing Liu
If you use significant portions of this code or the ideas from our paper, please cite it :)
'''
import argparse
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
#from dataset.CamVid import CamVid
import os
from model.build_BiSeNet import BiSeNet
import torch
#from tensorboardX import SummaryWriter
#import tqdm
from torch.nn import functional as F
import numpy as np
from utils import poly_lr_scheduler
from utils import reverse_one_hot, get_label_info, colour_code_segmentation, compute_global_accuracy

from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage

from dataset.dataset import train
from dataset.transform import Relabel, ToLabel, Colorize

image_transform = ToPILImage()
input_transform = Compose([
    Resize((480,480)),
    ToTensor(),
])
target_transform = Compose([
    Resize((480,480)),
    ToLabel(),
])

def val(args, model, dataloader, csv_path):
    print('start val!')
    label_info = get_label_info(csv_path)
    with torch.no_grad():
        model.eval()
        precision_record = []
        for i, (data, label) in enumerate(dataloader):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()

            # get RGB predict image
            predict = model(data).squeeze()
            predict = reverse_one_hot(predict)
            predict = colour_code_segmentation(np.array(predict), label_info)

            # get RGB label image
            label = label.squeeze()
            label = reverse_one_hot(label)
            label = colour_code_segmentation(np.array(label), label_info)
            # compute per pixel accuracy
            precision = compute_global_accuracy(predict, label)
            precision_record.append(precision)
        dice = np.mean(precision_record)
        print('precision per pixel for validation: %.3f' % dice)
        return dice

def train(args, model, optimizer, dataloader_train):
    step = 0
    for epoch in range(args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        model.train()
        loss_record = []
        for i,(data, label) in enumerate(dataloader_train):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()
            output = model(data)
            loss = torch.nn.CrossEntropyLoss()(output, label[:,0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            loss_record.append(loss.item())
            if i % 50 == 0:
                average = sum(loss_record) / len(loss_record)
                print('epoch:%f'%epoch,'step:%f'%i,'loss:%f'%average)
        loss_train_mean = np.mean(loss_record)
        print('loss for train : %f' % (loss_train_mean))
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'epoch_{}.pth'.format(epoch)))


def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs to train for')
    parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')

    parser.add_argument('--checkpoint_step', type=int, default=1, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
    parser.add_argument('--dataset', type=str, default="CamVid", help='Dataset you are using.')
    parser.add_argument('--crop_height', type=int, default=640, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=640, help='Width of cropped/resized input image to network')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101", help='The context path model you are using.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate used for train')
    parser.add_argument('--data', type=str, default='/path/to/data',help='path of training data')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=2, help='num of object classes (with void)')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')

    args = parser.parse_args(params)

    # create dataset and dataloader
    dataloader_train = DataLoader(train(input_transform, target_transform),num_workers=1, batch_size=2, shuffle=True)

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * 4 / 1000 / 1000))

    # build optimizer
    optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)

    # load pretrained model if exists
    if args.pretrained_model_path is not None:
        print('load model from %s ...' % args.pretrained_model_path)
        model.module.load_state_dict(torch.load(args.pretrained_model_path))
        print('Done!')

    # train
    train(args, model, optimizer, dataloader_train)

if __name__ == '__main__':
    params = [
        '--num_epochs', '70',
        '--learning_rate', '0.0001',
        '--data', '/path/to/CamVid',
        '--num_workers', '4',
        '--num_classes', '2',
        '--cuda', '0',
        '--batch_size', '1',
        '--save_model_path', './checkpoints',
        #'--pretrained_model_path','./checkpoints-30/epoch_29.pth'
    ]
    main(params)

