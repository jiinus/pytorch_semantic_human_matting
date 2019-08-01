import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import cv2
import numpy as np
from data import dataset
from model import network
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from matting_measure import matting_measure


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Semantic Human Matting !')
    parser.add_argument('--train_fg_list', type=str, default='./data/train_fg_list.txt', help="training fore-ground images lists")
    parser.add_argument('--val_fg_list', type=str, default='./data/val_fg_list.txt', help="val fore-ground images lists")
    parser.add_argument('--bg_list', type=str, default='./data/bg_list.txt', help='train back-ground images list, one file')
    parser.add_argument('--saveDir', default='./ckpt', help='model save dir')
    parser.add_argument('--trainData', default='human_matting_data', help='train dataset name')

    parser.add_argument('--continue_train', action='store_true', default=False, help='continue training the training')
    parser.add_argument('--pretrain', action='store_true', help='load pretrained model from t_net & m_net ')
    parser.add_argument('--without_gpu', action='store_true', default=False, help='no use gpu')

    parser.add_argument('--train_nThreads', type=int, default=4, help='number of threads for data loading')
    parser.add_argument('--val_nThreads', type=int, default=1, help='number of threads for data loading')
    parser.add_argument('--train_batch', type=int, default=4, help='input batch size for train')
    parser.add_argument('--val_batch', type=int, default=1, help='input batch size for val')
    parser.add_argument('--patch_size', type=int, default=400, help='patch size for train')

    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--lrDecay', type=int, default=100)
    parser.add_argument('--lrdecayType', default='keep')
    parser.add_argument('--nEpochs', type=int, default=300, help='number of epochs to train')
    parser.add_argument('--save_epoch', type=int, default=5, help='number of epochs to save model')
    parser.add_argument('--print_iter', type=int, default=1000, help='pring loss and save image')

    parser.add_argument('--train_phase', type=str, default= 'end_to_end', help='train phase')
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode')

    args = parser.parse_args()
    return args


def set_lr(args, epoch, optimizer):

    lrDecay = args.lrDecay
    decayType = args.lrdecayType
    if decayType == 'keep':
        lr = args.lr
    elif decayType == 'step':
        epoch_iter = (epoch + 1) // lrDecay
        lr = args.lr / 2**epoch_iter
    elif decayType == 'exp':
        k = math.log(2) / lrDecay
        lr = args.lr * math.exp(-k * epoch)
    elif decayType == 'poly':
        lr = args.lr * math.pow((1 - epoch / args.nEpochs), 0.9)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

  

class Train_Log():
    def __init__(self, args):
        self.args = args

        self.save_dir = os.path.join(args.saveDir, args.train_phase)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.save_dir_model = os.path.join(self.save_dir, 'model')
        if not os.path.exists(self.save_dir_model):
            os.makedirs(self.save_dir_model)

        if os.path.exists(self.save_dir + '/log.txt'):
            self.logFile = open(self.save_dir + '/log.txt', 'a')
        else:
            self.logFile = open(self.save_dir + '/log.txt', 'w')

        #graph writer and best loss record
        self.writer = SummaryWriter(args.saveDir)
        self.best_loss = float('inf')

        # in case pretrained weights need to be loaded
        if self.args.pretrain:
            self.t_path = os.path.join(args.saveDir, 'pre_train_t_net', 'model', 'ckpt_best.pth')
            self.m_path = os.path.join(args.saveDir, 'pre_train_m_net', 'model', 'ckpt_best.pth')
            assert os.path.isfile(self.t_path) and os.path.isfile(self.m_path), \
                'Wrong dir for pretrained models:\n{},{}'.format(self.t_path, self.m_path)

            
    def save_model(self, model, optimizer, epoch, loss_val):
        lastest_out_path = "{}/ckpt_lastest.pth".format(self.save_dir_model)
        best_out_path = "{}/ckpt_best.pth".format(self.save_dir_model)
        if loss_val < self.best_loss:
            self.best_loss = loss_val
            torch.save({
            'epoch': epoch,
            'best_loss':self.best_loss,
            'state_dict': model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict()
            }, best_out_path)

        torch.save({
            'epoch': epoch,
            'best_loss':self.best_loss,
            'state_dict': model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict()
            }, lastest_out_path)

    def load_pretrain(self, model):
        t_ckpt = torch.load(self.t_path)
        model.load_state_dict(t_ckpt['state_dict'], strict=False)
        m_ckpt = torch.load(self.m_path)
        model.load_state_dict(m_ckpt['state_dict'], strict=False)
        print('=> loaded pretrained t_net & m_net pretrained models !')

        return model

    def load_model(self, model, optimizer):
        lastest_out_path = "{}/ckpt_lastest.pth".format(self.save_dir_model)
        ckpt = torch.load(lastest_out_path)
        self.best_loss = ckpt['best_loss']
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(lastest_out_path, ckpt['epoch']))

        return start_epoch, model, optimizer

    def save_log(self, log):
        self.logFile.write(log + '\n')

# initialise conv2d weights
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def loss_f_T(trimap_pre, trimap_gt):
    criterion = nn.CrossEntropyLoss()
    L_t = criterion(trimap_pre, trimap_gt[:, 0, :, :].long())

    return L_t


def loss_f_M(img, alpha_pre, alpha_gt, bg, fg, trimap):
    # -------------------------------------
    # prediction loss L_p
    # ------------------------
    eps = 1e-6
    # l_alpha
    L_alpha = torch.sqrt(torch.pow(alpha_pre - alpha_gt, 2.) + eps).mean()

    comp_pred = alpha_pre * fg + (1 - alpha_pre) * bg

    # be careful about here: if img's range is [0,1] then eps should divede 255
    L_composition = torch.sqrt(torch.pow(img - comp_pred, 2.) + eps).mean()

    L_p = 0.5 * L_alpha + 0.5 * L_composition

    return L_p, L_alpha, L_composition

def loss_function(img, trimap_pre, trimap_gt, alpha_pre, alpha_gt, bg, fg):

    # -------------------------------------
    # classification loss L_t
    # ------------------------
    criterion = nn.CrossEntropyLoss()
    L_t = criterion(trimap_pre, trimap_gt[:,0,:,:].long())

    # -------------------------------------
    # prediction loss L_p
    # ------------------------
    eps = 1e-6
    # l_alpha
    L_alpha = torch.sqrt(torch.pow(alpha_pre - alpha_gt, 2.) + eps).mean()

    comp_pred = alpha_pre * fg + (1 - alpha_pre) * bg

    # be careful about here: if img's range is [0,1] then eps should divede 255
    L_composition = torch.sqrt(torch.pow(img - comp_pred, 2.) + eps).mean()
    L_p = 0.5 * L_alpha + 0.5 * L_composition

    # train_phase
    loss = L_p + 0.01*L_t
        
    return loss, L_alpha, L_composition, L_t


def main():
    args = get_args()

    if args.without_gpu:
        print("use CPU !")
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print("No GPU is is available !")

    print("============> Building model ...")
    trainlog = Train_Log(args)
    if args.train_phase == 'pre_train_t_net':
        model = network.net_T()
    elif args.train_phase == 'pre_train_m_net':
        model = network.net_M()
        model.apply(weight_init)
    elif args.train_phase == 'end_to_end':
        model = network.net_F()
        if args.pretrain:
            model = trainlog.load_pretrain(model)
    else:
        raise ValueError('Wrong train phase request!')
    train_data = dataset.human_matting_data(args, split='train')
    val_data = dataset.human_matting_data(args, split='val')
    model.to(device)

    print(args)
    print("============> Loading datasets ...")

    trainloader = DataLoader(train_data,
                             batch_size=args.train_batch, 
                             drop_last=True, 
                             shuffle=True, 
                             num_workers=args.train_nThreads, 
                             pin_memory=True)

    valloader = DataLoader(val_data,
                             batch_size=args.val_batch, 
                             drop_last=True, 
                             shuffle=True, 
                             num_workers=args.val_nThreads, 
                             pin_memory=True)

    print("============> Set optimizer ...")
    lr = args.lr
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), \
                                   lr=lr, betas=(0.9, 0.999), 
                                   weight_decay=0.0005)    

    print("============> Start Train ! ...")
    start_epoch = 1
    if args.continue_train:
        start_epoch, model, optimizer = trainlog.load_model(model, optimizer)

    model.train() 
    for epoch in range(start_epoch, args.nEpochs+1):

        train_loss_ = 0
        train_L_alpha_ = 0
        train_L_composition_ = 0
        train_L_cross_ = 0
        train_SAD_ = 0
        train_MSE_ = 0
        train_Gradient_ = 0
        train_Connectivity_ = 0

        val_loss_ = 0
        val_L_alpha_ = 0
        val_L_composition_ = 0
        val_L_cross_ = 0
        val_SAD_ = 0
        val_MSE_ = 0
        val_Gradient_ = 0
        val_Connectivity_ = 0

        if args.lrdecayType != 'keep':
            lr = set_lr(args, epoch, optimizer)

        t0 = time.time()
        for i, sample_batched in enumerate(trainloader):

            optimizer.zero_grad()

            if args.train_phase == 'pre_train_t_net':
                img, trimap_gt = sample_batched['image'], sample_batched['trimap']
                img, trimap_gt = img.to(device), trimap_gt.to(device)

                trimap_pre = model(img)
                if args.debug:  #debug only
                    assert tuple(trimap_pre.shape) == (args.train_batch, 3, args.patch_size, args.patch_size)
                    assert tuple(trimap_gt.shape) == (args.train_batch, 1, args.patch_size, args.patch_size)

                train_loss = loss_f_T(trimap_pre, trimap_gt)

                train_loss_ += train_loss.item()

            elif args.train_phase == 'pre_train_m_net':
                img, trimap_gt, alpha_gt, bg, fg = sample_batched['image'], sample_batched['trimap'], sample_batched['alpha'], sample_batched['bg'], sample_batched['fg']
                img, trimap_gt, alpha_gt, bg, fg = img.to(device), trimap_gt.to(device), alpha_gt.to(device), bg.to(device), fg.to(device)

                alpha_pre = model(img, trimap_gt)
                train_loss, train_L_alpha, train_L_composition = loss_f_M(img, alpha_pre, alpha_gt, bg, fg, trimap_gt)

                train_loss_ += train_loss.item()
                train_L_alpha_ += train_L_alpha.item()
                train_L_composition_ += train_L_composition.item()

                alpha_pre = alpha_pre[:, 0, :, :].cpu().detach().numpy()
                alpha_gt = alpha_gt[:, 0, :, :].cpu().detach().numpy()

                SAD, MSE, Gradient, Connectivity = matting_measure(alpha_pre, alpha_gt)
                train_SAD_ += SAD
                train_MSE_ += MSE
                train_Gradient_ += Gradient
                train_Connectivity_ += Connectivity



            elif args.train_phase == 'end_to_end':
                img, trimap_gt, alpha_gt, bg, fg = sample_batched['image'], sample_batched['trimap'], sample_batched['alpha'], sample_batched['bg'], sample_batched['fg']
                img, trimap_gt, alpha_gt, bg, fg = img.to(device), trimap_gt.to(device), alpha_gt.to(device), bg.to(device), fg.to(device)

                trimap_pre, alpha_pre, img, alpha_gt, bg, fg = model({'img': img, 'alpha_g': alpha_gt, 'back': bg, 'front': fg})
                train_loss, train_L_alpha, train_L_composition, train_L_cross = loss_function(img, trimap_pre, trimap_gt, alpha_pre, alpha_gt, bg, fg)

                train_loss_ += train_loss.item()
                train_L_alpha_ += train_L_alpha.item()
                train_L_composition_ += train_L_composition.item()
                train_L_cross_ += train_L_cross.item()

                alpha_pre = alpha_pre[:, 0, :, :].cpu().detach().numpy()
                alpha_gt = alpha_gt[:, 0, :, :].cpu().detach().numpy()

                SAD, MSE, Gradient, Connectivity = matting_measure(alpha_pre, alpha_gt)
                train_SAD_ += SAD
                train_MSE_ += MSE
                train_Gradient_ += Gradient
                train_Connectivity_ += Connectivity

            train_loss.backward()
            optimizer.step()

        for j, sample_batched in enumerate(valloader):
            if args.train_phase == 'pre_train_t_net':
                img, trimap_gt = sample_batched['image'], sample_batched['trimap']
                img, trimap_gt = img.to(device), trimap_gt.to(device)

                trimap_pre = model(img)
                if args.debug:  #debug only
                    assert tuple(trimap_pre.shape) == (args.train_batch, 3, args.patch_size, args.patch_size)
                    assert tuple(trimap_gt.shape) == (args.train_batch, 1, args.patch_size, args.patch_size)

                val_loss = loss_f_T(trimap_pre, trimap_gt)

                val_loss_ += val_loss.item()

            elif args.train_phase == 'pre_train_m_net':
                img, trimap_gt, alpha_gt, bg, fg = sample_batched['image'], sample_batched['trimap'], sample_batched['alpha'], sample_batched['bg'], sample_batched['fg']
                img, trimap_gt, alpha_gt, bg, fg = img.to(device), trimap_gt.to(device), alpha_gt.to(device), bg.to(device), fg.to(device)

                alpha_pre = model(img, trimap_gt)
                val_loss, val_L_alpha, val_L_composition = loss_f_M(img, alpha_pre, alpha_gt, bg, fg, trimap_gt)

                val_loss_ += val_loss.item()
                val_L_alpha_ += val_L_alpha.item()
                val_L_composition_ += val_L_composition.item()

                alpha_pre = alpha_pre[:, 0, :, :].cpu().detach().numpy()
                alpha_gt = alpha_gt[:, 0, :, :].cpu().detach().numpy()

                SAD, MSE, Gradient, Connectivity = matting_measure(alpha_pre, alpha_gt)
                val_SAD_ += SAD
                val_MSE_ += MSE
                val_Gradient_ += Gradient
                val_Connectivity_ += Connectivity

            elif args.train_phase == 'end_to_end':
                img, trimap_gt, alpha_gt, bg, fg = sample_batched['image'], sample_batched['trimap'], sample_batched['alpha'], sample_batched['bg'], sample_batched['fg']
                img, trimap_gt, alpha_gt, bg, fg = img.to(device), trimap_gt.to(device), alpha_gt.to(device), bg.to(device), fg.to(device)

                trimap_pre, alpha_pre, img, alpha_gt, bg, fg = model({'img': img, 'alpha_g': alpha_gt, 'back': bg, 'front': fg})
                val_loss, val_L_alpha, val_L_composition, val_L_cross = loss_function(img, trimap_pre, trimap_gt, alpha_pre, alpha_gt, bg, fg)

                val_loss_ += val_loss.item()
                val_L_alpha_ += val_L_alpha.item()
                val_L_composition_ += val_L_composition.item()
                val_L_cross_ += val_L_cross.item()

                alpha_pre = alpha_pre[:, 0, :, :].cpu().detach().numpy()
                alpha_gt = alpha_gt[:, 0, :, :].cpu().detach().numpy()

                SAD, MSE, Gradient, Connectivity = matting_measure(alpha_pre, alpha_gt)
                val_SAD_ += SAD
                val_MSE_ += MSE
                val_Gradient_ += Gradient
                val_Connectivity_ += Connectivity


        # shuffle data after each epoch to recreate the dataset
        print('epoch end, shuffle datasets again ...')
        #trainloader.dataset.shuffle_data()

        t1 = time.time()

        if args.train_phase == 'pre_train_t_net':
            train_loss_ = train_loss_ / (i + 1)
            val_loss_ = val_loss_ / (j + 1)

            trainlog.writer.add_scalar('train_loss', train_loss_, epoch)
            trainlog.writer.add_scalar('val_loss', val_loss_, epoch)

            log = "[{} / {}]\ttime: {:.0f}\ttrain_loss: {:.5f}\tval_loss: {:.5f}" \
                .format(epoch, args.nEpochs, t1-t0, train_loss_, val_loss_)

        elif args.train_phase == 'pre_train_m_net':
            train_loss_ = train_loss_ / (i + 1)
            train_L_alpha_ = train_L_alpha_ / (i + 1)
            train_L_composition_ = train_L_composition_ / (i + 1)
            train_SAD_ = train_SAD_ / (i + 1)
            train_MSE_ = train_MSE_ / (i + 1)
            train_Gradient_ = train_Gradient_ / (i + 1)
            train_Connectivity_ = train_Connectivity_ / (i + 1)

            val_loss_ = val_loss_ / (j + 1)
            val_L_alpha_ = val_L_alpha_ / (j + 1)
            val_L_composition_ = val_L_composition_ / (j + 1)
            val_SAD_ = val_SAD_ / (j + 1)
            val_MSE_ = val_MSE_ / (j + 1)
            val_Gradient_ = val_Gradient_ / (j + 1)
            val_Connectivity_ = val_Connectivity_ / (j + 1)

            trainlog.writer.add_scalar('train_loss', train_loss_, epoch)
            trainlog.writer.add_scalar('train_loss_a', train_L_alpha_, epoch)
            trainlog.writer.add_scalar('train_loss_c', train_L_composition_, epoch)
            trainlog.writer.add_scalar('train_SAD', train_SAD_, epoch)
            trainlog.writer.add_scalar('train_MSE', train_MSE_, epoch)
            trainlog.writer.add_scalar('train_Gradient', train_Gradient_, epoch)
            trainlog.writer.add_scalar('train_Connectivity', train_Connectivity_, epoch)

            trainlog.writer.add_scalar('val_loss', val_loss_, epoch)
            trainlog.writer.add_scalar('val_loss_a', val_L_alpha_, epoch)
            trainlog.writer.add_scalar('val_loss_c', val_L_composition_, epoch)
            trainlog.writer.add_scalar('val_SAD', val_SAD_, epoch)
            trainlog.writer.add_scalar('val_MSE', val_MSE_, epoch)
            trainlog.writer.add_scalar('val_Gradient', val_Gradient_, epoch)
            trainlog.writer.add_scalar('val_Connectivity', val_Connectivity_, epoch)

            log = "[{} / {}]\ttime: {:.0f}\ttrain_loss: {:.5f}\ttrain_loss_a: {:.5f}\ttrain_loss_c: {:.5f}\n \
                train_SAD: {:.5f}\ttrain_MSE: {:.5f}\ttrain_Gradient: {:.5f}\ttrain_Connectivity: {:.5f}\n \
                val_loss: {:.5f}\tval_loss_a: {:.5f}\tval_loss_c: {:.5f}\n \
                val_SAD: {:.5f}\tval_MSE: {:.5f}\tval_Gradient: {:.5f}\tval_Connectivity: {:.5f}" \
                .format(epoch, args.nEpochs, t1 - t0, \
                        train_loss_, train_L_alpha_, train_L_composition_, \
                        train_SAD_, train_MSE_, train_Gradient_, train_Connectivity_, \
                        val_loss_, val_L_alpha_, val_L_composition_, \
                        val_SAD_, val_MSE_, val_Gradient_, val_Connectivity_)

        elif args.train_phase == 'end_to_end':
            train_loss_ = train_loss_ / (i + 1)
            train_L_alpha_ = train_L_alpha_ / (i + 1)
            train_L_composition_ = train_L_composition_ / (i + 1)
            train_L_cross_ = train_L_cross_ / (i + 1)
            train_SAD_ = train_SAD_ / (i + 1)
            train_MSE_ = train_MSE_ / (i + 1)
            train_Gradient_ = train_Gradient_ / (i + 1)
            train_Connectivity_ = train_Connectivity_ / (i + 1)

            val_loss_ = val_loss_ / (j + 1)
            val_L_alpha_ = val_L_alpha_ / (j + 1)
            val_L_composition_ = val_L_composition_ / (j + 1)
            val_L_cross_ = val_L_cross_ / (j + 1)
            val_SAD_ = val_SAD_ / (j + 1)
            val_MSE_ = val_MSE_ / (j + 1)
            val_Gradient_ = val_Gradient_ / (j + 1)
            val_Connectivity_ = val_Connectivity_ / (j + 1)

            trainlog.writer.add_scalar('train_loss', train_loss_, epoch)
            trainlog.writer.add_scalar('train_loss_a', train_L_alpha_, epoch)
            trainlog.writer.add_scalar('train_loss_c', train_L_composition_, epoch)
            trainlog.writer.add_scalar('train_loss_t', train_L_cross_, epoch)
            trainlog.writer.add_scalar('train_SAD', train_SAD_, epoch)
            trainlog.writer.add_scalar('train_MSE', train_MSE_, epoch)
            trainlog.writer.add_scalar('train_Gradient', train_Gradient_, epoch)
            trainlog.writer.add_scalar('train_Connectivity', train_Connectivity_, epoch)

            trainlog.writer.add_scalar('val_loss', val_loss_, epoch)
            trainlog.writer.add_scalar('val_loss_a', val_L_alpha_, epoch)
            trainlog.writer.add_scalar('val_loss_c', val_L_composition_, epoch)
            trainlog.writer.add_scalar('val_loss_t', val_L_cross_, epoch)
            trainlog.writer.add_scalar('val_SAD', val_SAD_, epoch)
            trainlog.writer.add_scalar('val_MSE', val_MSE_, epoch)
            trainlog.writer.add_scalar('val_Gradient', val_Gradient_, epoch)
            trainlog.writer.add_scalar('val_Connectivity', val_Connectivity_, epoch)

            log = "[{} / {}]\ttime: {:.0f}\ttrain_loss: {:.5f}\ttrain_loss_a: {:.5f}\ttrain_loss_c: {:.5f}\ttrain_loss_t: {:.5f}\n \
                train_SAD: {:.5f}\ttrain_MSE: {:.5f}\ttrain_Gradient: {:.5f}\ttrain_Connectivity: {:.5f}\n \
                val_loss: {:.5f}\tval_loss_a: {:.5f}\tval_loss_c: {:.5f}\tval_loss_t: {:.5f}\n \
                val_SAD: {:.5f}\tval_MSE: {:.5f}\tval_Gradient: {:.5f}\tval_Connectivity: {:.5f}" \
                .format(epoch, args.nEpochs, t1 - t0, \
                        train_loss_, train_L_alpha_, train_L_composition_, train_L_cross_, \
                        train_SAD_, train_MSE_, train_Gradient_, train_Connectivity_, \
                        val_loss_, val_L_alpha_, val_L_composition_, val_L_cross_,\
                        val_SAD_, val_MSE_, val_Gradient_, val_Connectivity_)
        print(log)
        trainlog.save_log(log)
        trainlog.save_model(model, optimizer, epoch, val_loss_)




if __name__ == "__main__":
    main()
