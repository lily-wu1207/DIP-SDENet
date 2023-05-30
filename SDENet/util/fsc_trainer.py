import os
import sys
import time
import logging
from math import ceil

import torch
from torch import optim
from torch.utils.data import DataLoader

# from datasets.fsc_data import FSCData
from data.prepare_data import FSC147
from model.SDCAT import SDCAT
from models.genloss import GeneralizedLoss
# from losses.losses import DownMSELoss
from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import wandb


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    dmaps = torch.stack(transposed_batch[1], 0)
    ex_list = transposed_batch[2]
    return images, dmaps, ex_list


class FSCTrainer(Trainer):
    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        args = self.args
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            raise Exception("gpu is not available")

        train_datasets = FSC147(args.data_dir)

        train_dataloaders = DataLoader(train_datasets,
                                       collate_fn=FSC147.collate_fn,
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=args.num_workers,
                                       pin_memory=True)
        val_datasets = FSC147(args.data_dir, mode='test')
        val_dataloaders = DataLoader(val_datasets, 1, shuffle=False,
                                                      num_workers=args.num_workers, pin_memory=True)
        self.dataloaders = {'train': train_dataloaders, 'val': val_dataloaders}

        self.model = SRVIT(dcsize=args.dcsize)
        self.model.to(self.device)

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

        self.criterion = GeneralizedLoss(128.)
        self.criterion.to(self.device)

        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.best_mae_at = 0
        self.best_count = 0

        self.start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
                self.best_mae = checkpoint['best_mae']
                self.best_mse = checkpoint['best_mse']
                self.best_mae_at = checkpoint['best_mae_at']
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))

        if args.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.step, gamma=args.gamma, last_epoch=self.start_epoch-1)
        elif args.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.t_max, eta_min=args.eta_min, last_epoch=self.start_epoch-1)


        save_model = torch.load('./ckpt_epoch_best.pth')['model']
        model_dict = self.model.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        # print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
        model_dict.update(state_dict)
        self.model.load_state_dict(model_dict)

        for param in self.model.named_parameters():
            # print(param[0])
            if param[0].split('.')[0] == 'encoder':
                param[1].requires_grad = False
                print(param[0])

    def train(self):
        args = self.args
        self.epoch = None
        # self.val_epoch()
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-' * 5)
            self.epoch = epoch
            self.train_epoch()
            # mae, mse, loss = validate(self.dataloaders['val'], self.model, self.criterion)
            self.scheduler.step()
            if epoch >= args.val_start and (epoch % args.val_epoch == 0 or epoch == args.max_epoch - 1):
                self.val_epoch()

    def train_epoch(self):
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.train()

        # Iterate over data.
        # for inputs, targets, ex_list in tqdm(self.dataloaders['train']):
        for (samples, boxes, targets, imgids) in tqdm(self.dataloaders['train']):
            # inputs = inputs.to(self.device)
            # targets = targets.to(self.device) * self.args.log_param
            samples = samples.cuda(non_blocking=True) # image
            boxes = boxes.cuda(non_blocking=True) # boxes
            targets = targets.cuda(non_blocking=True) * self.args.log_param # dotmap

            with torch.set_grad_enabled(True):
                et_dmaps = self.model(samples, boxes)
                # loss = self.criterion(et_dmaps, targets)
                bsize = torch.stack((boxes[:, 4] - boxes[:, 2], boxes[:, 3] - boxes[:, 1]), dim=-1)
                bs_mean = bsize.view(-1, 3, 2).float().mean(dim=1)
                loss = self.criterion(et_dmaps, targets, box_size=bs_mean)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                N = samples.size(0)
                pre_count = torch.sum(et_dmaps.view(N, -1), dim=1).detach().cpu().numpy()
                gd_count = torch.sum(targets.view(N, -1), dim=1).detach().cpu().numpy()
                res = pre_count - gd_count
                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(res * res), N)
                epoch_mae.update(np.mean(abs(res)), N)

        logging.info('Epoch {} Train, Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, epoch_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
                             time.time() - epoch_start))
        wandb.log({'Train/loss': epoch_loss.get_avg(),
                   'Train/lr': self.scheduler.get_last_lr()[0],
                   'Train/epoch_mae': epoch_mae.get_avg()}, step=self.epoch)

        model_state_dic = self.model.state_dict()
        # if(self.epoch % 3 == 0):
        save_path = self.save_dir + '{}_ckpt.tar'.format(self.epoch)
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic,
            'best_mae': self.best_mae,
            'best_mse': self.best_mse,
            'best_mae_at': self.best_mae_at,
        }, save_path)
        self.save_list.append(save_path)  # control the number of saved models

    @torch.no_grad()
    def validate(self, data_loader, model, criterion):
        model.eval()

        batch_time = AverageMeter()
        loss_meter = AverageMeter()

        mae_meter = AverageMeter()
        mse_meter = AverageMeter()

        end = time.time()
        for idx, (images, boxes, target, imgids) in enumerate(data_loader):
            images = images.cuda(non_blocking=True)
            boxes = boxes.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            bsize = target.size(0)
            # compute output
            with torch.no_grad():
                output = model(images, boxes)
                output = F.relu(output, inplace=True)
                output = output / 128.
                tarsum = target.sum(dim=(1, 2, 3))

            loss = criterion(output * 128., target)
            diff = torch.abs(output.sum(dim=(1, 2, 3)) - tarsum)
            mae, mse = diff.mean(), (diff ** 2).mean()

            loss_meter.update(loss.item(), bsize)
            mae_meter.update(mae.item(), bsize)
            mse_meter.update(mse.item(), bsize)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        return mae_meter.avg, mse_meter.avg ** 0.5, loss_meter.avg

    def val_epoch(self):
        epoch_start = time.time()
        self.model.eval()
        epoch_res = []

        for inputs, count, ex_list, name in tqdm(self.dataloaders['val']):
            inputs = inputs.to(self.device)
            # inputs are images with different sizes
            b, c, h, w = inputs.shape
            h, w = int(h), int(w)
            assert b == 1, 'the batch size should equal to 1 in validation mode'

            max_size = 2000
            if h > max_size or w > max_size:
                h_stride = int(ceil(1.0 * h / max_size))
                w_stride = int(ceil(1.0 * w / max_size))
                h_step = h // h_stride
                w_step = w // w_stride
                input_list = []
                for i in range(h_stride):
                    for j in range(w_stride):
                        h_start = i * h_step
                        if i != h_stride - 1:
                            h_end = (i + 1) * h_step
                        else:
                            h_end = h
                        w_start = j * w_step
                        if j != w_stride - 1:
                            w_end = (j + 1) * w_step
                        else:
                            w_end = w
                        input_list.append(inputs[:, :, h_start:h_end, w_start:w_end])
                with torch.set_grad_enabled(False):
                    pre_count = 0.0
                    for input_ in input_list:
                        output = self.model(input_)
                        pre_count += torch.sum(output) 
            else:
                with torch.set_grad_enabled(False):
                    output = self.model(inputs)
                    pre_count = torch.sum(output)

            epoch_res.append(count[0].item() - pre_count.item() / self.args.log_param)
            # epoch_res.append(count[0].item())

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))

        logging.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, mse, mae, time.time() - epoch_start))

        model_state_dic = self.model.state_dict()
        if mae < self.best_mae:
            self.best_mse = mse
            self.best_mae = mae
            self.best_mae_at = self.epoch
            logging.info("SAVE best mse {:.2f} mae {:.2f} model @epoch {}".format(self.best_mse, self.best_mae, self.epoch))
            if self.args.save_all:
                torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.best_count)))
                self.best_count += 1
            else:
                torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model.pth'))

        logging.info("best mae {:.2f} mse {:.2f} @epoch {}".format(self.best_mae, self.best_mse, self.best_mae_at))

        if self.epoch is not None:
            wandb.log({'Val/bestMAE': self.best_mae,
                       'Val/MAE': mae,
                       'Val/MSE': mse,
                      }, step=self.epoch)


