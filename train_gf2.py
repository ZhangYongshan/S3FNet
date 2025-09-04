import argparse
import json
import os

from utils.load_train_data import Dataset_Pro
from utils.metrics import SAM, ERGAS

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.loss import cc
from utils.loss import cc_channel
from tqdm import tqdm
import pandas as pd

'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']
# torch.autograd.set_detect_anomaly(True)
criteria_fusion = nn.L1Loss(reduction='mean')

# my args
parser = argparse.ArgumentParser()
from model.s3fnet import net
parser.add_argument('--model_name', type=str, default='s3fnet')
parser.add_argument('--results_dir', type=str, default='../outputs/output_gf2')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--coeffs_level', type=int, default=(0.5, 0.3, 0.2))
parser.add_argument('--resnet_block_nums', type=int, default=(1, 1, 1), help='number of encoder_resnet_blocks')
parser.add_argument('--restormer_block_nums', type=int, default=(2, 2, 2), help='number of encoder_restormer_blocks')
parser.add_argument('--use_attention', type=bool, default=(1, 0, 0, 1),
                    help='whether use ms_use_ca, ms_use_sa, pan_use_ca,pan_use_sa')
parser.add_argument('--s3block_deeps', type=int, default=(1, 1, 1))
parser.add_argument('--coeff_decomp', type=float, default=0.5)
parser.add_argument('--fenzi', type=float, default=0.001,
                    help='在计算cc时的分子，目的是控制让相似度尽可能小，但默认先为0')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=.0)
parser.add_argument('--clip_grad_norm_value', type=float, default=0.01)
parser.add_argument('--optim_step', type=int, default=100, help='Optimizer step size')
parser.add_argument('--optim_gamma', type=float, default=0.5, help='Optimizer gamma value')
parser.add_argument('--ckpt', type=int, default=10, help='Checkpoint')
parser.add_argument('--num_epochs', type=int, default=300)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--train_data_path', type=str,
                    default='/home/xiongjz/xjzwork/dataset/wv3/training_wv3/train_wv3.h5',
                    help='Path of the training dataset.')
parser.add_argument('--val_data_path', type=str,
                    default='/home/xiongjz/xjzwork/dataset/wv3/training_wv3/valid_wv3.h5',
                    help='Path of the validation dataset.')

args = parser.parse_args()

# Model
# model = nn.DataParallel(net(ms_dim=8, resnet_block_nums=args.resnet_block_nums,
#                             restormer_block_nums=args.restormer_block_nums,
#                             use_attention=args.use_attention).to(args.device))
model = net(ms_dim=4, resnet_block_nums=args.resnet_block_nums, restormer_block_nums=args.restormer_block_nums,
            use_attention=args.use_attention).to(args.device)

# optimizer, scheduler and loss function
optimizer = torch.optim.Adam(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.optim_step, gamma=args.optim_gamma)

MSELoss = nn.MSELoss()
L1Loss = nn.L1Loss()


# Loss_ssim = kornia.losses.SSIM(11, reduction='mean')


def prepare_training_data(args):
    train_set = Dataset_Pro(args.train_data_path)
    validate_set = Dataset_Pro(args.val_data_path)
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=args.batch_size,
                                      shuffle=True, pin_memory=True, drop_last=True)
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=args.batch_size,
                                      shuffle=True, pin_memory=True, drop_last=True)
    return training_data_loader, validate_data_loader


training_data_loader, validate_data_loader = prepare_training_data(args)

# batch = next(iter(training_data_loader))
#
# from torchsummary import summary
# summary(model)

timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''

torch.backends.cudnn.benchmark = True
prev_time = time.time()


def train(model, train_data_loader, optimizer, epoch, args):
    model.train()
    total_train_loss = .0
    total_train_fusion_loss = .0
    train_bar = tqdm(train_data_loader, ncols=200)
    for batch in train_bar:
        gt, pan, ms = batch[0].to(args.device), batch[3].to(args.device), batch[4].to(args.device)

        optimizer.zero_grad()

        img_Fuse, features_M_B, features_M_D, features_P_B, features_P_D = model(ms, pan)

        # cc_losses_B = [cc(features_P_B, features_M_B) for features_P_B, features_M_B in zip(features_P_B, features_M_B)]
        # cc_losses_D = [cc(features_P_D, features_M_D) for features_P_D, features_M_D in zip(features_P_D, features_M_D)]

        cc_losses_pan = [cc(features_P_B, features_P_D) for features_P_B, features_P_D in
                         zip(features_P_B, features_P_D)]
        cc_losses_ms = [cc_channel(features_M_B, features_M_D) for features_M_B, features_M_D in
                        zip(features_M_B, features_M_D)]

        loss_decomp = .0
        for i, (cc_loss_pan, cc_loss_ms) in enumerate(zip(cc_losses_pan, cc_losses_ms)):
            loss_decomp += (args.coeffs_level[i] * args.fenzi * 0.5) / (1.01 + cc_loss_pan) + (
                        args.coeffs_level[i] * args.fenzi*0.5) / (1.01 + cc_loss_ms)

        # 损失函数的计算修改为有监督的方式
        fusion_loss = criteria_fusion(img_Fuse, gt)
        train_loss = fusion_loss + args.coeff_decomp * loss_decomp
        train_loss.backward()
        nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=args.clip_grad_norm_value, norm_type=2)

        optimizer.step()

        total_train_loss += train_loss.item()
        total_train_fusion_loss += fusion_loss.item()

        train_bar.set_description(
            'Train Epoch:[{}/{}], lr:{:.6f}, Loss:{:.6f}, Fusion Loss:{:.6f}, cc_losses_ms:{:.6f}, cc_losses_pan:{:.6f}'.format(
                epoch,
                args.num_epochs,
                optimizer.param_groups[
                    0][
                    'lr'],
                train_loss.item(),
                fusion_loss.item(),
                cc_losses_ms[
                    0],
                cc_losses_pan[
                    0]))

    return total_train_loss / len(train_data_loader), total_train_fusion_loss / len(train_data_loader)


def val(model, val_data_loader, epoch, args):
    model.eval()  # 将模型设为评估模式
    total_val_loss = .0
    total_val_fusion_loss = .0
    total_sam = .0
    total_ergas = .0

    with torch.no_grad():  # 测试阶段不需要计算梯度
        val_bar = tqdm(val_data_loader, ncols=200)
        for batch in val_bar:
            gt, pan, ms = batch[0].to(args.device), batch[3].to(args.device), batch[4].to(args.device)
            img_Fuse, features_MB, features_MD, features_PB, features_PD = model(ms, pan)

            # 计算损失
            fusion_loss = criteria_fusion(img_Fuse, gt)
            cc_losses_B = [cc(features_PB, features_MB) for features_PB, features_MB in
                           zip(features_PB, features_MB)]
            cc_losses_D = [cc(features_PD, features_MD) for features_PD, features_MD in
                           zip(features_PD, features_MD)]
            cc_losses_PD_PB = [cc(features_PD, features_PB) for features_PD, features_PB in
                               zip(features_PD, features_PB)]
            cc_losses_PD_MB = [cc(features_PD, features_MB) for features_PD, features_MB in
                               zip(features_PD, features_MB)]
            cc_losses_MD_MB = [cc(features_MD, features_MB) for features_MD, features_MB in
                               zip(features_MD, features_MB)]
            cc_losses_MD_PB = [cc(features_MD, features_PB) for features_MD, features_PB in
                               zip(features_MD, features_PB)]

            cc_losses_pan = [cc(features_P_B, features_P_D) for features_P_B, features_P_D in
                             zip(features_PB, features_PD)]
            cc_losses_ms = [cc_channel(features_M_B, features_M_D) for features_M_B, features_M_D in
                            zip(features_MB, features_MD)]

            loss_decomp = .0
            for i, (cc_loss_pan, cc_loss_ms) in enumerate(zip(cc_losses_pan, cc_losses_ms)):
                loss_decomp += (args.coeffs_level[i] * args.fenzi * 0.5) / (1.01 + cc_loss_pan) + (
                        args.coeffs_level[i] * args.fenzi * 0.5) / (1.01 + cc_loss_ms)

            # 统计损失
            val_loss = fusion_loss.item() + args.coeff_decomp * loss_decomp
            total_val_loss += val_loss
            total_val_fusion_loss += fusion_loss.item()
            sam = SAM(img_Fuse, gt)
            total_sam += sam
            ergas = ERGAS(img_Fuse, gt, 4)
            total_ergas += ergas

            val_bar.set_description(
                'Val Epoch:[{}/{}], lr:{:.6f}, Loss:{:.6f}, Fusion Loss:{:.6f}, SAM:{:.6f}, ERGAS:{:.6f}, '
                'cc_losses_ms:{:.6f}, cc_losses_pan:{:.6f}'.format(
                    epoch,
                    args.num_epochs,
                    optimizer.param_groups[
                        0][
                        'lr'],
                    val_loss.item(),
                    fusion_loss.item(),
                    sam,
                    ergas,
                    cc_losses_ms[
                        0],
                    cc_losses_pan[
                        0]))

    return total_val_loss.item() / len(val_data_loader), total_val_fusion_loss / len(
        val_data_loader), total_sam.item() / len(
        val_data_loader), total_ergas.item() / len(
        val_data_loader), cc_losses_B, cc_losses_D, cc_losses_PD_PB, cc_losses_PD_MB, cc_losses_MD_MB, cc_losses_MD_PB, cc_losses_ms, cc_losses_pan


if not os.path.exists(args.results_dir):
    os.mkdir(args.results_dir)
with open(args.results_dir + '/args.json', 'w') as fid:
    json.dump(args.__dict__, fid, indent=2)
results = {'train_fusion_loss': [], 'val_fusion_loss': [], 'SAM': [], 'ERGAS': [], 'lr': [], 'cc_losses_B': [],
           'cc_losses_D': [], 'cc_losses_PD_PB': [], 'cc_losses_PD_MB': [], 'cc_losses_MD_MB': [],
           'cc_losses_MD_PB': [], 'cc_losses_ms': [], 'cc_channel_losses_pan': []}


for epoch in range(1, args.num_epochs + 1):
    train_loss, train_fusion_loss = train(model, training_data_loader, optimizer, epoch, args)
    val_loss, val_fusion_loss, sam, ergas, cc_losses_B, cc_losses_D, cc_losses_PD_PB, cc_losses_PD_MB, cc_losses_MD_MB, cc_losses_MD_PB, cc_losses_ms, cc_losses_pan = val(
        model, validate_data_loader, epoch, args)
    results['train_fusion_loss'].append(train_fusion_loss)
    results['val_fusion_loss'].append(val_fusion_loss)
    results['SAM'].append(sam)
    results['ERGAS'].append(ergas)
    results['lr'].append(optimizer.param_groups[0]['lr'])
    results['cc_losses_D'].append(cc_losses_D)
    results['cc_losses_B'].append(cc_losses_B)
    results['cc_losses_PD_PB'].append(cc_losses_PD_PB)
    results['cc_losses_PD_MB'].append(cc_losses_PD_MB)
    results['cc_losses_MD_MB'].append(cc_losses_MD_MB)
    results['cc_losses_MD_PB'].append(cc_losses_MD_PB)
    results['cc_losses_ms'].append(cc_losses_ms)
    results['cc_channel_losses_pan'].append(cc_losses_pan)

    # save statistics
    data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
    data_frame.to_csv(args.results_dir + '/train_log.csv', index_label='epoch')
    scheduler.step()

    if optimizer.param_groups[0]['lr'] <= 1e-6:
        optimizer.param_groups[0]['lr'] = 1e-6

    if (epoch) % args.ckpt == 0:
        checkpoint_name = f'{args.model_name}e{epoch:04}.pth'
        # save model
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), },
                   args.results_dir + '/' + checkpoint_name)

