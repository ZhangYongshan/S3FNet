import argparse
import logging
import os
import warnings

import scipy.io as sio
import torch

from utils.load_test_data import load_h5py_with_hp
from utils.loss import cc, cc_channel

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cuda'
# config
from models.ablation.cdmmunet_base.cdmmunet import net

scale = 'fr'
model_names = ('s3fnet',)
epochs = (30,40,50,60,70,80,210)
#epochs = (50, 100,120, 150,180, 200, 210, 220, 250)

for model_name in model_names:
    for epoch in epochs:
        parser = argparse.ArgumentParser()

        parser.add_argument('--save_dir', type=str,
                            default='result2mat/output_gf2_' + (
                                'fr' if scale == 'fr' else 'rr') + f'_{model_name}e{epoch}')
        parser.add_argument('--ckpt_path', type=str,
                            default=f'outputs/output_gf2_alpha/{model_name}/{model_name}e{epoch:04}.pth',
                            help='模型参数保存地址')
        parser.add_argument('--channels', type=int, default=32, help='Feature channels')
        # 在远程服务器上时地址
        # parser.add_argument('--file_path', type=str,
        #                     default=(
        #                         '/home/xiongjz/xjzwork/pycharm_tmp/gf2_test_h5/test_gf2_multiExm1.h5' if scale == 'rr' else '/home/xiongjz/xjzwork/pycharm_tmp/gf2_test_h5/test_gf2_OrigScale_multiExm1.h5'),
        #                     help='Absolute path of the test file (in h5 format).')
        # 在本机运行时
        parser.add_argument('--file_path', type=str,
                            default=(
                                r"/home/xiongjiazhuang/dataset/gf2/test_gf2/reduced_examples/test_gf2_multiExm1.h5" if scale == 'rr' else r"/home/xiongjiazhuang/dataset/gf2/test_gf2/full_examples/test_gf2_OrigScale_multiExm1.h5"),
                            help='Absolute path of the test file (in h5 format).')

        args = parser.parse_args()

        model = net(ms_dim=4, resnet_block_nums=(1, 1, 1), use_attention=(1, 0, 0, 1),
                    restormer_block_nums=(2, 2, 2)).to(device)
        print(torch.load(args.ckpt_path)['state_dict'].keys())
        print(model.state_dict().keys())
        model.load_state_dict(torch.load(args.ckpt_path)['state_dict'])
        # from torchsummary import summary
        #
        # summary(model)

        model.eval()
        ms, _, pan, _ = load_h5py_with_hp(args.file_path)

        # get size
        image_num, C, h, w = ms.shape
        _, _, H, W = pan.shape
        cut_size = 64  # must be divided by 4, we recommand 64
        ms_size = cut_size // 4


        def create_folder(file_name):
            folder_name = os.path.splitext(file_name)[0]  # 获取文件名（不包含扩展名）

            if os.path.exists(folder_name):  # 如果文件夹已经存在
                print(f"Folder {folder_name} already exists. Deleting...")
                try:
                    os.rmdir(folder_name)  # 删除文件夹
                    print(f"Deleted folder {folder_name}.")
                except OSError as e:
                    print(f"Error: {folder_name} : {e.strerror}")

            try:
                os.mkdir(folder_name)  # 创建新的文件夹
                print(f"Created folder {folder_name}.")
            except OSError as e:
                print(f"Error: {folder_name} : {e.strerror}")


        create_folder(args.save_dir)  # 创造保存的文件夹

        for k in range(image_num):
            with torch.no_grad():
                x1, x2 = ms[k, :, :, :].to(device), pan[k, 0, :, :].to(device)

                x1 = x1.unsqueeze(dim=0).float()
                x2 = x2.unsqueeze(dim=0).unsqueeze(dim=1).float()
                output = torch.zeros(1, C, H, W).to(device)

                scale_H = H // cut_size
                scale_W = W // cut_size
                for i in range(scale_H):
                    for j in range(scale_W):
                        MS = x1[:, :, i * ms_size: (i + 1) * ms_size,
                             j * ms_size: (j + 1) * ms_size]
                        PAN = x2[:, :, i * cut_size: (i + 1) * cut_size,
                              j * cut_size: (j + 1) * cut_size]
                        # 对ms和pan进行预处理

                        img_Fuse, spa_feature, spe_feature, fuse_feature = model(MS, PAN)

                        # 计算损失
                        spa_cc = [cc(spa_feature, fuse_feature) for spa_feature, fuse_feature in
                                  zip(spa_feature, fuse_feature)]
                        spe_cc = [cc_channel(spe_feature, fuse_feature) for spe_feature, fuse_feature in
                                  zip(spe_feature, fuse_feature)]

                        sr = torch.clamp(img_Fuse, 0, 1)
                        output[:, :, i * cut_size: (i + 1) * cut_size, j * cut_size: (j + 1) * cut_size] = \
                            sr * 2047.

                output = torch.squeeze(output).permute(1, 2, 0).cpu().detach().numpy()  # HxWxC
                save_name = os.path.join(args.save_dir,
                                         'output_' + ('os_' if scale == 'fr' else '') + 'mulExm_' + str(k + 1) + '.mat')
                sio.savemat(save_name, {'sr': output})

matlab_str = '{'
for model_name in model_names:
    for epoch in epochs:
        matlab_str += f'\'{model_name}e{epoch}\''
        if model_name is not model_names[-1] or epoch is not epochs[-1]:
            matlab_str += ','
        else:
            matlab_str += '};'
print(matlab_str)
