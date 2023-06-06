import torch
import os
import argparse
from utils import dataparallel
import scipy.io as sio
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import imgvision as iv
import nm
import cv2

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="PyTorch HSIFUSION")
parser.add_argument('--data_path', default='/home/czy/NET/spectral/OLU_net_1/datasets/TSA_real_data/Measurements/', type=str,help='path of data')
parser.add_argument('--mask_path', default='/home/czy/NET/spectral/OLU_net_1/datasets/TSA_real_data/mask.mat', type=str,help='path of mask')
parser.add_argument("--size", default=660, type=int, help='the size of trainset image')
parser.add_argument("--trainset_num", default=2000, type=int, help='total number of trainset')
parser.add_argument("--testset_num", default=5, type=int, help='total number of testset')
parser.add_argument("--seed", default=1, type=int, help='Random_seed')
parser.add_argument("--batch_size", default=1, type=int, help='batch_size')
parser.add_argument("--isTrain", default=False, type=bool, help='train or test')
parser.add_argument("--pretrained_model_path", default=None, type=str)
opt = parser.parse_args()
print(opt)

def getRGB(dWave,maxPix=1,gamma=1):
    waveArea = [380,440,490,510,580,645,780]
    minusWave = [0,440,440,510,510,645,780]
    deltWave = [1,60,50,20,70,65,35]
    for p in range(len(waveArea)):
        if dWave<waveArea[p]:
            break
    pVar = abs(minusWave[p]-dWave)/deltWave[p]
    rgbs = [[0,0,0],[pVar,0,1],[0,pVar,1],[0,1,pVar],
            [pVar,1,0],[1,pVar,0],[1,0,0],[0,0,0]]
        #在光谱边缘处颜色变暗
    if (dWave>=380) & (dWave<420):
        alpha = 0.3+0.7*(dWave-380)/(420-380)
    elif (dWave>=420) & (dWave<701):
        alpha = 1.0
    elif (dWave>=701) & (dWave<780):
        alpha = 0.3+0.7*(780-dWave)/(780-700)
    else:
        alpha = 0       #非可见区
    return [maxPix*(c*alpha)**gamma for c in rgbs[p]]

def get_image(hyper_images, i, expand_numb=1):

    img = hyper_images[:,:,i]
    h, w =img.shape
    IMG = np.ones([h,w,3])

    R = getRGB(nm.spectral_wave[i])[0]
    G = getRGB(nm.spectral_wave[i])[1]
    B = getRGB(nm.spectral_wave[i])[2]

    IMG[:, :, 0] = B * img
    IMG[:, :, 1] = G * img
    IMG[:, :, 2] = R * img
    IMG = IMG * expand_numb
    return IMG

def prepare_data(path, file_num):
    HR_HSI = np.zeros((((660,714,file_num))))
    for idx in range(file_num):
        ####  read HrHSI
        path1 = os.path.join(path) + 'scene' + str(idx+1) + '.mat'
        data = sio.loadmat(path1)
        HR_HSI[:,:,idx] = data['meas_real']
        plt.imshow(HR_HSI[:,:,idx])
        plt.show()
        HR_HSI[HR_HSI < 0] = 0.0
        HR_HSI[HR_HSI > 1] = 1.0
    return HR_HSI

def load_mask(path,size=660):
    ## load mask
    data = sio.loadmat(path)
    mask = data['mask']
    mask_3d = np.tile(mask[:, :, np.newaxis], (1, 1, 28))
    mask_3d_shift = np.zeros((size, size + (28 - 1) * 2, 28))
    mask_3d_shift[:, 0:size, :] = mask_3d
    for t in range(28):
        mask_3d_shift[:, :, t] = np.roll(mask_3d_shift[:, :, t], 2 * t, axis=1)
    mask_3d_shift_s = np.sum(mask_3d_shift ** 2, axis=2, keepdims=False)
    mask_3d_shift_s[mask_3d_shift_s == 0] = 1
    mask_3d_shift = torch.FloatTensor(mask_3d_shift.copy()).permute(2, 0, 1)
    mask_3d_shift_s = torch.FloatTensor(mask_3d_shift_s.copy())
    return mask_3d_shift.unsqueeze(0), mask_3d_shift_s.unsqueeze(0)

HR_HSI = prepare_data(opt.data_path, 5)
mask_3d_shift, mask_3d_shift_s = load_mask('/home/czy/NET/spectral/OLU_net_1/datasets/TSA_real_data/mask.mat')

pretrained_model_path = "/home/czy/NET/spectral/OLU_net_1/real/train_code/exp/OLU_7stg/2023_01_18_20_06_41/model_040.pth"
# pretrained_model_path = "/home/czy/NET/spectral/WST/real/train_code/exp/dauhst_9stg/2022_12_20_15_33_44/model_263.pth"
model = torch.load(pretrained_model_path)
model = model.eval()
model = dataparallel(model, 1)
psnr_total = 0
k = 0
save_path = './Results/daust/'

for j in range(5):
    with torch.no_grad():
        meas = HR_HSI[:,:,j]
        meas = meas / meas.max() * 0.8
        meas = torch.FloatTensor(meas)
        # meas = torch.FloatTensor(meas).unsqueeze(2).permute(2, 0, 1)
        input = meas.unsqueeze(0)
        input = Variable(input)
        input = input.cuda()
        mask_3d_shift = mask_3d_shift.cuda()
        mask_3d_shift_s = mask_3d_shift_s.cuda()
        input_mask = (mask_3d_shift, mask_3d_shift_s)
        out = model(input, input_mask)
        result = out
        result = result.clamp(min=0., max=1.)
    k = k + 1
    if not os.path.exists(save_path):  # Create the model directory if it doesn't exist
        os.makedirs(save_path)
    res = result.cpu().permute(2,3,1,0).squeeze(3).numpy()
    # save_file = save_path + f'{j}.mat'
    # sio.savemat(save_file, {'res':res})
    convertor = iv.spectra(illuminant='D50', band=nm.spectral_wave)
    Image = convertor.space(res, space='srgb')
    plt.imshow(Image)
    plt.show()

    WAVE = [12, 14, 22, 24, 27]
    i = 0
    images = get_image(res, WAVE[i])
    im_rgb = images
    im_rgb = im_rgb.astype(np.float32)
    im_rgb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2BGR)

    plt.imshow(im_rgb)
    plt.show()

