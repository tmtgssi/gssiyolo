import os
import random, cv2, lpips
import numpy as np
import torch
from pytorch_msssim import ssim
import colorful as c


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
import os
import numpy as np
import torch
import cv2

# def Visualization_of_training_results(pred_img, input_img, input_mask, save_path, iterations):
#     # Process only the first 4 samples
#     batch_size = 4
#     input_img = input_img[:batch_size]
#     input_mask = input_mask[:batch_size]
#     pred_img = pred_img[:batch_size]

#     # Convert to (B, H, W, C) and scale to 0–255
#     input_img_np = (input_img.permute(0, 2, 3, 1).cpu().detach().numpy() * 255).astype(np.uint8)
#     pred_img_np = (pred_img.permute(0, 2, 3, 1).cpu().detach().numpy() * 255).astype(np.uint8)
#     input_mask_np = input_mask.permute(0, 2, 3, 1).cpu().detach().numpy()

#     # Ensure mask is binary (0 or 1), shape: (B, H, W, 1)
#     if input_mask_np.max() > 1.0:
#         input_mask_np = input_mask_np / 255.0

#     input_mask_np = (input_mask_np > 0.5).astype(np.uint8)

#     # Apply mask to input image: masked image = image * (1 - mask)
#     masked_img_np = input_img_np * (1 - input_mask_np)

#     # Concatenate vertically in batch, horizontally across types
#     original_concat = np.concatenate(input_img_np, axis=0)
#     masked_concat = np.concatenate(masked_img_np, axis=0)
#     pred_concat = np.concatenate(pred_img_np, axis=0)

#     # Final row: original, masked, prediction
#     output = np.concatenate([original_concat, masked_concat, pred_concat], axis=1)

#     # Ensure save directory exists
#     sample_dir = os.path.join(save_path, 'samples')
#     os.makedirs(sample_dir, exist_ok=True)

#     # Fix horizontal flip with ascontiguousarray
#     cv2.imwrite(os.path.join(sample_dir, f"{iterations}.jpg"), np.ascontiguousarray(output[:, :, ::-1]))

#     return None


def Visualization_of_training_results(pred_img, input_img, input_mask, save_path, iterations):

    # without edge
    current_img = input_img[:1, ...]
    current_img = current_img.permute(0, 2,3,1) * 255
    original_img = np.concatenate(current_img.cpu().detach().numpy().astype(np.uint8), axis=0)  # GT
    mask = input_mask[:1, ...].permute(0, 2, 3, 1)
    current_img = (current_img * (1 - mask)).cpu().detach().numpy().astype(np.uint8)
    current_img = np.concatenate(current_img, axis=0)  # GT with masks
    pred_img_output = pred_img[:1, :, :, :]
    pred_img_output = pred_img_output.permute(0, 2, 3, 1) * 255
    pred_img_output = np.concatenate(pred_img_output.cpu().detach().numpy().astype(np.uint8), axis=0)  # pred_img

    output = np.concatenate([original_img, current_img, pred_img_output],
                            axis=1)  # GT + GT with mask + pred_img

    save_path = save_path + '/samples'
    os.makedirs(save_path, exist_ok=True)
    cv2.imwrite(save_path + '/' + str(iterations) + '.jpg', output[:, :, ::-1])

    return None

def PSNR(GT, Pred):
    mse = torch.mean((GT - Pred) ** 2)
    PSNR = 20 * torch.log10(1.0 / torch.sqrt(mse))     #https://github.com/bonlime/pytorch-tools/blob/master/pytorch_tools/metrics/psnr.py
    return PSNR

def SSIM(GT, Pred):
    SSIM = ssim(GT, Pred, data_range=1.0, size_average=True)    #https://github.com/VainF/pytorch-msssim
    return SSIM

def LPIPS_SET():
    loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
    #loss_fn_vgg = lpips.LPIPS(net='vgg')  # closer to "traditional" perceptual loss, when used for optimization
    return loss_fn_alex

def LPIPS(GT, Pred, alex):
    # LPIPS = loss_fn_vgg(GT, Pred,)  #https://pypi.org/project/lpips/
    LPIPS = alex(GT, Pred)  #https://pypi.org/project/lpips/
    LPIPS = LPIPS.mean()
    return LPIPS

def FID(GT, Pred):
    pass
    return FID  # noused

def save_img(Pred, save_img_path, name):
    for n in range(Pred.shape[0]):
        os.makedirs(save_img_path, exist_ok=True)
        pre_img = Pred[n:n+1, ...]
        pre_img = pre_img.permute(0, 2, 3, 1) * 255
        names = name[n]
        pre_img = np.concatenate(pre_img.cpu().detach().numpy().astype(np.uint8), axis=0)  # GT
        cv2.imwrite(save_img_path+str(names), pre_img[:, :, ::-1])
        print(c.magenta(save_img_path+str(names)))




