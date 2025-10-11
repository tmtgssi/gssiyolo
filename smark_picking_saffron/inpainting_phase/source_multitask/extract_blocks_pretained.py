import os, torch, yaml, time
import colorful as c
from torch.utils.data.dataloader import DataLoader
from my_utils import set_seed, SSIM, PSNR, LPIPS, LPIPS_SET, save_img
from models.model import inpaint_model
from dataset import get_dataset
from collections import OrderedDict


# set seed
set_seed(1234)
gpu = torch.device("cuda")

# open config file
with open('./config/model_config.yml', 'r') as config:
    args = yaml.safe_load(config)

# Define the model
Inpaint_model = inpaint_model(args)

# Define the dataset
test_dataset = get_dataset(args['test_path'], test_mask_path=args['test_mask_1~60_path'],is_train=False, image_size=args['image_size'])

# set initial
iterations = 0
Total_time = []

data = torch.load(args['test_ckpt'])
Inpaint_model.load_state_dict(data['state_dict'],strict=False)
model = Inpaint_model.to(args['gpu'])

# Collect all modules with same input/output channels or usable for transfer
transferable_blocks = OrderedDict({
    "pad1": model.pad1,
    "conv1": model.conv1,
    "conv2": model.conv2,
    "conv3": model.conv3,
    "conv4": model.conv4,
    "RDB1": model.RDB1,
    "RDB2": model.RDB2,
    "RDB3": model.RDB3,
    "RDB4": model.RDB4,
    "GFF_1x1": model.GFF_1x1,
    "GFF_3x3": model.GFF_3x3,
    "convt1": model.convt1,
    "convt2": model.convt2,
    "convt3": model.convt3,
    "convt4": model.convt4,
    "batchNorm": model.batchNorm,
    "linear_to_384": model.linear_to_384,
})

# Save individual blocks (optional)
for name, module in transferable_blocks.items():
    torch.save(module.state_dict(), f"./ckpt/elements/{name}_pretrained.pth")

# Print transferable module names
print("✅ Transferable modules:")
for name in transferable_blocks.keys():
    print(name)

