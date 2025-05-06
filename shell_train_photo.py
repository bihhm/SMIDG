import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--domain", "-d", default="photo", help="Target")
parser.add_argument("--gpu", "-g", default=2, type=int, help="Gpu ID")
parser.add_argument("--times", "-t", default=10, type=int, help="Repeat times")
parser.add_argument("--root", default=None, type=str)

args = parser.parse_args()

###############################################################################
# 
source = ["photo", "cartoon", "art_painting", "sketch"]
target = args.domain
source.remove(target)
input_dir = '/home/q22301197/DG/CIRL/SMIDG/data/datalists_PACS'
output_dir = '/home/q22301197/DG/CIRL/SMIDG/train_resoult'
config = "PACS/ResNet50"
domain_name = target
path = os.path.join(output_dir, config.replace("/", "_"), domain_name)
##############################################################################

for i in range(args.times):
    seed = 2833798182657649885   #photo   97.37
    os.system(f'CUDA_VISIBLE_DEVICES={args.gpu} '
              f'python train_fft_new.py '
              f'--source {source[0]} {source[1]} {source[2]} '
              f'--target {target} '
              f'--input_dir {input_dir} '
              f'--output_dir {output_dir} '
              f'--seed {seed} '
              f'--config {config}',)

