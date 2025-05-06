import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--domain", "-d", default="sketch", help="Target")
parser.add_argument("--gpu", "-g", default=2, type=int, help="Gpu ID")
parser.add_argument("--times", "-t", default=1, type=int, help="Repeat times")
parser.add_argument("--root", default=None, type=str)

args = parser.parse_args()

###############################################################################
# 
source = ["photo", "cartoon", "art_painting", "sketch"]

# source =["Art", "Clipart", "Product", "Real_World"]
# source =["location_38", "location_43", "location_46", "location_100"]
# source =  ["mnist", "mnist_m", "svhn", "syn"]

# source = ["Caltech101","LabelMe","SUN09","VOC2007"]
# source = ["clipart","infograph","painting","quickdraw","real","sketch"]
target = args.domain
source.remove(target)

# input_dir = '/home/q22301197/DG/CIRL/CIRL_main/data/datalists_digits_dg'
# input_dir = '/home/q22301197/DG/CIRL/CIRL_main/data/datalists_office_home'
input_dir = '/home/q22301197/DG/CIRL/CIRL_main/data/datalists_PACS'

# input_dir = '/home/q22301197/DG/CIRL/CIRL_main/data/datalists_VLCS'
# input_dir = '/home/q22301197/DG/CIRL/CIRL_main/data/datalists_Domain_net'
# input_dir = '/home/q22301197/DG/CIRL/CIRL_main/data/datalists_Te'
output_dir = '/home/q22301197/DG/CIRL/CIRL_main/train_resoult'

config = "PACS/ResNet50"
# config = "Office-Home/ResNet18"
# config = "Digits-DG/cnn"
# config = "PACS/ResNet50"
# config = "VLCS/ResNet18"
# config = "VLCS/ResNet50"
# config = "Domain_net/ResNet50"
# config = "TerraIncognita/ResNet50"


domain_name = target
path = os.path.join(output_dir, config.replace("/", "_"), domain_name)
##############################################################################

for i in range(args.times):
    # seed = 2833798182657649885   #photo   97.37
    # seed = 100
    # seed = 16035547538127746815   #RealWorld
    # seed = 4807980965478880571  #RealWorld 76.77
    # seed = 9948908824056498087   # mnist_m 
    # seed = 16151362751254617599  #clipart  3852658311212743716
    # seed = 10439163910065408216     #clipart
    # seed = 2615214830236913096   #clipart
    # seed =4452975344699044963   #mnist_m
    # seed = 4896750259841287514  # Art   61.06%
    # seed = 12635349700579447737  #  Art  62.05%
    # seed = 10159642149355315305 #  art_painting
    # seed = 8032608406203760444
    seed = 12635349700579447737   #Art

    os.system(f'CUDA_VISIBLE_DEVICES={args.gpu} '
              f'python train_fft_new.py '
            #   f'--source {source[0]} {source[1]} {source[2]} {source[3]} {source[4]} '
              f'--source {source[0]} {source[1]} {source[2]} '
              f'--target {target} '
              f'--input_dir {input_dir} '
              f'--output_dir {output_dir} '
              f'--seed {seed} '
              f'--config {config}',)

