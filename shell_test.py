import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--domain", "-d", default="cartoon", help="Target")
parser.add_argument("--gpu", "-g", default=2, type=int, help="Gpu ID")
parser.add_argument("--times", "-t", default=1, type=int, help="Repeat times")

args = parser.parse_args()

###############################################################################
source = ["photo", "cartoon", "art_painting", "sketch"]
# source =["Art", "Clipart", "Product", "Real_World"]
# source =  ["mnist", "mnist_m", "svhn", "syn"]
target = args.domain
source.remove(target)

# input_dir = '/home/q22301197/DG/CIRL/CIRL_main/data/datalists_office_home'
input_dir = '/home/q22301197/DG/CIRL/CIRL_main/data/datalists_PACS'
output_dir = '/home/q22301197/DG/CIRL/CIRL_main/test_resoult'

config = "PACS/ResNet18"
# config = "Office-Home/ResNet18"

domain_name = target
path = os.path.join(output_dir, config.replace("/", "_"), domain_name)
# ckpt_path = "/home/q22301197/DG/CIRL/CIRL_main/train_resoult/PACS_ResNet18/photo/2023-10-06-23-52-49/best_model.tar"
# ckpt_path = "/home/q22301197/DG/CIRL/CIRL_main/train_resoult/PACS_ResNet18/art_painting/2023-10-04-23-34-21/best_model.tar"
ckpt_path = "/home/q22301197/DG/CIRL/CIRL_main/train_resoult/PACS_ResNet18/cartoon/2023-10-03-19-11-53/best_model.tar"   #cartoon 81.14%
# ckpt_path = "/home/q22301197/DG/CIRL/CIRL_main/train_resoult/PACS_ResNet18/cartoon/2023-10-02-10-13-27/best_model.tar"   #cartoon   76.15%
# ckpt_path = "/home/q22301197/DG/CIRL/CIRL_main/train_resoult/PACS_ResNet18/cartoon/2023-10-02-14-47-11/best_model.tar"  #cartoon 79.23%
# ckpt_path = "/home/q22301197/DG/CIRL/CIRL_main/train_resoult/PACS_ResNet18/art_painting/2023-10-06-21-04-19/best_model.tar" #art 83.36%
# ckpt_path = "/home/q22301197/DG/CIRL/CIRL_main/train_resoult/PACS_ResNet18/cartoon/2023-10-06-21-00-04/best_model.tar" #cartoon 82.12%
# ckpt_path = "/home/q22301197/DG/CIRL/CIRL_main/train_resoult/PACS_ResNet18/cartoon/2024-01-09-12-30-02/4-0.tar"      #DeepAll
# ckpt_path = "/home/q22301197/DG/CIRL/CIRL_main/train_resoult/PACS_ResNet18/cartoon/2024-01-09-12-30-02/42-0.tar"
# ckpt_path = "/home/q22301197/DG/CIRL/CIRL_main/train_resoult/PACS_ResNet18/cartoon/2024-01-09-12-30-02/10-0.tar"
# ckpt_path = "/home/q22301197/DG/CIRL/CIRL_main/train_resoult/PACS_ResNet18/cartoon/2023-10-11-19-56-24/best_model.tar"      #cartoon 77  96
# ckpt_path = "/home/q22301197/DG/CIRL/CIRL_main/train_resoult/PACS_ResNet18/cartoon/2024-01-08-21-47-10/7-0.tar"  #cartoon 74


# ckpt_path = '/home/q22301197/DG/CIRL/CIRL_main/train_resoult/PACS_ResNet18/cartoon/2024-01-30-00-03-23/0-0.tar'
##############################################################################

for i in range(args.times):
    os.system(f'CUDA_VISIBLE_DEVICES={args.gpu} '
              f'python test.py '
              f'--source {source[0]} {source[1]} {source[2]} '
              f'--target {target} '
              f'--input_dir {input_dir} '
              f'--output_dir {output_dir} '
              f'--config {config} '
              f'--ckpt {ckpt_path}')
