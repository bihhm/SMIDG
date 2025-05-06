
config = {}

use_seed = False   #是否开启随机种子
seed=100  #   默认100

use_hx = True    #是否开启熵加权

batch_size = 16
epoch = 20
warmup_epoch = 5
# **********************************************************************
warmdown_epoch_ssl = 5
warmup_epoch_sup = 5
lam_const_ssl = 0.02   #自监督对比损失开始的系数
lam_const_sup = 0.05   #有监督对比损失结束的最大系数

tsl = 0.07
tcl = 0.2

alpa_dis_sup = 0.5   #最大化sup差异的系数
# ***********************************************************************
warmup_type = "sigmoid"
lr = 0.001
lr_decay_rate = 0.1
lam_const = 0.5   # loss weight for factorization loss


T = 10.0
k = 410

k_fft = 470

num_classes = 65

config["batch_size"] = batch_size
config["epoch"] = epoch
config["num_classes"] = num_classes
config["lam_const"] = lam_const
# ************************************************************************************
config["open_ssl_sup"] = True     #是否开启自监督和有监督系数增加和递减混合

config["lam_const_ssl"] = lam_const_ssl
config["lam_const_sup"] = lam_const_sup
config["warmdown_epoch_ssl"] = warmdown_epoch_ssl
config["warmup_epoch_sup"] = warmup_epoch_sup
config["warmup_epoch"] = warmup_epoch
config["warmup_type"] = warmup_type
config["warmdown_type"] = "sigmoid_down"

config["tsl"]=tsl
config["tcl"]=tcl
config["min_dis_sup"]=True   #是否最大化双路sup的差异

config["alpa_dis_sup"]=alpa_dis_sup
# *************************************************************************************

config["const_weight_dowm"] = False   #是否在总的对比损失上开启系数递减
config["T"] = T
config["k"] = k
config["k_fft"] = k_fft

config["seed"] = seed
config["use_seed"] = use_seed

config["use_hx"] = use_hx
config["unite_loss"] = False      #是否开启多损失尺度归一

# data configs
data_opt = {
    "image_size": 224,
    "use_crop": True,
    "jitter": 0.4,
    "from_domain": "all",
    "alpha": 0.2,
}

config["data_opt"] = data_opt

# inv configs
inv = {
    "inv_beta": 0.05,
    "features": 512,
    "inv_alpha": 0.01,
    "knn": 0,  #是否开始个体不变性损失的邻近样本（具体临近个数）
    "al_un":0.05,   #个体不变性损失的系数
    "un":False,   #是否开启个体不变性损失
    "un_epoch": 5  #个体不变性损失的训练轮数
    
}
config["inv"] = inv

#supcon configs
supcon = {
    "is_supcon": 1,    #是否开启对比损失
    "isexpand" : 1,   #是否开启数据增强
    "al_supcon": 0.5,  #有监督对比学习的系数
    "is_fac" : False,    #是否开启协方差损失
    "AM" : False,    #是否开启相位抖动
    "open_const_weight" : False , #是否开启损失系数递增
    "is_awl" : False   #是否开始损失系数的自动学习

}

config["supcon"] = supcon


# network configs
networks = {}

encoder = {
    "name": "resnet18",
}
networks["encoder"] = encoder

classifier = {
    "name": "base",
    "in_dim": 512,
    "num_classes": num_classes
}
networks["classifier"] = classifier

config["networks"] = networks


# optimizer configs
optimizer = {}

encoder_optimizer = {
    "optim_type": "sgd",
    "lr": lr,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "nesterov": True,
    "sched_type": "step",
    "lr_decay_step": 40,
    "lr_decay_rate": lr_decay_rate
}
optimizer["encoder_optimizer"] = encoder_optimizer

classifier_optimizer = {
    "optim_type": "sgd",
    "lr": 10*lr,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "nesterov": True,
    "sched_type": "step",
    "lr_decay_step": 40,
    "lr_decay_rate": lr_decay_rate
}
optimizer["classifier_optimizer"] = classifier_optimizer

config["optimizer"] = optimizer
