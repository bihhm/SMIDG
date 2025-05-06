config = {}

use_seed = False   #是否开启随机种子
seed=100  #   默认100

use_hx = True    #是否开启熵加权

batch_size = 16
epoch = 50
warmup_epoch = 5
warmup_type = "sigmoid"
lr = 0.001
lr_decay_rate = 0.1
lam_const = 0.05    # loss weight for factorization loss
T = 10.0
k = 1228

num_classes = 10

config["batch_size"] = batch_size
config["epoch"] = epoch
config["num_classes"] = num_classes
config["lam_const"] = lam_const
config["warmup_epoch"] = warmup_epoch
config["warmup_type"] = warmup_type
config["warmdown_type"] = "sigmoid_down"
config["const_weight_dowm"] = False
config["T"] = T
config["k"] = k

config["seed"] = seed
config["use_seed"] = use_seed

config["use_hx"] = use_hx
config["unite_loss"] = False

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
    "is_unsup" : False,  #是否开启有监督自监督超参数混合
    "is_fac" : False,    #是否开启协方差损失
    "AM" : False ,   #是否开启相位抖动
    "open_const_weight" : True , #是否开启损失系数递增
    "is_awl" : False
}

config["supcon"] = supcon


# network configs
networks = {}

encoder = {
    "name": "resnet50",
}
networks["encoder"] = encoder

classifier = {
    "name": "base",
    "in_dim": 2048,
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
    "lr_decay_step": int(epoch * 0.8),
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
    "lr_decay_step": int(epoch * 0.8),
    "lr_decay_rate": lr_decay_rate
}
optimizer["classifier_optimizer"] = classifier_optimizer

config["optimizer"] = optimizer
