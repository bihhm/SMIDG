import argparse
import ast
from collections import deque
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from models.model_factory import *
from optimizer.optimizer_helper import get_optim_and_scheduler
from data import *
from utils.Logger import Logger
from utils.tools import *
from models.classifier import Masker
from models.classifier import Masker_FFT

from utils.invarice import InvNet
from utils.supcon import SupConLoss , AutomaticMetricLoss, collect_feature
import time

import warnings
warnings.filterwarnings("ignore")
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=available_datasets, help="Target")
    parser.add_argument("--seed", default=0, help="The seed")
    parser.add_argument("--input_dir", default=None, help="The directory of dataset lists")
    parser.add_argument("--dataset", default="PACS", help="The directory of dataset lists")
    parser.add_argument("--output_dir", default=None, help="The directory to save logs and models")
    parser.add_argument("--config", default=None, help="Experiment configs")
    parser.add_argument("--tf_logger", type=ast.literal_eval, default=True, help="If true will save tensorboard compatible logs")
    args = parser.parse_args()
    config_file = "config." + args.config.replace("/", ".")
    print(f"\nLoading experiment {args.config}\n")
    config = __import__(config_file, fromlist=[""]).config

    return args, config


class Trainer:
    def __init__(self, args, config, device):
        config["seed"] = args.seed
        self.args = args
        self.config = config
        self.device = device
        self.global_step = 0


        
        # networks
        self.encoder = get_encoder_from_config(self.config["networks"]["encoder"]).to(device)
        self.classifier = get_classifier_from_config(self.config["networks"]["classifier"]).to(device)
        self.classifier_ad = get_classifier_from_config(self.config["networks"]["classifier"]).to(device)
        dim = self.config["networks"]["classifier"]["in_dim"]
        self.masker = Masker(in_dim=dim, num_classes = dim,middle = 4*dim,k=self.config["k"]).to(device)
        
        self.masker_fft = Masker_FFT(self.device,in_dim=self.config["num_groups"]*7*7*2, num_classes = self.config["num_groups"]*7*7*2,middle = self.config["num_groups"]*7*7*4*2,k=self.config["k_fft"],num_groups=self.config["num_groups"]).to(device)

        auxiliary_weight = 0.2
        self.awl_global = AutomaticMetricLoss(num=1, init_weight=0.2, min_weights=[auxiliary_weight,auxiliary_weight])

       
        # optimizers
        self.encoder_optim, self.encoder_sched = \
            get_optim_and_scheduler(self.encoder, self.config["optimizer"]["encoder_optimizer"])
        self.classifier_optim, self.classifier_sched = \
            get_optim_and_scheduler(self.classifier, self.config["optimizer"]["classifier_optimizer"])
        self.classifier_ad_optim, self.classifier_ad_sched = \
            get_optim_and_scheduler(self.classifier_ad, self.config["optimizer"]["classifier_optimizer"])
        self.masker_optim, self.masker_sched = \
            get_optim_and_scheduler(self.masker, self.config["optimizer"]["classifier_optimizer"])
        self.masker_fft_optim, self.masker_fft_sched = \
            get_optim_and_scheduler(self.masker_fft, self.config["optimizer"]["classifier_optimizer"])
        self.awl_global_optim, self.awl_global_sched = \
            get_optim_and_scheduler(self.masker, self.config["optimizer"]["classifier_optimizer"])

        # dataloaders
        self.train_loader,num_sum = get_fourier_train_dataloader(args=self.args, config=self.config)
        self.val_loader = get_val_dataloader(args=self.args, config=self.config)
        self.test_loader = get_test_loader(args=self.args, config=self.config)
        self.eval_loader = {'val': self.val_loader, 'test': self.test_loader}


        self.features = self.config["inv"]["features"]
        self.inv_beta = self.config["inv"]["inv_beta"]
        self.knn = self.config["inv"]["knn"]
        self.inv_alpha = self.config["inv"]["inv_alpha"]
        model_inv = InvNet(self.features, num_sum,
                            beta=self.inv_beta, knn=self.knn,
                            alpha=self.inv_alpha)
        self.model_inv = model_inv

        self.supcon_loss = SupConLoss()

        self.al_supcon = self.config["supcon"]["al_supcon"]

        self.al_un = self.config["inv"]["al_un"]
        self.un = self.config["inv"]["un"]
        self.un_epoch = self.config["inv"]["un"]

        self.open_const_weight = self.config["supcon"]["open_const_weight"]

    def _do_epoch(self):
        criterion = nn.CrossEntropyLoss()
        

        # turn on train mode

        # all_features = []
        # all_labels = []
        # feature, labels = collect_feature(self.train_loader, self.encoder, self.device)
        # task = self.args.target
        # # ttt(self.current_epoch, self.args, feature,labels, task)
        # all_features=feature.cpu().numpy()
        # all_labels=labels.cpu().numpy()
        # # all_features = np.concatenate(all_features, axis=0)
        # # all_labels = np.concatenate(all_labels, axis=0)

        # # 使用 t-SNE 进行降维
        # # 获取当前时间戳
        # current_timestamp = int(time.time())
        # # tsne = TSNE(n_components=2, perplexity=45, random_state=20)
        # tsne = TSNE(n_components=2, perplexity=45)
        # features_tsne = tsne.fit_transform(all_features)
        # task=self.args.target
        # if not os.path.exists(osp.join("/home/q22301197/DG/CIRL/CIRL_main/画图/MVCE", self.args.dataset, task)):
        #     os.makedirs(osp.join("/home/q22301197/DG/CIRL/CIRL_main/画图/MVCE", self.args.dataset, task))
        # tSNE_filename = osp.join("/home/q22301197/DG/CIRL/CIRL_main/画图/MVCE", self.args.dataset, task, f"{current_timestamp}baseline_new.pdf")
        # class_labels = {0: 'dog', 1: 'elephant', 2: 'giraffe', 3: 'guitar', 4: 'horse', 5: 'house', 6: 'person'}
        # colors = [
        # '#1f77b4',  # 蓝色
        # '#ff7f0e',  # 橙色
        # '#2ca02c',  # 绿色
        # '#d62728',  # 红色
        # '#9467bd',  # 紫色
        # '#8c564b',  # 棕色
        # '#e377c2',  # 粉红色
        # # '#7f7f7f',  # 灰色
        # # '#bcbd22',  # 黄色
        # # '#17becf',  # 青色
        # # '#aec7e8',  # 浅蓝色
        # # '#ffbb78',  # 浅橙色
        # # '#98df8a',  # 浅绿色
        # # '#ff9896',  # 浅红色
        # # '#c5b0d5',  # 浅紫色
        # # '#c49c94',  # 浅棕色
        # # '#f7b6d2',  # 浅粉色
        # # '#c7c7c7',  # 浅灰色
        # # '#dbdb8d',  # 浅黄色
        # # '#9edae5',  # 浅青色
        # # '#393b79',  # 深蓝色
        # # '#637939',  # 深绿色
        # # '#8c6d31',  # 深棕色
        # # '#843c39',  # 深红色
        # # '#7b4173',  # 深紫色
        # # '#5254a3',  # 深蓝紫色
        # # '#637b53',  # 深绿褐色
        # # '#8ca252',  # 深黄绿色
        # # '#bd9e39',  # 深黄褐色
        # # '#ad494a',   # 深红褐色
        # # '#9c9ede'   # 浅蓝紫色
        # ]
        # fig, ax = plt.subplots(figsize=(10, 10))

        # # 设置刻度线朝里，且在四周都显示
        # ax.tick_params(direction='in', which='both',length=10)
        # ax.xaxis.set_ticks_position('both')
        # ax.yaxis.set_ticks_position('both')

        # # 绘制散点图，对每个类别分别绘制
        # scatter_plots = []
        # for class_idx, class_name in class_labels.items():
        #     idxs = np.where(all_labels == class_idx)
        #     scatter_plot = ax.scatter(features_tsne[idxs, 0], features_tsne[idxs, 1], c=colors[class_idx], label=class_name)
        #     scatter_plots.append(scatter_plot)

        # # 添加图例，并控制内容
        # legend = ax.legend(handles=scatter_plots, loc='upper right',prop={'size': 25},handlelength=0, handletextpad=1)

        # # 控制图例内容的样式，如字体大小和标题
        # legend.get_title().set_fontsize('12') # 设置图例标题的字体大小
        # plt.setp(legend.get_texts(), fontsize='14') # 设置图例文本的字体大小

        # # 保存图像
        # plt.savefig(tSNE_filename, bbox_inches='tight')



        # target_feature, target_labels = collect_feature(target_loader_1, classifier, gnn_network, DEVICE)
        self.encoder.train()
        self.classifier.train()
        self.classifier_ad.train()
        self.masker.train()
        self.masker_fft.train()
        for it, (batch, label, domain) in enumerate(self.train_loader):
            # print(index)
            # time.sleep(10)
            # preprocessing
            if self.config["supcon"]["isexpand"]:
                batch = torch.cat(batch, dim=0).to(self.device)
                labels = torch.cat(label, dim=0).to(self.device)
            else:
                batch = batch.to(self.device)
                labels = label.to(self.device)
            # print(batch.shape)


            # if self.args.target in pacs_dataset:
            #     labels -= 1
            # zero grad
            self.encoder_optim.zero_grad()
            self.classifier_optim.zero_grad()
            self.classifier_ad_optim.zero_grad()
            self.masker_optim.zero_grad()

            self.masker_fft_optim.zero_grad()


            # forward
            loss_dict = {}
            correct_dict = {}
            num_samples_dict = {}
            total_loss = 0.0

            ## --------------------------step 1 : update G and C -----------------------------------
            features,f_spatial = self.encoder(batch)     #cnn出来的特征  出来的特征可能需要特殊处理，请注意******************************************************************************************
            features = features.to(self.device)
            f_spatial = f_spatial.to(self.device)
            # print("##################################")
            # print(f_spatial.shape)
            # print(features.shape)

            Resnet_dim=2048
            # Resnet_dim=512

            num_groups = 16
            f_spatial = f_spatial.view(64,num_groups,Resnet_dim//num_groups,7,7).mean(dim=2).squeeze(2)


            masks_sup = self.masker(features.detach()).to(self.device)#此处为mask出来的前景特征
            masks_inf = (torch.ones_like(masks_sup) - masks_sup).to(self.device)

            masks_fft_abs_sup,masks_fft_pha_sup = self.masker_fft(f_spatial.detach())
            # print(masks_fft_sup.shape)
            # f_spatial=f_spatial.view(64,1,64,7,7).repeat(1,8,1,1,1).view(64,512,7,7)
            
            masks_fft_abs_sup = masks_fft_abs_sup.view(64,num_groups,-1).view(64,num_groups,7,7)
            masks_fft_pha_sup = masks_fft_pha_sup.view(64,num_groups,-1).view(64,num_groups,7,7)

            # # 假设 masks_fft_sup 是由神经网络学习出来的，我们取其前一半
            # masks_fft_sup_half = masks_fft_sup[:, :256]  # 取前256个频率分量

            # # 为了使掩码共轭对称，我们需要复制并翻转masks_fft_sup_half的后半部分
            # # 注意：如果N是偶数，中间的元素（Nyquist频率）应该只有一个值，因此我们不翻转它
            # # masks_fft_sup_nyquist = masks_fft_sup[:, 256:257]  # 可能的Nyquist频率
            # masks_fft_sup_conj_symm = torch.flip(masks_fft_sup_half, dims=[1])  # 取共轭对称部分

            # # 现在，我们将masks_fft_sup的前半部分、Nyquist频率（如果有的话），以及共轭对称部分连接起来
            # masks_fft_sup = torch.cat([masks_fft_sup_half, masks_fft_sup_conj_symm], dim=1)


            masks_fft_abs_inf = (torch.ones_like(masks_fft_abs_sup) - masks_fft_abs_sup).to(self.device)
            masks_fft_pha_inf = (torch.ones_like(masks_fft_abs_sup) - masks_fft_abs_sup).to(self.device)

            if self.current_epoch <= 5:
                masks_sup = torch.ones_like(features.detach()).to(self.device)
                masks_inf = torch.ones_like(features.detach()).to(self.device)
                masks_fft_abs_sup = torch.ones_like(f_spatial.detach()).to(self.device)
                masks_fft_abs_inf = torch.ones_like(f_spatial.detach()).to(self.device)
                masks_fft_pha_inf = torch.ones_like(f_spatial.detach()).to(self.device)
                masks_fft_pha_sup = torch.ones_like(f_spatial.detach()).to(self.device)
            # print(masks_fft_sup.shape)
            # print(f_spatial.shape)
            f_fft = np.fft.fft2(f_spatial.detach().cpu().numpy(), axes=(2,3))
            f_fft_sup_pha = np.angle(f_fft)*masks_fft_pha_sup.detach().cpu().numpy()
            f_fft_sup_abs = np.abs(f_fft)*masks_fft_abs_sup.detach().cpu().numpy()
            fea_fft_sup = f_fft_sup_abs * np.exp(1j * f_fft_sup_pha)
            fea_fft_sup = (np.fft.fftshift(fea_fft_sup, axes=(2,3))).real
            fea_fft_sup = torch.from_numpy(fea_fft_sup).to(self.device)
            fea_fft_sup=fea_fft_sup.view(64,1,num_groups,7,7).repeat(1,Resnet_dim//num_groups,1,1,1).view(64,Resnet_dim,7,7)
            fea_fft_sup = fea_fft_sup.view(64,Resnet_dim,7*7).mean(dim=-1)

            f_fft_inf_pha = np.angle(f_fft)*masks_fft_pha_inf.detach().cpu().numpy()
            f_fft_inf_abs = np.abs(f_fft)*masks_fft_abs_inf.detach().cpu().numpy()
            fea_fft_inf = f_fft_inf_abs * np.exp(1j * f_fft_inf_pha)
            fea_fft_inf = (np.fft.fftshift(fea_fft_inf, axes=(2,3))).real
            fea_fft_inf = torch.from_numpy(fea_fft_inf).to(self.device)
            fea_fft_inf=fea_fft_inf.view(64,1,num_groups,7,7).repeat(1,Resnet_dim//num_groups,1,1,1).view(64,Resnet_dim,7,7)
            fea_fft_inf = fea_fft_inf.view(64,Resnet_dim,7*7).mean(dim=-1)
            # print(masks_sup)
            # print(masks_sup.shape)    # (64,512)


            #*******************************************个体不变性损失*******************************************************
            if self.un and self.current_epoch <= self.un_epoch:
                loss_un = self.model_inv(features[:16], index, epoch=self.current_epoch)
                loss_dict["loss_un"] = loss_un.item()
            else:
                loss_un = 0.0
                loss_dict["loss_un"] = loss_un

            features_sup = features * masks_sup   #把前景信息提取出来

            original_dtype = features_sup.dtype


            features_inf = features * masks_inf

            features_sup = (0.5*features_sup + 0.5*fea_fft_sup).to(dtype=original_dtype)
            features_inf = (0.5*features_inf + 0.5*fea_fft_inf).to(dtype=original_dtype)

            scores_sup = self.classifier(features_sup)   #对于前景和背景，用两个分类器分别分类输出
            scores_inf = self.classifier_ad(features_inf)
            

            if self.config["use_hx"] and self.current_epoch >=0:
                pro_scores_sup = F.softmax(scores_sup, dim=1)
                p1 = pro_scores_sup[torch.arange(pro_scores_sup.size(0)), labels]
                pro_scores_inf = F.softmax(scores_inf, dim=1)
                p2 = pro_scores_inf[torch.arange(pro_scores_inf.size(0)), labels]

                # print(p1)
                h_sup = -p1*torch.log(p1)
                h_inf = -p2*torch.log(p2)
                # print(h_sup)
                # print(h_inf)

                dif_h = torch.abs(h_sup - h_inf)


                if self.current_epoch>=200:
                    # print("********************************************************P 1：*****************************************************")
                    # print(p1)
                    # print("********************************************************P 2：*****************************************************")
                    # print(p2)
                    # time.sleep(10)
                    print("********************************************************前景的熵：*****************************************************")
                    print(h_sup)
                    print("********************************************************背景的熵：*****************************************************")
                    print(h_inf)
                    print("*********************************************************熵的差：*********************************************************")
 
                min_dif = torch.min(dif_h)
                max_dif = torch.max(dif_h)

                dif_h = 0.1-(dif_h-min_dif)*(0.1-0)/(max_dif-min_dif)+0 +1
                # mean = dif_h.mean()
                # std = dif_h.std()
                # dif_h = (dif_h - mean) / std
                # print(dif_h)
                # time.sleep(50)


                # print(dif_h)
                # time.sleep(10)
            else :
                dif_h = torch.exp(torch.zeros(scores_sup.size(0))).to(self.device)
            loss_dict["dif_h"] = dif_h.mean()
            # print(dif_h)
            if self.config["supcon"]["is_supcon"]  and self.current_epoch>=0:
                if self.config["open_ssl_sup"]:
                    loss_unsup = self.supcon_loss(self.config["tsl"],features_sup*(dif_h.unsqueeze(1).repeat(1,features_sup.size(1))),is_unsup = self.config["open_ssl_sup"],epoch=self.current_epoch)
                    loss_dict["loss_unsup"] = loss_unsup.item()
                    loss_sup = self.supcon_loss(self.config["tcl"],features_sup*(dif_h.unsqueeze(1).repeat(1,features_sup.size(1))),is_unsup = False,epoch=self.current_epoch,labels=labels)
                    loss_dict["loss_sup"] = loss_sup.item()
                else:
                    loss_unsup = 0.0
                    loss_dict["loss_unsup"] = loss_unsup
                    loss_sup = self.supcon_loss(features_sup*(dif_h.unsqueeze(1).repeat(1,features_sup.size(1))),is_unsup = False,epoch=self.current_epoch,labels=labels,fusion=True)
                    loss_dict["loss_sup"] = loss_sup.item()
                if self.config["open_ssl_sup"]:
                    const_weight_ssl = get_current_consistency_weight(epoch=self.current_epoch,
                                                            weight=self.config["lam_const_ssl"],
                                                            rampup_length=self.config["warmdown_epoch_ssl"],
                                                            rampup_type=self.config["warmdown_type"])
                    const_weight_sup = get_current_consistency_weight(epoch=self.current_epoch,
                                                            weight=self.config["lam_const_sup"],
                                                            rampup_length=self.config["warmup_epoch_sup"],
                                                            rampup_type=self.config["warmup_type"])
                    totle_conloss = loss_unsup*const_weight_ssl+const_weight_sup*loss_sup
                    
                else:
                    totle_conloss = loss_sup + 0.5*loss_unsup
                
            else:
                totle_conloss = 0.0
            # if self.config["min_dis_sup"]:
            #     total_loss+=self.config["alpa_dis_sup"]*dis_sup.mean()
            loss_dict["cl_ssl"] = totle_conloss

            assert batch.size(0) % 2 == 0
            split_idx = int(batch.size(0) / 2)
            features_ori, features_aug = torch.split(features, split_idx)
            assert features_ori.size(0) == features_aug.size(0)

            # classification loss for sup feature    前景的loss
            loss_cls_sup = criterion(scores_sup, labels)
            loss_dict["sup"] = loss_cls_sup.item()
            correct_dict["sup"] = calculate_correct(scores_sup, labels)
            num_samples_dict["sup"] = int(scores_sup.size(0))

            # classification loss for inf feature    
            loss_cls_inf = criterion(scores_inf, labels)
            loss_dict["inf"] = loss_cls_inf.item()
            correct_dict["inf"] = calculate_correct(scores_inf, labels)
            num_samples_dict["inf"] = int(scores_inf.size(0))

            # factorization loss for features between ori and aug
            if self.config["supcon"]["is_fac"]:
                loss_fac = factorization_loss(features_ori,features_aug)
                loss_dict["fac"] = loss_fac.item()
            else:
                loss_fac = 0
                loss_dict["fac"] = loss_fac

            if self.open_const_weight:
                # get consistency weight
                const_weight = get_current_consistency_weight(epoch=self.current_epoch,
                                                            weight=self.config["lam_const"],
                                                            rampup_length=self.config["warmup_epoch"],
                                                            rampup_type=self.config["warmup_type"])
                const_weight_down = get_current_consistency_weight(epoch=self.current_epoch,
                                                            weight=self.config["lam_const"],
                                                            rampup_length=self.config["warmup_epoch"],
                                                            rampup_type=self.config["warmdown_type"])
            else:
                const_weight = self.al_supcon
            # calculate total loss

            if self.config["supcon"]["is_awl"]:
                awl_loss ,loss_weights_tmp, loss_bias_tmp= self.awl_global(totle_conloss) 
                total_loss = 0.5*loss_cls_sup + 0.5*loss_cls_inf + awl_loss
            else:
                if self.config["const_weight_dowm"]:
                    total_loss = 0.5*loss_cls_sup + 0.5*loss_cls_inf  + const_weight_down * totle_conloss +loss_un*self.al_un + const_weight_down * loss_fac
                elif self.config["unite_loss"] :
                    total_loss = loss_cls_sup+loss_cls_inf/(loss_cls_inf/loss_cls_sup).detach()+totle_conloss/(totle_conloss/loss_cls_sup).detach()
                elif self.config["open_ssl_sup"]:
                    total_loss = 0.5*loss_cls_sup + 0.5*loss_cls_inf + totle_conloss
                else:
                    total_loss = 0.5*loss_cls_sup + 0.5*loss_cls_inf  + const_weight * totle_conloss +loss_un*self.al_un + const_weight * loss_fac
            loss_dict["total"] = total_loss.item()

            if self.config["const_weight_dowm"]:   #是否开始损失系数的递减
                loss_dict["const_weight_dowm"] = const_weight_down
            elif self.config["open_ssl_sup"]:
                loss_dict["const_weight_ssl"] = const_weight_ssl
                loss_dict["const_weight_sup"] = const_weight_sup

            elif self.config["supcon"]["open_const_weight"]:
                loss_dict["const_weight"] = const_weight

            # backward
            # time.sleep(10)
            # print("back之前")
            total_loss.backward()
            # print("back之后")


            # time.sleep(10)

            # update
            self.encoder_optim.step()
            self.classifier_optim.step()
            self.classifier_ad_optim.step()
            # self.awl_global_optim.step()


            ## ---------------------------------- step2: update masker------------------------------
            self.masker_optim.zero_grad()
            self.masker_fft_optim.zero_grad()
            features,f_spatial = self.encoder(batch)

            features = features.to(self.device)
            f_spatial = f_spatial.to(self.device)
            f_spatial = f_spatial.view(64,num_groups,Resnet_dim//num_groups,7,7).mean(dim=2).squeeze(2)

            masks_sup = self.masker(features.detach()).to(self.device)#此处为mask出来的前景特征
            masks_inf = (torch.ones_like(masks_sup) - masks_sup).to(self.device)

            masks_fft_abs_sup,masks_fft_pha_sup = self.masker_fft(f_spatial.detach())
            
            masks_fft_abs_sup = masks_fft_abs_sup.view(64,num_groups,-1).view(64,num_groups,7,7)
            masks_fft_pha_sup = masks_fft_pha_sup.view(64,num_groups,-1).view(64,num_groups,7,7)

            masks_fft_abs_inf = (torch.ones_like(masks_fft_abs_sup) - masks_fft_abs_sup).to(self.device)
            masks_fft_pha_inf = (torch.ones_like(masks_fft_abs_sup) - masks_fft_abs_sup).to(self.device)


            f_fft = np.fft.fft2(f_spatial.detach().cpu().numpy(), axes=(2,3))
            f_fft_sup_pha = np.angle(f_fft)*masks_fft_pha_sup.detach().cpu().numpy()
            f_fft_sup_abs = np.abs(f_fft)*masks_fft_abs_sup.detach().cpu().numpy()
            fea_fft_sup = f_fft_sup_abs * np.exp(1j * f_fft_sup_pha)
            fea_fft_sup = (np.fft.fftshift(fea_fft_sup, axes=(2,3))).real
            fea_fft_sup = torch.from_numpy(fea_fft_sup).to(self.device)
            fea_fft_sup=fea_fft_sup.view(64,1,num_groups,7,7).repeat(1,Resnet_dim//num_groups,1,1,1).view(64,Resnet_dim,7,7)
            fea_fft_sup = fea_fft_sup.view(64,Resnet_dim,7*7).mean(dim=-1)

            f_fft_inf_pha = np.angle(f_fft)*masks_fft_pha_inf.detach().cpu().numpy()
            f_fft_inf_abs = np.abs(f_fft)*masks_fft_abs_inf.detach().cpu().numpy()
            fea_fft_inf = f_fft_inf_abs * np.exp(1j * f_fft_inf_pha)
            fea_fft_inf = (np.fft.fftshift(fea_fft_inf, axes=(2,3))).real
            fea_fft_inf = torch.from_numpy(fea_fft_inf).to(self.device)
            fea_fft_inf=fea_fft_inf.view(64,1,num_groups,7,7).repeat(1,Resnet_dim//num_groups,1,1,1).view(64,Resnet_dim,7,7)
            fea_fft_inf = fea_fft_inf.view(64,Resnet_dim,7*7).mean(dim=-1)


            features_sup = features * masks_sup 
            features_inf = features * masks_inf
            original_dtype = features_sup.dtype
            # dis_sup = F.cosine_similarity(features_sup,fea_fft_sup)
            # dis_inf = F.cosine_similarity(features_inf,fea_fft_inf)
            features_sup = (0.5*features_sup + 0.5*fea_fft_sup).to(dtype=original_dtype)
            features_inf = (0.5*features_inf + 0.5*fea_fft_inf).to(dtype=original_dtype)

            
            scores_sup = self.classifier(features_sup)
            scores_inf = self.classifier_ad(features_inf)

            loss_cls_sup = criterion(scores_sup, labels)
            loss_cls_inf = criterion(scores_inf, labels)
            loss_sup_inf = 0.5*loss_cls_sup - 0.5*loss_cls_inf 
            # loss_fft_sup_inf = 0.5*dis_sup.mean()-0.5*dis_inf.mean()
            loss_dict["loss_cls_sup"] = loss_cls_sup
            loss_dict["loss_cls_inf"] = loss_cls_inf
            loss_dict["loss_sup_sub_inf"] = loss_sup_inf
            # loss_dict["loss_dis"] = loss_fft_sup_inf
            # loss_dict["loss_dis_sup_fft"] = dis_sup.mean()
            # loss_dict["loss_dis_inf_fft"] = dis_inf.mean()
            
            total_loss = 0.5*loss_cls_sup - 0.5*loss_cls_inf 
            dis_sup = F.cosine_similarity(features_sup,fea_fft_sup)
            loss_dict["loss_dis_sup_fft"] = dis_sup.mean()
            if self.config["min_dis_sup"]:
                total_loss+=self.config["alpa_dis_sup"]*dis_sup.mean()
            loss_dict["mask_total_loss"] = total_loss
            total_loss.backward()
            self.masker_optim.step()

            self.masker_fft_optim.step()

            self.global_step += 1

            # record
            self.logger.log(
                it=it,
                iters=len(self.train_loader),
                losses=loss_dict,
                samples_right=correct_dict,
                total_samples=num_samples_dict
            )

        # turn on eval modea
        self.encoder.eval()
        self.classifier.eval()
        self.masker.eval()
        self.masker_fft.eval()
        self.classifier_ad.eval()

        # evaluation
        with torch.no_grad():
            for phase, loader in self.eval_loader.items():
                total = len(loader.dataset)
                class_correct = self.do_eval(loader)
                class_acc = float(class_correct) / total
                self.logger.log_test(phase, {'class': class_acc})
                self.results[phase][self.current_epoch] = class_acc

            # save from best model
            if self.results['test'][self.current_epoch] >= self.best_acc:
                self.best_acc = self.results['test'][self.current_epoch]
                self.best_epoch = self.current_epoch + 1
                self.logger.save_best_model(self.encoder, self.classifier, self.best_acc,self.current_epoch)

    def do_eval(self, loader):
        correct = 0
        for it, (batch, domain) in enumerate(loader):
            data, labels, domains = batch[0].to(self.device), batch[1].to(self.device), domain.to(self.device)
            # if self.args.target in pacs_dataset:
            #     labels -= 1
            features,_ = self.encoder(data)
            scores = self.classifier(features)
            correct += calculate_correct(scores, labels)
        return correct


    def do_training(self):
        self.logger = Logger(self.args, self.config, update_frequency=30)
        self.logger.save_config()

        self.epochs = self.config["epoch"]
        self.results = {"val": torch.zeros(self.epochs), "test": torch.zeros(self.epochs)}

        self.best_acc = 0
        self.best_epoch = 0
        self.logger.save_best_model(self.encoder, self.classifier, self.best_acc,0)

        for self.current_epoch in range(self.epochs):

            # step schedulers
            self.encoder_sched.step()
            self.classifier_sched.step()

            self.logger.new_epoch([group["lr"] for group in self.encoder_optim.param_groups])
            self._do_epoch()
            self.logger.finish_epoch()

        # save from best model
        val_res = self.results['val']
        test_res = self.results['test']
        self.logger.save_best_acc(val_res, test_res, self.best_acc, self.best_epoch - 1)

        return self.logger


def main():
    args, config = get_args()
    if config["use_seed"]:
        # print("KKKK")
        seed = args.seed
        torch.manual_seed(seed)
        # np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = torch.initial_seed()
    args.seed = seed
    print(seed)
    trainer = Trainer(args, config, device)
    trainer.do_training()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
