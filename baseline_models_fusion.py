from Models.generate_model import *
from Models.res2net import res2net50_v1b_26w_4s,res2net50_v1b_14w_8s,res2net101_v1b_26w_4s
from Models.fundus_swin_network import build_model as fundus_build_model
from Models.unetr import UNETR_base_3DNet
import random
import torch.nn.functional as F
from perturbot.match import get_coupling_egw_labels_ott, get_coupling_fot
import numpy as np
def cosine_loss(x, y):
    # 先归一化特征
        # 如果是1维tensor，需要先增加一个维度
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if y.dim() == 1:
        y = y.unsqueeze(0)
    
    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)
    return 1 - F.cosine_similarity(x, y).mean()



class Medical_base_2DNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, num_classes=10):
        super(Medical_base_2DNet, self).__init__()
        # ---- ResNet Backbone ----
        self.res2net = res2net50_v1b_26w_4s(pretrained=True)
    def forward(self, x):
        #origanal x do:
        x = self.res2net.conv1(x)
        x = self.res2net.bn1(x)
        x = self.res2net.relu(x)
        x = self.res2net.maxpool(x)      # bs, 64, 128, 128
        # ---- low-level features ----
        x1 = self.res2net.layer1(x)      # bs, 256, 128, 128
        x2 = self.res2net.layer2(x1)     # bs, 512, 64, 64
        x3 = self.res2net.layer3(x2)     # bs, 1024, 32, 32
        x4 = self.res2net.layer4(x3)     # bs, 2048, 16, 16
        x4 = self.res2net.avgpool(x4)    # bs, 2048, 1, 1
        x4 = x4.view(x4.size(0), -1)  # bs, 1， 2048,
        return x4





class Medical_base_3DNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, classifier_OCT_dims,num_classes=10):
        super(Medical_base_3DNet, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet_3DNet = generate_model(model_type='resnet', model_depth=10, input_W=classifier_OCT_dims[0][0],
                                           input_H=classifier_OCT_dims[0][1], input_D=classifier_OCT_dims[0][2],
                                           resnet_shortcut='B',
                                           no_cuda=True, gpu_id=[0],
                                           pretrain_path='/mnt/sdb/feilong/aaai_retinal/Retinal_OCT/Confidence_MedIA/pretrain/resnet_10_23dataset.pth', nb_class=num_classes)

    def forward(self, x):
        
        x = self.resnet_3DNet.conv1(x)
        x = self.resnet_3DNet.bn1(x)
        x = self.resnet_3DNet.relu(x)
        x = self.resnet_3DNet.maxpool(x)  # bs, 64, 32, 32,64
        # ---- low-level features ----
        x1 = self.resnet_3DNet.layer1(x)  # bs, 64, 32, 32,64
        x2 = self.resnet_3DNet.layer2(x1)  # bs, 128, 16, 16,32
        x3 = self.resnet_3DNet.layer3(x2)  # bs, 256, 16, 16,32
        x4 = self.resnet_3DNet.layer4(x3)  # bs, 512, 16, 16，32
        x4 = self.resnet_3DNet.avgpool(x4) # bs, 512, 16, 1，1
        x4 = x4.view(x4.size(0), -1) # 8192
        return x4




#  --------------------------------------------------------  1
class Multi_ResNet(nn.Module):
    def __init__(self, classes, modalties, classifiers_dims, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param views: Number of modalties
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(Multi_ResNet, self).__init__()
        self.modalties = modalties
        self.classes = classes
        self.lambda_epochs = lambda_epochs
        # ---- 2D Res2Net Backbone ----
        self.res2net_2DNet = Medical_base_2DNet(num_classes=self.classes)
        # ---- 3D ResNet Backbone ----
        classifier_OCT_dims = classifiers_dims[0]
        self.resnet_3DNet = Medical_base_3DNet(classifier_OCT_dims,num_classes=self.classes)
        self.sp = nn.Softplus()
        self.fc = nn.Linear(8192, classes)
        self.ce_loss = nn.CrossEntropyLoss()
        self.fundus2oct = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4096, 6144),
        )
        self.oct2fundus = nn.Sequential(
            nn.Linear(6144, 4096),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4096, 2048),
        )
        self.oct_fusion = nn.Sequential(
            nn.Linear(6144*2, 6144),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(6144, 6144),
        )
        self.attention_fundus = SelfAttentionBlock(embed_dim=2048, num_heads=4,ff_dim=2048, dropout=0.1)
    def forward(self, X, y, T_feature_2, training=True):
        # bs,2048
        backboneout_1 = self.res2net_2DNet(X[0])
        # bs,6144
        backboneout_2 = self.resnet_3DNet(X[1])
        if training:
            grouped_fundus_feature = self.group_features_by_label(y.cpu(),backboneout_1.cpu().detach().numpy())
            grouped_oct_feature = self.group_features_by_label(y.cpu(),backboneout_2.cpu().detach().numpy())
            labels = sorted(grouped_oct_feature.keys())
            # fundus2oct    
            T_dict, log = get_coupling_egw_labels_ott((grouped_fundus_feature, grouped_oct_feature))
            # T_feature, fm_log = get_coupling_fot((grouped_fundus_feature,grouped_oct_feature), T_dict)
            T_feature_2, fm_log = get_coupling_fot((grouped_oct_feature,grouped_fundus_feature), T_dict)
            OT_fudus = torch.from_numpy(np.concatenate([grouped_fundus_feature[l] for l in labels])).to('cuda')
            OT_oct = torch.from_numpy(np.concatenate([grouped_oct_feature[l] for l in labels])).to('cuda')
            print("OT_fudus",OT_fudus)  
            T = torch.from_numpy(
                self.mdict_to_matrix(T_dict,
                    np.concatenate([np.ones(grouped_fundus_feature[l].shape[0]) * l for l in labels]),
                    np.concatenate([np.ones(grouped_oct_feature[l].shape[0]) * l for l in labels]),
                )
            )
            T[T.sum(axis=-1) == 0, :] = 1e-8
            print("T.shape",T.shape)
            x_idx = torch.arange(OT_fudus.shape[0])  # 生成 [0,1,2,...,31]
            ot_loss = 0
            pred_oct_feature = []
            for i in range(OT_oct.shape[0]):
                Y_idx = torch.multinomial(T[x_idx[i], :], 1)
                sample_Y = OT_oct[Y_idx]
                hat_Y = self.fundus2oct(OT_fudus[i])
                ot_loss+=cosine_loss(hat_Y,sample_Y)
                # ot_loss+=F.mse_loss(sample_Y,hat_Y)
                pred_oct_feature.append(hat_Y)
            # oct2fundus      
            T_dict_2, log_2 = get_coupling_egw_labels_ott((grouped_oct_feature, grouped_fundus_feature))
            print("OT_fudus",OT_fudus)  
            T_2 = torch.from_numpy(
                self.mdict_to_matrix(T_dict_2,
                    np.concatenate([np.ones(grouped_oct_feature[l].shape[0]) * l for l in labels]),
                    np.concatenate([np.ones(grouped_fundus_feature[l].shape[0]) * l for l in labels]),
                )
            )
            T_2[T_2.sum(axis=-1) == 0, :] = 1e-8
            print("T_2.shape",T_2.shape)
            x_idx = torch.arange(OT_oct.shape[0])  # 生成 [0,1,2,...,31]
            ot_loss_2 = 0
            pred_fundus_feature = []
            for i in range(OT_oct.shape[0]):
                Y_idx = torch.multinomial(T_2[x_idx[i], :], 1)
                sample_Y = OT_fudus[Y_idx]
                hat_Y = self.oct2fundus(OT_oct[i])
                ot_loss_2+=cosine_loss(hat_Y,sample_Y)
                # ot_loss+=F.mse_loss(sample_Y,hat_Y)
                pred_fundus_feature.append(hat_Y)
            ot_loss/=OT_oct.shape[0]
            ot_loss_2/=OT_oct.shape[0]
            ot_loss+=ot_loss_2
            pred_fundus_feature = torch.stack(pred_fundus_feature)
            pred_oct_feature = torch.stack(pred_oct_feature)
            print("ot_loss",ot_loss)
            # T_feature = torch.from_numpy(T_feature).to('cuda')
            T_feature_2 = torch.from_numpy(T_feature_2).to('cuda')
            # ot_feature = torch.mm(backboneout_1,T_feature)
            ot_feature_2 = torch.mm(backboneout_2,T_feature_2)
            # oct fusion
            oct_feature = self.oct_fusion(torch.cat([backboneout_2,pred_oct_feature],dim=1))
            # fundus fusion
            f1_fundus = backboneout_1.unsqueeze(0)     # (1, B, 2048)
            f2_fundus = ot_feature_2.unsqueeze(0)        # (1, B, 2048)
            f3_fundus = pred_fundus_feature.unsqueeze(0)  # (1, B, 2048)
            tokens_fundus = torch.cat([f1_fundus, f2_fundus, f3_fundus], dim=0)  # (3, B, 2048)
            att_out_fundus = self.attention_fundus(tokens_fundus)
            att_out_fundus = att_out_fundus.transpose(0,1)   # -> (B, 3, 2048)
            att_flat_fundus = att_out_fundus.mean(dim=1)
            

        else:
            pred_oct_feature = self.fundus2oct(backboneout_1)
            pred_fundus_feature = self.oct2fundus(backboneout_2)
            T_feature_2 = torch.from_numpy(T_feature_2).to('cuda')
            ot_feature_2 = torch.mm(backboneout_2,T_feature_2)
            oct_feature = self.oct_fusion(torch.cat([backboneout_2,pred_oct_feature],dim=1))
            # fundus fusion
            f1_fundus = backboneout_1.unsqueeze(0)     # (1, B, 2048)
            f2_fundus = ot_feature_2.unsqueeze(0)        # (1, B, 2048)
            f3_fundus = pred_fundus_feature.unsqueeze(0)  # (1, B, 2048)
            tokens_fundus = torch.cat([f1_fundus, f2_fundus, f3_fundus], dim=0)  # (3, B, 2048)
            att_out_fundus = self.attention_fundus(tokens_fundus)
            att_out_fundus = att_out_fundus.transpose(0,1)   # -> (B, 3, 2048)
            att_flat_fundus = att_out_fundus.mean(dim=1)
        # ----------------------------
        # 融合
        combine_features =torch.cat([att_flat_fundus,oct_feature],1)

        pred = self.fc(combine_features)
        # print(pred.shape)
        loss = self.ce_loss(pred, y)

        loss = torch.mean(loss)
        if training:
            return pred, loss, ot_loss
        else:
            return pred, loss
    def mdict_to_matrix(self,M_dict, source_labels, target_labels):
        Mtot = np.zeros((len(source_labels), len(target_labels)))
        for l, M in M_dict.items():
            Mtot[
                np.ix_(np.where(source_labels == l)[0], np.where(target_labels == l)[0])
            ] = M
        return Mtot
    def group_features_by_label(self, y, p, num_classes=3):
        """
        y: [64] 标签
        p: [64, 5, 196] batch_size, num_clusters, features
        return: 字典,key是类别,value是该类别所有特征的numpy数组
        """
        unique_labels = np.unique(y)

        grouped_features = {int(label): [] for label in unique_labels}        
        # 将tensor转换为numpy数组
        y_np = y
        p_np = p
        
        # 遍历每个样本
        for label, features in zip(y_np, p_np):
            label = int(label)  # 确保标签是整数
            grouped_features[label].append(features)  # [5, 196]
        
        # 将每个类别的特征堆叠成numpy数组
        for label in grouped_features:
            if grouped_features[label]:  # 如果该类别有样本
                grouped_features[label] = np.stack(grouped_features[label])
                # shape: [num_samples_in_class, 5, 196]
        
        return grouped_features
#---------------------------
class SelfAttentionBlock(nn.Module):
    """
    一个完整的 Transformer Encoder Block:
      - Multi-Head Self-Attention
      - Feed Forward Network (FFN)
      - 残差连接 + LayerNorm
    """

    def __init__(self, embed_dim=256, num_heads=4, ff_dim=1024, dropout=0.1):
        """
        :param embed_dim:  输入和输出的特征维度
        :param num_heads:  多头数
        :param ff_dim:     前馈网络中的隐藏层维度 (通常大于 embed_dim)
        :param dropout:    dropout 概率
        """
        super().__init__()
        # 多头自注意力
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=False)

        # 残差 + LayerNorm
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # 前馈网络 (两层全连接)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )

        # 残差 + LayerNorm
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: 形状 (seq_len, batch_size, embed_dim)
        :return:
          out: 同样的 (seq_len, batch_size, embed_dim)
        """

        # --- 1) 自注意力子层 ---
        # Q=K=V = x (自注意力)
        attn_out, _ = self.self_attn(x, x, x)
        # 残差连接 + LayerNorm
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        # --- 2) 前馈子层 ---
        ffn_out = self.ffn(x)
        # 残差连接 + LayerNorm
        x = x + self.dropout2(ffn_out)
        out = self.norm2(x)

        return out



