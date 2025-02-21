from Models.generate_model import *
from Models.res2net import res2net50_v1b_26w_4s,res2net50_v1b_14w_8s,res2net101_v1b_26w_4s
from Models.fundus_swin_network import build_model as fundus_build_model
from Models.unetr import UNETR_base_3DNet
import random
import torch.nn.functional as F
from perturbot.perturbot.match import get_coupling_egw_labels_ott, get_coupling_fot
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
class Medical_feature_2DNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, num_classes=10):
        super(Medical_feature_2DNet, self).__init__()
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
        return x4


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

class Medical_base2_2DNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, num_classes=10):
        super(Medical_base2_2DNet, self).__init__()
        # ---- ResNet Backbone ----
        self.res2net = res2net50_v1b_14w_8s(pretrained=True)
        # self.res2net = res2net50_v1b_26w_4s(pretrained=True)
        # self.res2net = res2net101_v1b_26w_4s(pretrained=True)

    def forward(self, x):
        #origanal x do:
        x = self.res2net.conv1(x)
        x = self.res2net.bn1(x)
        x = self.res2net.relu(x)
        x = self.res2net.maxpool(x)      # bs, 64, 64, 64
        # ---- low-level features ----
        x1 = self.res2net.layer1(x)      # bs, 256, 64, 64
        x2 = self.res2net.layer2(x1)     # bs, 512, 32, 32
        x3 = self.res2net.layer3(x2)     # bs, 1024, 16, 16
        x4 = self.res2net.layer4(x3)     # bs, 2048, 8, 8
        x4 = self.res2net.avgpool(x4)
        x4 = x4.view(x4.size(0), -1)
        return x4

class Medical_base_dropout_2DNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, num_classes=10):
        super(Medical_base_dropout_2DNet, self).__init__()
        # ---- ResNet Backbone ----
        self.res2net = res2net50_v1b_26w_4s(pretrained=True)
    def forward(self, x):
        #origanal x do:
        x = self.res2net.conv1(x)
        x = self.res2net.bn1(x)
        x = self.res2net.relu(x)
        x = self.res2net.maxpool(x)      # bs, 64, 64, 64
        # dropout layer
        x = F.dropout(x, p=0.2)
        # ---- low-level features ----
        x1 = self.res2net.layer1(x)      # bs, 256, 64, 64
        x2 = self.res2net.layer2(x1)     # bs, 512, 32, 32
        x3 = self.res2net.layer3(x2)     # bs, 1024, 16, 16
        x4 = self.res2net.layer4(x3)     # bs, 2048, 8, 8
        # dropout layer
        x4 = F.dropout(x4, p=0.2)
        x4 = self.res2net.avgpool(x4)
        x4 = x4.view(x4.size(0), -1)
        return x4

class Medical_2DNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, num_classes=10):
        super(Medical_2DNet, self).__init__()
        # ---- ResNet Backbone ----
        self.res2net = res2net50_v1b_26w_4s(pretrained=True)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        #origanal x do:
        x = self.res2net.conv1(x)
        x = self.res2net.bn1(x)
        x = self.res2net.relu(x)
        x = self.res2net.maxpool(x)      # bs, 64, 64, 64
        # ---- low-level features ----
        x1 = self.res2net.layer1(x)      # bs, 256, 64, 64
        x2 = self.res2net.layer2(x1)     # bs, 512, 32, 32
        x3 = self.res2net.layer3(x2)     # bs, 1024, 16, 16
        x4 = self.res2net.layer4(x3)     # bs, 2048, 8, 8
        x4 = self.res2net.avgpool(x4)
        x4 = x4.view(x4.size(0), -1)
        out = self.fc(x4)
        return out


class Medical_3DNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, classifier_OCT_dims,num_classes=10):
        super(Medical_3DNet, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet_3DNet = generate_model(model_type='resnet', model_depth=10, input_W=classifier_OCT_dims[0][0],
                                           input_H=classifier_OCT_dims[0][1], input_D=classifier_OCT_dims[0][2],
                                           resnet_shortcut='B',
                                           no_cuda=True, gpu_id=[0],
                                           pretrain_path='./pretrain/resnet_10_23dataset.pth', nb_class=num_classes)
        if classifier_OCT_dims[0][0] == 128:
            self.fc = nn.Linear(8192, num_classes) # MMOCTF
        else:
            self.fc = nn.Linear(3072, num_classes) # OLIVES

    def forward(self, x):

        x = self.resnet_3DNet.conv1(x)
        x = self.resnet_3DNet.bn1(x)
        x = self.resnet_3DNet.relu(x)
        x = self.resnet_3DNet.maxpool(x)  # bs, 64, 64, 64
        # ---- low-level features ----
        x1 = self.resnet_3DNet.layer1(x)  # bs, 256, 64, 64
        x2 = self.resnet_3DNet.layer2(x1)  # bs, 512, 32, 32
        x3 = self.resnet_3DNet.layer3(x2)  # bs, 1024, 16, 16
        x4 = self.resnet_3DNet.layer4(x3)  # bs, 2048, 8, 8
        x4 = self.resnet_3DNet.avgpool(x4)
        x4 = x4.view(x4.size(0), -1)
        out = self.fc(x4)
        return out

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

class Medical_feature_3DNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, classifier_OCT_dims,num_classes=10):
        super(Medical_feature_3DNet, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet_3DNet = generate_model(model_type='resnet', model_depth=10, input_W=classifier_OCT_dims[0][0],
                                           input_H=classifier_OCT_dims[0][1], input_D=classifier_OCT_dims[0][2],
                                           resnet_shortcut='B',
                                           no_cuda=True, gpu_id=[0],
                                           pretrain_path='./pretrain/resnet_10_23dataset.pth', nb_class=num_classes)

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
        return x4

class Medical_base2_3DNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, classifier_OCT_dims,num_classes=10):
        super(Medical_base2_3DNet, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet_3DNet = generate_model(model_type='resnet', model_depth=18, input_W=classifier_OCT_dims[0][0],
                                           input_H=classifier_OCT_dims[0][1], input_D=classifier_OCT_dims[0][2],
                                           resnet_shortcut='A',
                                           no_cuda=True, gpu_id=[0],
                                           pretrain_path='./pretrain/resnet_18_23dataset.pth', nb_class=num_classes)

    def forward(self, x):

        x = self.resnet_3DNet.conv1(x)
        x = self.resnet_3DNet.bn1(x)
        x = self.resnet_3DNet.relu(x)
        x = self.resnet_3DNet.maxpool(x)  # bs, 64, 32, 32,64
        # ---- low-level features ----
        x1 = self.resnet_3DNet.layer1(x)  ## bs, 64, 32, 32,64
        x2 = self.resnet_3DNet.layer2(x1)  # bs, 128, 16, 16,32
        x3 = self.resnet_3DNet.layer3(x2)  # bs, 256, 16, 16,32
        x4 = self.resnet_3DNet.layer4(x3)  # bs, 512, 16, 16，32
        x4 = self.resnet_3DNet.avgpool(x4)  # bs, 512, 16, 1，1
        x4 = x4.view(x4.size(0), -1) # 8192
        return x4

class Medical_base_dropout_3DNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, classifier_OCT_dims,num_classes=10):
        super(Medical_base_dropout_3DNet, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet_3DNet = generate_model(model_type='resnet', model_depth=10, input_W=classifier_OCT_dims[0][0],
                                           input_H=classifier_OCT_dims[0][1], input_D=classifier_OCT_dims[0][2],
                                           resnet_shortcut='B',
                                           no_cuda=True, gpu_id=[0],
                                           pretrain_path='./pretrain/resnet_10_23dataset.pth', nb_class=num_classes)

    def forward(self, x):

        x = self.resnet_3DNet.conv1(x)
        x = self.resnet_3DNet.bn1(x)
        x = self.resnet_3DNet.relu(x)
        x = self.resnet_3DNet.maxpool(x)  # bs, 64, 64, 64
        # dropout layer
        x = F.dropout(x, p=0.2)
        # ---- low-level features ----
        x1 = self.resnet_3DNet.layer1(x)  # bs, 256, 64, 64
        x2 = self.resnet_3DNet.layer2(x1)  # bs, 512, 32, 32
        x3 = self.resnet_3DNet.layer3(x2)  # bs, 1024, 16, 16
        x4 = self.resnet_3DNet.layer4(x3)  # bs, 2048, 8, 8
        x4 = self.resnet_3DNet.avgpool(x4)
        # dropout layer
        x4 = F.dropout(x4, p=0.2)
        x4 = x4.view(x4.size(0), -1)
        return x4

class ResNet3D(nn.Module):

    def __init__(self, classes, modalties, classifiers_dims, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param views: Number of modalties
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(ResNet3D, self).__init__()
        self.modalties = modalties
        self.classes = classes
        self.lambda_epochs = lambda_epochs

        # ---- 3D ResNet Backbone ----
        classifier_OCT_dims = classifiers_dims
        self.resnet_3DNet = Medical_3DNet(classifier_OCT_dims,num_classes=self.classes)
        self.Classifiers= nn.ModuleList([self.resnet_3DNet])
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.sfm = nn.Softmax()

    def forward(self, X, y):
        output = self.infer(X[1])
        loss = 0
        for v_num in range(self.modalties):
            pred = output[v_num]
            # label = F.one_hot(y, num_classes=self.classes)
            # loss = self.ce_loss(label, pred)
            loss = self.ce_loss(pred, y)

        loss = torch.mean(loss)
        return pred, loss

    def infer(self, input):
        """
        :param input: Multi-view data
        :return: evidence of every view
        """
        evidence = dict()
        for m_num in range(self.modalties):
            backbone_output = self.Classifiers[m_num](input)
            evidence[m_num] = self.sfm(backbone_output)
        return evidence



class Res2Net2D(nn.Module):

    def __init__(self, classes, modalties, classifiers_dims, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param views: Number of modalties
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(Res2Net2D, self).__init__()
        self.modalties = modalties
        self.classes = classes
        self.lambda_epochs = lambda_epochs
        # ---- 2D Res2Net Backbone ----
        classifier_Fundus_dims = classifiers_dims[0]
        self.res2net_2DNet = Medical_2DNet(num_classes=self.classes)
        self.Classifiers= nn.ModuleList([self.res2net_2DNet])
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.sfm = nn.Softmax()

    def forward(self, X, y):
        output = self.infer(X[0])
        # loss = 0
        # for v_num in range(self.modalties):
        #     pred = output[v_num]
        #     # label = F.one_hot(y, num_classes=self.classes)
        #     # loss = self.ce_loss(label, pred)
        #     loss = self.ce_loss(pred, y)

        # loss = torch.mean(loss)
        # # return pred, loss
        # print(output.keys())
        print("output.shape",output.shape)
        return output

    def infer(self, input):
        """
        :param input: Multi-view data
        :return: evidence of every view
        """
        evidence = dict()
        for m_num in range(self.modalties):
            backbone_output = self.Classifiers[m_num](input)
            evidence[m_num] = self.sfm(backbone_output)
        # return evidence
        return backbone_output


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
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)

    def forward(self, query, key, value):
        # query, key, value shape: (seq_len, batch_size, embed_dim)
        attn_output, _ = self.attention(query, key, value)
        return attn_output


class Multi_ResNet_cross(nn.Module):
    def __init__(self, classes, modalties, classifiers_dims, lambda_epochs=1):
        super(Multi_ResNet_cross, self).__init__()
        self.modalties = modalties
        self.classes = classes
        self.lambda_epochs = lambda_epochs
        # ---- 2D Res2Net Backbone ----
        self.res2net_2DNet = Medical_base_2DNet(num_classes=self.classes)

        # ---- 3D ResNet Backbone ----
        classifier_OCT_dims = classifiers_dims[0]
        self.resnet_3DNet = Medical_base_3DNet(classifier_OCT_dims, num_classes=self.classes)

        # Cross-attention layer
        embed_dim = 2048  # Adjust based on your feature dimension
        self.cross_attention = CrossAttention(dim=embed_dim)

        self.pool = nn.AvgPool1d(kernel_size=3, stride=3)

        # Final classification layer
        self.fc = nn.Linear(embed_dim, classes)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, X, y):
        # Process each modality
        backboneout_1 = self.res2net_2DNet(X[0])  # 2D modality
        backboneout_2 = self.resnet_3DNet(X[1])  # 3D modality

        # Ensure both outputs have the same shape
        backboneout_1 = backboneout_1.unsqueeze(0)  # (1, batch_size, embed_dim)
        backboneout_2 = backboneout_2.unsqueeze(0)  # (1, batch_size, embed_dim)
        backboneout_2 = self.pool(backboneout_2)

        # Apply cross-attention
        cross_attn_output = self.cross_attention(backboneout_1, backboneout_2, backboneout_2)

        # Flatten output for classification
        cross_attn_output = cross_attn_output.squeeze(0)  # (batch_size, embed_dim)

        # Final classification
        pred = self.fc(cross_attn_output)
        loss = self.ce_loss(pred, y)

        return pred, loss, cross_attn_output



from torchvision.models import vit_b_16

class VisionTransformer3D(nn.Module):
    def __init__(self, input_dims, num_classes=10, patch_size=(4, 4, 4), dim=768, depth=6, heads=8, mlp_dim=2048, dropout=0.1):
        super(VisionTransformer3D, self).__init__()

        self.patch_size = patch_size
        self.dim = dim

        # Calculate the number of patches
        num_patches = (input_dims[0] // patch_size[0]) * (input_dims[1] // patch_size[1]) * (input_dims[2] // patch_size[2])
        patch_dim = patch_size[0] * patch_size[1] * patch_size[2]

        # Linear embedding for patches
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Transformer blocks
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=mlp_dim,
                dropout=dropout
            ),
            num_layers=depth
        )

        # Final classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        # Reshape and flatten the patches
        batch_size, channels, depth, height, width = x.shape

        x = x.unfold(2, self.patch_size[0], self.patch_size[0]) \
             .unfold(3, self.patch_size[1], self.patch_size[1]) \
             .unfold(4, self.patch_size[2], self.patch_size[2]) \
             .contiguous()

        x = x.view(batch_size, channels, -1, self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.patch_size[0] * self.patch_size[1] * self.patch_size[2])

        # Embed the patches
        x = self.patch_to_embedding(x)

        # Add position embeddings and concatenate CLS token
        batch_size, seq_len, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embedding[:, :(seq_len + 1)]

        # Pass through Transformer
        x = self.transformer(x)

        # Classification using the CLS token
        cls_output = x[:, 0]
        return self.mlp_head(cls_output)
# ----------------------------------------------------------------------------------------------------------------
class Trans_cross(nn.Module):
    def __init__(self, classes, modalties, classifiers_dims, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param modalties: Number of modalties
        :param classifiers_dims: Dimension of the classifier
        :param lambda_epochs: KL divergence annealing epoch during training
        """
        super(Trans_cross, self).__init__()
        self.modalties = modalties
        self.classes = classes
        self.lambda_epochs = lambda_epochs

        # ---- 2D Vision Transformer Backbone ----
        self.transformer_2DNet = fundus_build_model()
        self.intermediate_dim_2D = 768  # Typically the hidden dimension in ViT

        # ---- 3D Vision Transformer Backbone ----
        self.transformer_3DNet = UNETR_base_3DNet(num_classes=self.classes)
        self.intermediate_dim_3D = 768  # Adjusted dimension for 3D transformer

        # Cross-attention layer
        embed_dim = 2048  # Adjust based on your feature dimension
        self.cross_attention = CrossAttention(dim=embed_dim)

        # Final classification layer
        self.fc = nn.Linear(embed_dim, classes)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, X, y):
        # Get features from 2D and 3D networks
        features_2D = self.transformer_2DNet(X[0])
        features_3D = self.transformer_3DNet(X[1])

        # Reshape features for cross-attention layer
        features_2D = features_2D.unsqueeze(0)  # Add batch dimension if necessary
        features_3D = features_3D.unsqueeze(0)  # Add batch dimension if necessary

        # Apply cross-attention
        cross_features = self.cross_attention(features_2D, features_3D)

        # Flatten the cross-attended features and pass through final classifier
        combined_pred = cross_features.squeeze(0)  # Remove the unnecessary dimension
        final_pred = self.fc(combined_pred)

        loss = self.ce_loss(final_pred, y)
        loss = torch.mean(loss)
        return final_pred, loss

# ---------------------------------------------------------------------------------------------------------------------------
class transformer_2DNet(nn.Module):
    def __init__(self, classes, modalties, classifiers_dims, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param modalties: Number of modalities
        :param classifiers_dims: Dimension of the classifier
        :param lambda_epochs: KL divergence annealing epoch during training
        """
        super(transformer_2DNet, self).__init__()
        self.modalties = modalties
        self.classes = classes
        self.lambda_epochs = lambda_epochs

        # ---- 2D Transformer Backbone ----
        self.transformer_2DNet = fundus_build_model()  # SWIN-Transformer

        # ---- Intermediate Fusion Classifier ----
        self.intermediate_fc = nn.Linear(1024, classes)

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, X, y):
        # Get features from 2D and 3D networks
        features_2D = self.transformer_2DNet(X[0])

        # Adjust 2D features to match the size of 3D features
        final_pred = self.intermediate_fc(features_2D[0])
        final_pred = torch.mean(final_pred, dim=1)

        # combined_pred = torch.tensor(0)
        # print('final_pred', final_pred.shape)  # Ensure this is [batch_size, num_classes]
        # print('y', y.shape)  # Ensure this is [batch_size]

        loss = self.ce_loss(final_pred, y)

        loss = torch.mean(loss)

        return final_pred, loss


class transformer_3DNet(nn.Module):
    def __init__(self, classes, modalties, classifiers_dims, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param modalties: Number of modalities
        :param classifiers_dims: Dimension of the classifier
        :param lambda_epochs: KL divergence annealing epoch during training
        """
        super(transformer_3DNet, self).__init__()
        self.modalties = modalties
        self.classes = classes
        self.lambda_epochs = lambda_epochs

        # ---- 2D Transformer Backbone ----
        self.transformer_3DNet = UNETR_base_3DNet(num_classes=self.classes)  # SWIN-Transformer

        # ---- Intermediate Fusion Classifier ----
        self.intermediate_fc = nn.Linear(768, classes)

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, X, y):
        # Get features from 2D and 3D networks
        features_3D = self.transformer_3DNet(X[1])

        # Adjust 3D features to match the size of 3D features
        final_pred = self.intermediate_fc(features_3D[0])
        final_pred = torch.mean(final_pred, dim=1)

        # combined_pred = torch.tensor(0)
        # print('final_pred', final_pred.shape)  # Ensure this is [batch_size, num_classes]
        # print('y', y.shape)  # Ensure this is [batch_size]

        loss = self.ce_loss(final_pred, y)

        loss = torch.mean(loss)

        return final_pred, loss





class MLC_trans(nn.Module):
    def __init__(self, classes, modalties, classifiers_dims, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param modalties: Number of modalities
        :param classifiers_dims: Dimension of the classifier
        :param lambda_epochs: KL divergence annealing epoch during training
        """
        super(MLC_trans, self).__init__()
        self.modalties = modalties
        self.classes = classes
        self.lambda_epochs = lambda_epochs

        # ---- 2D Transformer Backbone ----
        self.transformer_2DNet = fundus_build_model()  # SWIN-Transformer
        self.intermediate_dim_2D = 1024  # Typically the hidden dimension in ViT

        # ---- 3D Transformer Backbone ----
        # UNETR from MONAI
        self.transformer_3DNet = UNETR_base_3DNet(num_classes=self.classes)
        self.intermediate_dim_3D = 768  # Adjusted dimension for 3D transformer

        # Adjust dimensions to match
        self.adjust_dim_2D = nn.Linear(147456, self.intermediate_dim_3D)

        # ---- Intermediate Fusion Classifier ----
        self.intermediate_fc = nn.Linear(166656, classes)

        # ---- Later Fusion Classifier ----
        self.later_fc = nn.Linear(166656, classes)

        # ---- Combined Classifier ----
        self.combined_fc = nn.Linear(classes * 2, classes)

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, X, y):
        # Get features from 2D and 3D networks
        features_2D = self.transformer_2DNet(X[0])
        features_3D = self.transformer_3DNet(X[1])

        # Intermediate fusion
        intermediate_features_2D = self._extract_intermediate_features_2D(features_2D)  # 1024
        intermediate_features_3D = self._extract_intermediate_features_3D(features_3D)  # 768

        # Adjust 2D features to match the size of 3D features
        intermediate_features_2D = self.adjust_dim_2D(intermediate_features_2D)

        # Concatenate and fuse features
        combined_intermediate_features = torch.cat([intermediate_features_2D, intermediate_features_3D], dim=1)
        intermediate_pred = self.intermediate_fc(combined_intermediate_features)

        # Later fusion
        combined_final_features = torch.cat([intermediate_features_2D, intermediate_features_3D], dim=1)
        later_pred = self.later_fc(combined_final_features)

        # Combine both methods' results
        combined_pred = torch.cat([intermediate_pred, later_pred], dim=1)

        # Final prediction, ensure it matches [batch_size, num_classes]
        final_pred = self.combined_fc(combined_pred)

        # If the output has more dimensions than expected, consider pooling
        # Uncomment if pooling is necessary:
        # final_pred = torch.mean(final_pred, dim=1)  # or torch.max(final_pred, dim=1)[0]

        # Ensure targets are in the correct format
        if len(y.shape) > 1 and y.shape[1] > 1:
            y = torch.argmax(y, dim=1)
        y = y.long()

        # print('final_pred', final_pred.shape)  # Ensure this is [batch_size, num_classes]
        # print('y', y.shape)  # Ensure this is [batch_size]

        loss = self.ce_loss(final_pred, y)
        loss = torch.mean(loss)

        return final_pred, loss, combined_pred

    def _extract_intermediate_features_2D(self, features_2D):
        # Extract intermediate features from the 2D network
        if isinstance(features_2D, tuple):
            features_2D = features_2D[0]
        return features_2D.view(features_2D.size(0), -1)

    def _extract_intermediate_features_3D(self, features_3D):
        # Extract intermediate features from the 3D network
        if isinstance(features_3D, tuple):
            features_3D = features_3D[0]
        return features_3D.view(features_3D.size(0), -1)


class MLC(nn.Module):
    def __init__(self, classes, modalties, classifiers_dims, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param views: Number of modalties
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """  
        super(MLC, self).__init__()
        self.modalties = modalties
        self.classes = classes
        self.lambda_epochs = lambda_epochs

        # ---- 2D Res2Net Backbone ----
        self.res2net_2DNet = Medical_base_2DNet(num_classes=self.classes)
        self.intermediate_dim_2D = 2048  # 假设中间层的维度

        # ---- 3D ResNet Backbone ----
        classifier_OCT_dims = classifiers_dims[0]
        self.resnet_3DNet = Medical_base_3DNet(classifier_OCT_dims, num_classes=self.classes)
        self.intermediate_dim_3D = 4096  # 假设中间层的维度

        # ---- 中间融合分类器 ----
        self.intermediate_fc = nn.Linear(self.intermediate_dim_2D + self.intermediate_dim_3D, classes)

        # ---- 后融合分类器 ----
        self.later_fc = nn.Linear(8192, classes)

        # ---- 结合两种方法的分类器 ----
        self.combined_fc = nn.Linear(classes * 2, classes)

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, X, y):
        # 获取2D和3D网络的特征
        features_2D = self.res2net_2DNet(X[0])
        features_3D = self.resnet_3DNet(X[1])
        # print('features_3D', features_3D.shape)
        # print('features_2D', features_2D.shape)

        # 中间融合方法
        intermediate_features_2D = self._extract_intermediate_features_2D(features_2D)
        intermediate_features_3D = self._extract_intermediate_features_3D(features_3D)
        combined_intermediate_features = torch.cat([intermediate_features_2D, intermediate_features_3D], 1)
        intermediate_pred = self.intermediate_fc(combined_intermediate_features)

        # 后融合方法
        combined_final_features = torch.cat([features_2D, features_3D], 1)
        later_pred = self.later_fc(combined_final_features)

        # 结合两种方法的结果
        combined_pred = torch.cat([intermediate_pred, later_pred], 1)
        final_pred = self.combined_fc(combined_pred)

        loss = self.ce_loss(final_pred, y)
        loss = torch.mean(loss)
        return final_pred, loss, combined_final_features

    def _extract_intermediate_features_2D(self, features_2D):
        # 提取2D网络的中间特征
        return features_2D[:, :self.intermediate_dim_2D]

    def _extract_intermediate_features_3D(self, features_3D):
        # 提取3D网络的中间特征
        return features_3D[:, :self.intermediate_dim_3D]



#  --------------------------------------------------------  2
class Multi_EF_ResNet(nn.Module):

    def __init__(self, classes, modalties, classifiers_dims, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param views: Number of modalties
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(Multi_EF_ResNet, self).__init__()
        self.modalties = modalties
        self.classes = classes
        self.lambda_epochs = lambda_epochs
        # ---- 2D Res2Net Backbone ----
        self.res2net_2DNet = Medical_base_2DNet(num_classes=self.classes)

        # ---- 3D ResNet Backbone ----
        classifier_OCT_dims = classifiers_dims[0]

        # --- 2D early fusion conv
        if classifier_OCT_dims[0][-1] == 248:
            self.ef_conv = nn.Sequential(
                        nn.AvgPool2d(kernel_size=1, stride=[2,2],
                         ceil_mode=True, count_include_pad=False),
                        nn.Conv2d(3, 3, 1, 1))
            # self.fc = nn.Linear(3584, classes)

        else:
            self.ef_conv = nn.Sequential(
                        nn.AvgPool2d(kernel_size=1, stride=[4,2],
                         ceil_mode=True, count_include_pad=False),
                        nn.Conv2d(3, 3, 1, 1))
            # self.fc = nn.Linear(8704, classes)

        self.resnet_3DNet = Medical_base_3DNet(classifier_OCT_dims,num_classes=self.classes)
        self.sp = nn.Softplus()
        self.ce_loss = nn.CrossEntropyLoss()
        self.fc_pro = nn.Linear(192,96)
        self.fc = nn.Linear(6656, classes)


    def forward(self, X, y):
        X0_features = self.ef_conv(X[0])
        if self.classes == 2:
            if X[1].shape[-1] == 248:
                X[1].resize_(X[1].shape[0],X[1].shape[1],X[1].shape[2],X0_features.shape[-2],X0_features.shape[-1])
                combine_features = torch.cat([X0_features.unsqueeze(1),X[1].permute(0,1,2,4,3)],2)

            else:
                # print("X0_features.unsqueeze(1)",X0_features.unsqueeze(1).shape)
                # print("X[1].permute(0,1,2,4,3)",X[1].permute(0,1,2,4,3).shape)
                X0_features  = self.fc_pro(X0_features.unsqueeze(1))
                combine_features = torch.cat([X0_features ,X[1].permute(0,1,2,4,3)],2)
        else:
            X0_features = self.fc_pro(X0_features.unsqueeze(1))
            combine_features = torch.cat([X0_features, X[1].permute(0, 1, 2, 4, 3)], 2)
            # combine_features = torch.cat([X0_features.unsqueeze(1),X[1]],2)

        # backboneout_1 = self.res2net_2DNet(X[0])
        backboneout_2 = self.resnet_3DNet(combine_features)
        pred = self.fc(backboneout_2)
        loss = self.ce_loss(pred, y)

        loss = torch.mean(loss)
        return pred, loss


class CBAM2D(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAM2D, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class CBAM3D(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAM3D, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv3d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv3d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv3d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class Multi_CBAM_ResNet(nn.Module):

    def __init__(self, classes, modalties, classifiers_dims, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param views: Number of modalties
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(Multi_CBAM_ResNet, self).__init__()
        self.modalties = modalties
        self.classes = classes
        self.lambda_epochs = lambda_epochs
        # ---- 2D Res2Net Backbone ----
        self.res2net_2DNet = Medical_feature_2DNet(num_classes=self.classes)

        # ---- 3D ResNet Backbone ----
        classifier_OCT_dims = classifiers_dims[0]

        # ---- CBAM Layer----
        self.CBAM2D_layer = CBAM2D(2048)
        self.CBAM3D_layer = CBAM3D(512)
        # GAP
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.resnet_3DNet = Medical_feature_3DNet(classifier_OCT_dims,num_classes=self.classes)
        self.sp = nn.Softplus()
        # if classifier_OCT_dims[0][0] == 128:
        #     self.fc = nn.Linear(2048 + 8192, classes) # MMOCTF
        # else:
        #     self.fc = nn.Linear(2048 + 3072, classes) # OLIVES
        self.fc = nn.Linear(8192, classes)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, X, y):
        backboneout_1 = self.res2net_2DNet(X[0])
        backboneout_2 = self.resnet_3DNet(X[1])
        backboneout_1_CBAM = self.CBAM2D_layer(backboneout_1)
        backboneout_2_CBAM = self.CBAM3D_layer(backboneout_2)
        backboneout_1_CBAM_GAP = self.avgpool(backboneout_1_CBAM)
        backboneout_2_CBAM_GAP = self.avgpool(backboneout_2_CBAM)
        backboneout_1_CBAM_GAP = backboneout_1_CBAM_GAP.view(backboneout_1_CBAM_GAP.size(0), -1)
        backboneout_2_CBAM_GAP = backboneout_2_CBAM_GAP.view(backboneout_2_CBAM_GAP.size(0), -1)
        combine_features = torch.cat([backboneout_1_CBAM_GAP,backboneout_2_CBAM_GAP],1)
        pred = self.fc(combine_features)
        loss = self.ce_loss(pred, y)

        loss = torch.mean(loss)
        return pred, loss, combine_features


class Multi_ensemble_ResNet(nn.Module):

    def __init__(self, classes, modalties, classifiers_dims, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param views: Number of modalties
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(Multi_ensemble_ResNet, self).__init__()
        self.modalties = modalties
        self.classes = classes
        self.lambda_epochs = lambda_epochs
        # ---- 2D Res2Net Backbone ----
        self.res2net_2DNet = Medical_base2_2DNet(num_classes=self.classes)

        # ---- 3D ResNet Backbone ----
        classifier_OCT_dims = classifiers_dims[0]
        self.resnet_3DNet = Medical_base_3DNet(classifier_OCT_dims,num_classes=self.classes)
        self.sp = nn.Softplus()
        self.fc = nn.Linear(2048 + 8192, classes)
        self.ce_loss = nn.CrossEntropyLoss()


    def forward(self, X, y):
        backboneout_1 = self.res2net_2DNet(X[0])
        backboneout_2 = self.resnet_3DNet(X[1])
        combine_features = torch.cat([backboneout_1,backboneout_2],1)
        pred = self.fc(combine_features)
        loss = self.ce_loss(pred, y)

        loss = torch.mean(loss)
        return pred, loss

class Multi_ensemble_3D_ResNet(nn.Module):

    def __init__(self, classes, modalties, classifiers_dims, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param views: Number of modalties
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(Multi_ensemble_3D_ResNet, self).__init__()
        self.modalties = modalties
        self.classes = classes
        self.lambda_epochs = lambda_epochs
        # ---- 2D Res2Net Backbone ----
        self.res2net_2DNet = Medical_base_2DNet(num_classes=self.classes)

        # ---- 3D ResNet Backbone ----
        classifier_OCT_dims = classifiers_dims[0]
        self.resnet_3DNet = Medical_base2_3DNet(classifier_OCT_dims,num_classes=self.classes)
        self.sp = nn.Softplus()
        self.fc = nn.Linear(2048 + 8192, classes)
        self.ce_loss = nn.CrossEntropyLoss()


    def forward(self, X, y):
        backboneout_1 = self.res2net_2DNet(X[0])
        backboneout_2 = self.resnet_3DNet(X[1])
        combine_features = torch.cat([backboneout_1,backboneout_2],1)
        pred = self.fc(combine_features)
        loss = self.ce_loss(pred, y)

        loss = torch.mean(loss)
        return pred, loss


class Multi_dropout_ResNet(nn.Module):

    def __init__(self, classes, modalties, classifiers_dims, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param views: Number of modalties
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(Multi_dropout_ResNet, self).__init__()
        self.modalties = modalties
        self.classes = classes
        self.lambda_epochs = lambda_epochs
        # ---- 2D Res2Net Backbone ----
        self.res2net_2DNet = Medical_base_dropout_2DNet(num_classes=self.classes)

        # ---- 3D ResNet Backbone ----
        classifier_OCT_dims = classifiers_dims[0]
        self.resnet_3DNet = Medical_base_dropout_3DNet(classifier_OCT_dims,num_classes=self.classes)
        self.sp = nn.Softplus()

        if classifier_OCT_dims[0][0] == 128:
            self.fc = nn.Linear(2048 + 8192, classes) # MMOCTF
        else:
            self.fc = nn.Linear(2048 + 3072, classes) # OLIVES

        self.ce_loss = nn.CrossEntropyLoss()


    def forward(self, X, y):
        backboneout_1 = self.res2net_2DNet(X[0])
        backboneout_2 = self.resnet_3DNet(X[1])
        combine_features = torch.cat([backboneout_1,backboneout_2],1)
        pred = self.fc(combine_features)
        loss = self.ce_loss(pred, y)

        loss = torch.mean(loss)
        return pred, loss