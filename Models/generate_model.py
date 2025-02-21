import torch
from torch import nn
import os
import sys
from Models import resnet
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# sys.path.append(parent_dir )

def generate_model(model_type='resnet', model_depth=50,
                   input_W=224, input_H=224, input_D=224, resnet_shortcut='B',
                   no_cuda=False, gpu_id=[0],
                   pretrain_path = '/mnt/sdb/feilong/AAAI25/Confidence_MedIA/pretrain/resnet_10_23dataset.pth',
                   nb_class=1):
    assert model_type  in [
        'resnet'
    ]

    if model_type == 'resnet':
        assert model_depth in [10, 18, 34, 50, 101, 152, 200]

        if model_depth == 10:
            model = resnet.resnet10(
                sample_input_W=input_W,
                sample_input_H=input_H,
                sample_input_D=input_D,
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda,
                num_seg_classes=nb_class)
            fc_input = 256
        elif model_depth == 18:
            model = resnet.resnet18(
                sample_input_W=input_W,
                sample_input_H=input_H,
                sample_input_D=input_D,
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda,
                num_seg_classes=nb_class)
            fc_input = 512
        elif model_depth == 34:
            model = resnet.resnet34(
                sample_input_W=input_W,
                sample_input_H=input_H,
                sample_input_D=input_D,
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda,
                num_seg_classes=nb_class)
            fc_input = 512
        elif model_depth == 50:
            model = resnet.resnet50(
                sample_input_W=input_W,
                sample_input_H=input_H,
                sample_input_D=input_D,
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda,
                num_seg_classes=nb_class)
            fc_input = 2048
        elif model_depth == 101:
            model = resnet.resnet101(
                sample_input_W=input_W,
                sample_input_H=input_H,
                sample_input_D=input_D,
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda,
                num_seg_classes=nb_class)
            fc_input = 2048
        elif model_depth == 152:
            model = resnet.resnet152(
                sample_input_W=input_W,
                sample_input_H=input_H,
                sample_input_D=input_D,
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda,
                num_seg_classes=nb_class)
            fc_input = 2048
        elif model_depth == 200:
            model = resnet.resnet200(
                sample_input_W=input_W,
                sample_input_H=input_H,
                sample_input_D=input_D,
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda,
                num_seg_classes=nb_class)
            fc_input = 2048
        else:
            model = resnet.resnet10(
                sample_input_W=input_W,
                sample_input_H=input_H,
                sample_input_D=input_D,
                shortcut_type=resnet_shortcut,
                no_cuda=no_cuda,
                num_seg_classes=nb_class)
            fc_input = 256

        model.conv_seg = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(),
                                       nn.Linear(in_features=fc_input, out_features=nb_class, bias=True))

        if not no_cuda:
            if len(gpu_id) > 1:
                model = model.cuda()
                model = nn.DataParallel(model, device_ids=gpu_id)
                net_dict = model.state_dict()
            else:
                import os
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id[0])
                model = model.cuda()
                model = nn.DataParallel(model, device_ids=None)
                net_dict = model.state_dict()
        else:
            net_dict = model.state_dict()
        # root = /mnt/sdb/tangfeilong/Retinal_OCT/Confidence_MedIA
        print('loading pretrained model {}'.format(pretrain_path))
        pretrain = torch.load(r"C:\Users\yuqinkai\PycharmProjects\Code_AAAI\pretrain\resnet_10_23dataset.pth")
        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
        # k 是每一层的名称，v是权重数值
        net_dict.update(pretrain_dict)  # 字典 dict2 的键/值对更新到 dict 里。
        model.load_state_dict(net_dict)  # model.load_state_dict()函数把加载的权重复制到模型的权重中去

        print("-------- pre-train model load successfully --------")

    return model
