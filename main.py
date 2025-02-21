import sys
import os
import h5py
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir )
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/mnt/sdb/feilong/AAAI25/Confidence_MedIA/')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/mnt/sdb/feilong/AAAI25/Confidence_MedIA/dataloader/')))
from sklearn.manifold import TSNE
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from baseline_models_fusion import Multi_ResNet
from sklearn.model_selection import KFold
from matplotlib import cm
from data_gamma import GAMMA_dataset
from matplotlib.colors import ListedColormap

from metrics import cal_ece
import warnings
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import cohen_kappa_score
from metrics2 import calc_aurc_eaurc,calc_nll_brier
import torch.nn.functional as F
import logging
import time
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import label_binarize
import torch.nn as nn
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from perturbot.perturbot.match import get_coupling_egw_labels_ott, get_coupling_fot
from torch.utils.data import Dataset
    
def log_args(log_file):

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # args FileHandler to save log file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)

def loss_plot(args,loss):
    num = args.end_epochs
    x = [i for i in range(num)]
    plot_save_path = r'results/plot/'
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    save_loss = plot_save_path+str(args.model_name)+'_'+str(args.batch_size)+'_'+str(args.dataset)+'_'+str(args.end_epochs)+'_loss.jpg'
    list_loss = list(loss)
    plt.figure()
    plt.plot(x,loss,label='loss')
    plt.legend()
    plt.savefig(save_loss)



def save_results(filename, epoch, loss_meter, acc, precision, recall, f1, auc, kappa, specificity):
    with open(filename, 'a') as f:
        line = (f"Epoch: {epoch}, "
                f"Loss: {loss_meter.avg:.6f}, "
                f"Accuracy: {acc:.4f}, "
                f"Precision: {precision:.4f}, "
                f"Recall: {recall:.4f}, "
                f"F1 Score: {f1:.4f}, "
                f"AUC: {auc:.4f},"
                f"Kappa: {kappa:.4f}, "
                f"Specificity: {specificity:.4f}, "
                )
        f.write(line + "\n")


def metrics_plot(arg,name,*args):
    num = arg.end_epochs
    names = name.split('&')
    metrics_value = args
    i=0
    x = [i for i in range(num)]
    plot_save_path = r'results/plot/'
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    save_metrics = plot_save_path + str(arg.model_name) + '_' + str(arg.batch_size) + '_' + str(arg.dataset) + '_' + str(arg.end_epochs) + '_'+name+'.jpg'
    plt.figure()
    for l in metrics_value:
        plt.plot(x,l,label=str(names[i]))
        #plt.scatter(x,l,label=str(l))
        i+=1
    plt.legend()
    plt.savefig(save_metrics)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def find_in_u(list_acc,in_list,u_list,class_num=0):
    for i in range(len(list_acc)):
        if list_acc[i] == class_num:
            in_list.append(i)
    in_u_list = np.zeros(len(in_list))
    for j in range(len(in_list)):
        in_u_list[j] = (u_list[in_list[j]])
    return in_u_list

def train(epoch,train_loader,model, T_feature_2, best_acc=0.0):
    correct_num, data_num = 0, 0
    model.train()
    loss_meter = AverageMeter()
    all_targets = []
    all_predictions = []
    all_probabilities = []
    labels = []
    oct_property = []
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        for v_num in range(len(data)):
            data[v_num] = Variable(data[v_num].float().cuda())
        target = Variable(target.long().cuda())
        # labels.append(target)
        data_num += target.size(0)
        # target = Variable(np.array(target)).cuda())
        # refresh the optimizer
        optimizer.zero_grad()
        pred, loss,ot_loss= model(X=data, y=target,T_feature_2=T_feature_2,training=True)
        # oct_property.append(property_oct)
        print(pred.shape)
        print("ot_loss",ot_loss)
        print("loss",loss)
        predicted = pred.argmax(dim=-1)
        correct_num += (predicted == target).sum().item()
        all_targets.extend(target.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        probabilities = torch.nn.functional.softmax(pred, dim=1)
        all_probabilities.extend(probabilities.detach().cpu().numpy())  #
        loss = loss.mean()
        loss = loss+ot_loss
        # compute gradients and take step
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())
    aver_acc = correct_num / data_num
    # plot_tsne(oct_property,labels)
    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    all_probabilities = np.array(all_probabilities)
    if len(set(all_targets)) == 2:
        auc = roc_auc_score(all_targets, all_probabilities[:, 1])
    else:
        # For multi-class classification
        all_targets_one_hot = label_binarize(all_targets, classes=[0, 1, 2, 3])
        auc = roc_auc_score(all_targets_one_hot, all_probabilities, multi_class='ovr')
    print("All targets distribution:", np.bincount(all_targets))
    print("All predictions distribution:", np.bincount(all_predictions))
    conf_matrix = confusion_matrix(all_targets, all_predictions)
    tn = conf_matrix[0, 0]
    fp = conf_matrix[0, 1]
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    kappa = cohen_kappa_score(all_targets, all_predictions)
    print(f'Train Epoch: {epoch} \tLoss: {loss_meter.avg:.6f} \tAccuracy: {aver_acc:.4f}')
    print(f'Precision: {precision:.4f} \tRecall: {recall:.4f} \tF1 Score: {f1:.4f} \tAUC: {auc:.4f}')
    print(f'Specificity: {specificity:.4f} \tKappa: {kappa:.4f}')
    save_results(r'C:\Users\yuqinkai\PycharmProjects\Code_AAAI\base_code\res\log_txt\Glu_2_res2_dr_fusion_ab.txt', epoch, loss_meter, aver_acc, precision, recall, f1, auc, kappa, specificity)
    return loss_meter
def plot_tsne(list_oct_feature,labels):
    list_oct_feature = [feature.cpu().detach().numpy() for feature in list_oct_feature]  # 将每个张量移动到 CPU 并转换为 NumPy 数组
    list_oct_feature = np.concatenate(list_oct_feature, axis=0)  # 将所有 fusion_feature 合并为一个数组
    print('list_oct_feature',list_oct_feature.shape)
    all_labels = [label.cpu().detach().numpy() for label in labels]  # 将每个张量移动到 CPU 并转换为 NumPy 数组
    all_labels = np.concatenate(all_labels, axis=0)  # 将所有标签合并为一个数组
    unique_labels = np.unique(all_labels)
    custom_colors = ['red', 'yellow', 'blue', 'green']
    custom_cmap = ListedColormap(custom_colors)
    colors = cm.get_cmap('Set2', 8)  # 获取 Set2 颜色映射，最多 8 种颜色
    tsne = TSNE(n_components=2, random_state=13)
    oct_features_tsne = tsne.fit_transform(list_oct_feature)
    scatter2 = plt.scatter(oct_features_tsne[:, 0], oct_features_tsne[:, 1], c=all_labels,cmap = custom_cmap,s = 10)
    handles, labels = scatter2.legend_elements()
    plt.axis('off')
    plt.show()
def val(current_epoch, val_loader, model, T_feature_2, best_acc):
    model.eval()
    loss_meter = AverageMeter()
    correct_num, data_num = 0, 0
    all_targets = []
    all_predictions = []
    all_probabilities = []
    list_oct_feature = []
    list_fundus_feature = []
    tsne_pred_oct_feature = []
    tsne_pred_fundus_feature = []
    property_oct = []
    labels = []
    for batch_idx, (data, target) in enumerate(tqdm(val_loader)):
        for v_num in range(len(data)):
            data[v_num] = Variable(data[v_num].float().cuda())
        data_num += target.size(0)
        with torch.no_grad():
            target = Variable(target.long().cuda())
            # labels.append(target)
            pred, loss= model(X=data, y=target,T_feature_2=T_feature_2,training=False)
            # list_oct_feature.append(combine_features)
            # property_oct.append(property)
            loss = loss.mean()
            predicted = pred.argmax(dim=-1)
            correct_num += (predicted == target).sum().item()
            loss_meter.update(loss.item())
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            probabilities = torch.nn.functional.softmax(pred, dim=1)
            all_probabilities.extend(probabilities.detach().cpu().numpy())
    # plot_tsne(list_oct_feature,labels)
    aver_acc = correct_num / data_num
    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    all_probabilities = np.array(all_probabilities)
    print("all_targets",all_targets)
    print("all_predictions",all_predictions)
    if len(set(all_targets)) == 2:
        auc = roc_auc_score(all_targets, all_probabilities[:, 1])
    else:
        all_targets_one_hot = label_binarize(all_targets, classes=[0, 1, 2, 3])
        auc = roc_auc_score(all_targets_one_hot, all_probabilities, multi_class='ovr')
    conf_matrix = confusion_matrix(all_targets, all_predictions)
    tn = conf_matrix[0, 0]
    fp = conf_matrix[0, 1]
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    kappa = cohen_kappa_score(all_targets, all_predictions)
    print(f'Train Epoch: {epoch} \tLoss: {loss_meter.avg:.6f} \tAccuracy: {aver_acc:.4f}')
    print(f'Precision: {precision:.4f} \tRecall: {recall:.4f} \tF1 Score: {f1:.4f} \tAUC: {auc:.4f}')
    print(f'Specificity: {specificity:.4f} \tKappa: {kappa:.4f}')
    save_results(r'C:\Users\yuqinkai\PycharmProjects\Code_AAAI\base_code\res\log_txt\Glu_2_res2_dr_fusion.txt', current_epoch, loss_meter, aver_acc, precision, recall, f1, auc, kappa, specificity)

    if f1 > best_acc:
        best_acc = f1
        print('===========> Save best model!')
        file_name = os.path.join(args.save_dir, f"{args.model_name}_val_2_checkpoint_f1_{best_acc}.pth")
        torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, file_name)
    return loss_meter.avg, best_acc
def test(current_epoch, test_loader, model):
    model.eval()
    loss_meter = AverageMeter()
    correct_num, data_num = 0, 0
    all_targets = []
    all_predictions = []
    all_probabilities = []
    for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
        for v_num in range(len(data)):
            data[v_num] = Variable(data[v_num].float().cuda())
        data_num += target.size(0)
        with torch.no_grad():
            target = Variable(target.long().cuda())
            pred, loss = model(data, target)
            loss = loss.mean()
            predicted = pred.argmax(dim=-1)
            correct_num += (predicted == target).sum().item()
            loss_meter.update(loss.item())
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            probabilities = torch.nn.functional.softmax(pred, dim=1)
            all_probabilities.extend(probabilities.detach().cpu().numpy())
    aver_acc = correct_num / data_num
    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    all_probabilities = np.array(all_probabilities)
    # Calculate AUC for binary classification
    if len(set(all_targets)) == 2:
        auc = roc_auc_score(all_targets, all_probabilities[:, 1])
    else:
        # For multi-class classification
        all_targets_one_hot = label_binarize(all_targets, classes=[0, 1, 2, 3])
        auc = roc_auc_score(all_targets_one_hot, all_probabilities, multi_class='ovr')
    print('====> acc: {:.4f}'.format(aver_acc))
    print(f'Precision: {precision:.4f} \tRecall: {recall:.4f} \tF1 Score: {f1:.4f} \tAUC: {auc:.4f}')
    save_results('test_results.txt', current_epoch, loss_meter, aver_acc, precision, recall, f1, auc)
    return loss_meter.avg, best_acc
def test_ensemble(args, test_loader,models,epoch):
    if args.dataset == 'MGamma':
        deepen_times = 4
    else:
        deepen_times = 5

    # load ensemble models
    load_model=[]
    # load_model[0]=.23
    for i in range(deepen_times):
        print(i+1)
        if args.num_classes == 2:
            load_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                 args.save_dir,
                                 'Multi_DE' +str(i+1) + '_ResNet_' + args.dataset +'_'+ args.folder + '_epoch_{}.pth'.format(args.test_epoch))
        else:
            load_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                 args.save_dir,
                                 'Multi_DE' +str(i+1) + '_ResNet_' + args.dataset +'_'+ args.folder +  '_best_epoch.pth')

        load_model.append(torch.load(load_file))
        # KK =model[i]
        models[i].load_state_dict(load_model[i]['state_dict'])
    print('Successfully load all ensemble models')
    for model in models:
        model.eval()
    list_acc = []
    u_list =[]
    in_list = []
    label_list = []
    ece_list=[]
    prediction_list = []
    probability_list = []
    one_hot_label_list = []
    one_hot_probability_list = []
    correct_list=[]
    correct_num, data_num = 0, 0
    epoch_auc = 0
    start_time = time.time()
    time_list= []
    nll_list= []
    brier_list = []
    for batch_idx, (data, target) in enumerate(test_loader):
        for v_num in range(len(data)):
            data[v_num] = Variable(data[v_num].float().cuda())
        data_num += target.size(0)
        pred = torch.zeros(1,args.num_classes).cuda()
        with torch.no_grad():
            target = Variable(target.long().cuda())
            for i in range(deepen_times):
                # print('ensemble model:{}'.format(i))
                pred_i, _ = models[i](data, target)
                pred += pred_i
            pred = pred/deepen_times
            elapsed_time = time.time() - start_time
            time_list.append(elapsed_time)
            predicted = pred.argmax(dim=-1)
            correct_num += (predicted == target).sum().item()
            correct = (predicted == target)
            list_acc.append((predicted == target).sum().item())
            prediction_list.append(predicted.cpu().detach().float().numpy())
            label_list.append(target.cpu().detach().float().numpy())
            correct_list.append(correct.cpu().detach().float().numpy())
            probability = torch.softmax(pred, dim=1).cpu().detach().float().numpy()
            probability_list.append(torch.softmax(pred, dim=1).cpu().detach().float().numpy()[:,1])
            one_hot_probability_list.append(torch.softmax(pred, dim=1).squeeze(dim=0).cpu().detach().float().numpy())
            one_hot_label = F.one_hot(target, num_classes=args.num_classes).squeeze(dim=0).cpu().detach().float().numpy()
            one_hot_label_list.append(one_hot_label)
            ece_list.append(cal_ece(torch.squeeze(pred), target))
            # NLL brier
            nll, brier = calc_nll_brier(probability, pred, target, one_hot_label)
            nll_list.append(nll)
            brier_list.append(brier)
    logging.info('Single sample test time consumption {:.2f} seconds!'.format(sum(time_list)/len(time_list)))
    print('Single sample test time consumption {:.2f} seconds!'.format(sum(time_list)/len(time_list)))
    if args.num_classes > 2:
        epoch_auc = metrics.roc_auc_score(one_hot_label_list, one_hot_probability_list, multi_class='ovo')
    else:
        epoch_auc = metrics.roc_auc_score(label_list, probability_list)
    avg_acc = correct_num/data_num
    avg_ece = sum(ece_list)/len(ece_list)
    avg_kappa = cohen_kappa_score(prediction_list, label_list)
    F1_Score = f1_score(y_true=label_list, y_pred=prediction_list, average='weighted')
    Recall_Score = recall_score(y_true=label_list, y_pred=prediction_list, average='weighted')
    aurc, eaurc = calc_aurc_eaurc(probability_list, correct_list)
    if not os.path.exists(os.path.join(args.save_dir, "{}_{}_{}".format(args.model_name,args.dataset,args.folder))):
        os.makedirs(os.path.join(args.save_dir, "{}_{}_{}".format(args.model_name,args.dataset,args.folder)))

    avg_nll = sum(nll_list) / len(nll_list)
    avg_brier = sum(brier_list) / len(brier_list)
    with open(os.path.join(args.save_dir, "{}_{}_{}_Metric.txt".format(args.model_name, args.dataset, args.folder)),
              'w') as Txt:
        Txt.write(
            "Acc: {}, AUC: {}, AURC: {}, EAURC: {},  NLL: {}, BRIER: {}, F1_Score: {}, Recall_Score: {}, Kappa_Score: {}, ECE: {}\n".format(
                round(avg_acc, 6), round(epoch_auc, 6), round(aurc, 6), round(eaurc, 6), round(avg_nll, 6),
                round(avg_brier, 6), round(F1_Score, 6), round(Recall_Score, 6), round(avg_kappa, 6),
                round(avg_ece, 6)
            ))
    return avg_acc, epoch_auc, aurc, eaurc, avg_nll, avg_brier, F1_Score, Recall_Score, avg_kappa, avg_ece
def feature_extract(feature_extractor,data_Loader):
    list_fundus_features = []
    list_oct_features = []
    list_labels = []
    with torch.no_grad():  # 在不计算梯度的情况下进行
        model.eval()
        for (features, targets) in tqdm(data_Loader):
            for v_num in range(len(features)):
                features[v_num] = Variable(features[v_num].float().cuda())
            # ([BS, 3, 384, 384])    
            # ([BS, 1, 96, 96, 96])
            # print("features[0].shape",features[0].shape)
            # print("features[1].shape",features[1].shape)
            features_1= feature_extractor.res2net_2DNet(features[0])
            print("features_1.shape",features_1.shape)
            features_2= feature_extractor.resnet_3DNet(features[1])
            print("features_2.shape",features_2.shape)
            list_fundus_features.append(features_1.cpu().numpy())
            list_oct_features.append(features_2.cpu().numpy())
            list_labels.append(targets.numpy())
        input_dim = list_fundus_features[0].shape[1]
        fundus_features = np.concatenate(list_fundus_features, axis=0)
        fundus_features = torch.tensor(fundus_features, dtype=torch.float32)
        oct_features = np.concatenate(list_oct_features, axis=0)
        oct_features = torch.tensor(oct_features, dtype=torch.float32)
        train_labels = np.concatenate(list_labels, axis=0)
        train_labels = torch.tensor(train_labels, dtype=torch.long)
    return fundus_features, oct_features, train_labels
def group_features_by_label(y, p, num_classes=3):
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
def convert_dataset_to_hdf5(dataset, output_path):
    """将现有的dataset转换为HDF5格式"""
        # 先获取一个样本来确定形状
    sample_data, _ = dataset[0]
    fundus_shape = sample_data[0].shape  # 获取fundus图像的形状
    oct_shape = sample_data[1].shape     # 获取OCT图像的形状
    print("Fundus shape:", fundus_shape)
    print("OCT shape:", oct_shape)
    with h5py.File(output_path, 'w') as f:
        # 创建数据集
        fundus_dataset = f.create_dataset(
            'fundus_images', 
            shape=(len(dataset), 3, 384, 384),  # fundus_img_size
            dtype=np.float32,
            compression='gzip'
        )
        
        oct_dataset = f.create_dataset(
            'oct_images', 
            shape=(len(dataset), 1, 96, 96, 96),  # oct_img_size
            dtype=np.float32,
            compression='gzip'
        )
        
        labels_dataset = f.create_dataset(
            'labels', 
            shape=(len(dataset),),
            dtype=np.int64
        )
        
        # 保存文件列表
        file_list_dataset = f.create_dataset(
            'file_list',
            shape=(len(dataset),),
            dtype=h5py.special_dtype(vlen=str)
        )
        
        # 转换数据
        for idx in tqdm(range(len(dataset)), desc="Converting to HDF5"):
            data, label = dataset[idx]
            
            # 保存数据
            fundus_dataset[idx] = data[0].numpy()  # fundus图像
            oct_dataset[idx] = data[1].numpy()     # OCT图像
            labels_dataset[idx] = label
            file_list_dataset[idx] = dataset.file_list[idx][0]  # 保
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--end_epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--test_epoch', type=int, default=198, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--lambda_epochs', type=int, default=50, metavar='N',
                        help='gradually increase the value of lambda from 0 to 1')
    parser.add_argument('--modal_number', type=int, default=2, metavar='N',
                        help='modalties number')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--save_dir', default=r'C:\Users\yuqinkai\PycharmProjects\Code_AAAI\base_code\res\pth\dr_fusion\save', type=str)
    parser.add_argument("--model_name", default="Multi_ResNet", type=str, help="Multi_ResNet")
    parser.add_argument("--dataset", default="MMOCTF", type=str, help="MMOCTF/MGamma/Gamma/OLIVES")
    parser.add_argument("--folder", default="folder0", type=str, help="folder0/folder1/folder2/folder3/folder4")
    parser.add_argument("--mode", default="test", type=str, help="train/test/train&test")
    parser.add_argument("--model_base", default="transformer", type=str, help="transformer/cnn")
    parser.add_argument("--condition", default="noise", type=str, help="noise/normal")
    parser.add_argument("--condition_name", default="Gaussian", type=str, help="Gaussian/SaltPepper/All")
    parser.add_argument("--Condition_SP_Variance", default=0.005, type=int, help="Variance: 0.01/0.1")
    parser.add_argument("--Condition_G_Variance", default=0.05, type=int, help="Variance: 15/1/0.1")
    args = parser.parse_args()
    args.seed_idx = 11
    Condition_G_Variance = [0.1,0.2,0.3,0.4,0.5]
    args.dataset =="MGamma"
    args.modalties_name = ["FUN", "OCT"]
    # args.dims = [[(128, 256, 128)], [(512, 512)]]
    args.dims = [[(96, 96, 96)], [(384, 384)]]

    args.num_classes = 2
    args.modalties = len(args.dims)
    # args.base_path = r'C:\Users\yuqinkai\PycharmProjects\Code_AAAI\base_code\res\task1_gamma_grading\Glaucoma_grading/'
    # args.data_path = r'C:\Users\yuqinkai\PycharmProjects\Code_AAAI\base_code\res\task1_gamma_grading\Glaucoma_grading\training\multi-modality_images'
    # args.base_path = r'D:\Glaucoma_Harvard_enhance\mnt\sdb\feilong\Retinal_OCT\Medical_data\Glaucoma_Harvard_enhance\Train/'
    # args.data_path = r'D:\Glaucoma_Harvard_enhance\mnt\sdb\feilong\Retinal_OCT\Medical_data\Glaucoma_Harvard_enhance\Train\Training'
    # args.base_path = r'/mnt/datastore1/qinkaiyu/oct_fundus/mnt/sdb/feilong/Retinal_OCT/Medical_data/Glaucoma_Harvard_enhance/Train/'
    # args.data_path = r'/mnt/datastore1/qinkaiyu/oct_fundus/mnt/sdb/feilong/Retinal_OCT/Medical_data/Glaucoma_Harvard_enhance/Train/Training'
    # args.base_path = r'D:\AMD\mnt\sdb\feilong\Retinal_OCT\Medical_data\AMD\train/'
    # args.data_path = r'D:\AMD\mnt\sdb\feilong\Retinal_OCT\Medical_data\AMD\train\Image'
    args.base_path = r'D:\DR\mnt\sdb\feilong\Retinal_OCT\Medical_data\DR\train/'
    args.data_path = r'D:\DR\mnt\sdb\feilong\Retinal_OCT\Medical_data\DR\train/Image_new'
    filelists = os.listdir(args.data_path)
    kf = KFold(n_splits=5, shuffle=True, random_state=10)
    y = kf.split(filelists)
    count = 0
    train_filelists = [[], [], [], [], []]
    val_filelists = [[], [], [], [], []]
    for tidx, vidx in y:
        train_filelists[count], val_filelists[count] = np.array(filelists)[tidx], np.array(filelists)[vidx]
        count = count + 1
    f_folder = int(args.folder[-1])
    train_dataset = GAMMA_dataset(args, dataset_root=args.data_path,
                                      oct_img_size=args.dims[0],
                                      fundus_img_size=args.dims[1],
                                      mode='train',
                                      label_file=args.base_path + 'train.xlsx',
                                      filelists=np.array(train_filelists[f_folder]))
    all_train_dataset = GAMMA_dataset(args, dataset_root=args.data_path,
                                      oct_img_size=args.dims[0],
                                      fundus_img_size=args.dims[1],
                                      mode='train',
                                      label_file=args.base_path + 'train.xlsx',
                                      filelists=np.array(filelists))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    tsne_train_loader = torch.utils.data.DataLoader(all_train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = GAMMA_dataset(args, dataset_root=args.data_path,
                                    oct_img_size=args.dims[0],
                                    fundus_img_size=args.dims[1],
                                    mode='val',
                                    label_file=args.base_path + 'train.xlsx',
                                    filelists=np.array(val_filelists[f_folder]), )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)
    args.modalties_name = ["FUN", "OCT"]
    args.modalties = len(args.dims)
    model = Multi_ResNet(args.num_classes, args.modal_number, args.dims, args.lambda_epochs)

    N_mini_batches = len(train_loader)
    print('The number of training images = %d' % N_mini_batches)
    seed_num = list(range(1,11))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    model.cuda()
    best_acc = 0
    loss_list = []
    acc_list = []
    print(args.model_name)
    epoch = 0
    print('===========Train begining!===========')
    for epoch in range(args.start_epoch, args.end_epochs + 1):

        T_feature_2 = None
        model.train()
        print('Epoch {}/{}'.format(epoch, args.end_epochs))
        epoch_loss = train(epoch, train_loader, model, T_feature_2, best_acc=0.0)
        print("epoch %d avg_loss:%0.3f" % (epoch, epoch_loss.avg))
        print('===========Val begining!===========')
                # use for val
        fundus_features, oct_features, train_labels = feature_extract(model, train_loader)
        grouped_fudus_feature = group_features_by_label(train_labels,fundus_features.cpu().detach().numpy())
        grouped_oct_feature = group_features_by_label(train_labels,oct_features.cpu().detach().numpy())
            # fundus2oct
        T_dict, log = get_coupling_egw_labels_ott((grouped_fudus_feature,grouped_oct_feature))
        group_fudus_labels = sorted(grouped_fudus_feature.keys())
        group_oct_labels = sorted(grouped_oct_feature.keys())
        T_feature, fm_log = get_coupling_fot((grouped_fudus_feature,grouped_oct_feature), T_dict)
            # oct2fundus
        T_dict_2, log = get_coupling_egw_labels_ott((grouped_oct_feature,grouped_fudus_feature))
        group_oct_labels = sorted(grouped_oct_feature.keys())
        group_fudus_labels = sorted(grouped_fudus_feature.keys())
        T_feature_2, fm_log_2 = get_coupling_fot((grouped_oct_feature,grouped_fudus_feature), T_dict_2)
        val_loss, best_acc = val(epoch,tsne_train_loader,model,T_feature_2,best_acc)



