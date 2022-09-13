import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network as network
import loss as loss
import pre_process as prep
from torch.utils.data import DataLoader
import lr_schedule
from data_list import ImageList
from torch import linalg as LA
import math

def calc_pm(iter_num,  max_iter=5000.0):
    high = 1.0
    low = 0.0
    alpha = 10.0
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

class MyDataset(ImageList):
    def __init__(self, cfg, transform):
        self.btw_data = ImageList(cfg, transform=transform)
        self.imgs = self.btw_data.imgs

    def __getitem__(self, index):
        data, target = self.btw_data[index]
        return data, target, index

    def __len__(self):
        return len(self.btw_data)


class data_batch:
    def __init__(self, gt_data, batch_size: int, drop_last: bool, gt_flag: bool, num_class: int, num_batch: int) -> None:
        # if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
        #         batch_size <= 0:
        #     raise ValueError("batch_size should be a positive integer value, "
        #                      "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        gt_data = gt_data.astype(dtype=int)

        self.class_num = num_class
        self.batch_num = num_batch

        self.random_loader = False
        self.all_data = np.arange(len(gt_data))
        self.data_len = len(gt_data)

        self.rl_len= math.floor(self.data_len/ self.batch_num)

        self.i_range = len(gt_data)
        self.s_list = []
        if gt_flag == False:
            self.random_loader = True
            self.set_length(self.rl_len)
            self.i_range = self.rl_len
        else:
            for c_iter in range(self.class_num):
                cur_data = np.where(gt_data == c_iter)[0]
                self.s_list.append(cur_data)
                cur_length = math.floor((len(cur_data) * self.class_num) / self.batch_num)
                if(cur_length < self.data_len):
                    self.set_length(cur_length)
                    self.i_range = len(cur_data)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.prob_mat = np.zeros(())
        self.idx = 0
        self.c_iter = 0
        self.drop_class = set()

    def shuffle_list(self):
        for c_iter in range(self.class_num):
            np.random.shuffle(self.s_list[c_iter])

    def set_length(self, length: int):
        self.data_len = length

    def set_probmatrix(self, prob_mat):
        self.prob_mat = prob_mat

    def get_list(self):
        self.random_loader = False
        winList = np.argmax(self.prob_mat, axis=1)
        for c_iter in range(self.class_num):
            cur_data = np.where(winList == c_iter)[0]
            self.s_list.append(cur_data)
            cur_length = math.floor((len(cur_data) * self.class_num) / self.batch_num)
            if (cur_length < 1):
                self.drop_class.add(c_iter)
                continue
            if (cur_length < self.data_len):
                self.set_length(cur_length)
                self.i_range = len(cur_data)
        if(len(self.drop_class) > 0):
            cur_length = math.floor((self.i_range * (self.class_num-len(self.drop_class))) / self.batch_num)
            self.set_length(cur_length)
        return True

    def __iter__(self):
        batch = []
        bs = self.batch_num
        if(self.random_loader):
            while(True):
                np.random.shuffle(self.all_data)
                for idx in range(self.i_range):
                    for b_iter in range(bs):
                        batch.append(self.all_data[idx*bs+b_iter])
                    yield batch
                    batch = []
        else:
            batch_ctr = 0
            cur_ctr = 0
            pick_item = np.arange(self.class_num)
            while(True):
                new_round = False
                for idx in range(self.i_range):
                    if(new_round):
                        break
                    np.random.shuffle(pick_item)
                    for c_iter in range(self.class_num):
                        if(new_round):
                            break
                        c_iter_l = pick_item[c_iter]
                        if c_iter_l in self.drop_class:
                            continue
                        c_idx = idx % len(self.s_list[c_iter_l])
                        batch.append(self.s_list[c_iter_l][c_idx])
                        cur_ctr += 1
                        if(cur_ctr % bs == 0):
                            yield batch
                            batch = []
                            cur_ctr = 0
                            batch_ctr += 1
                            if(batch_ctr == self.data_len):
                                batch_ctr = 0
                                self.shuffle_list()
                                new_round = True

    def __len__(self):
        return self.data_len

    def get_range(self):
        return self.i_range

def image_classification_test(loader, model, test_10crop=True):
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader['test'][i]) for i in range(10)]
            for i in range(len(loader['test'][0])):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                for j in range(10):
                    _, predict_out = model(inputs[j])
                    outputs.append(nn.Softmax(dim=1)(predict_out))
                outputs = sum(outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        else:
            iter_test = iter(loader["test"])
            for i in range(len(loader['test'])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                labels = labels
                _, outputs = model(inputs)
                outputs = nn.Softmax(dim=1)(outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy, all_output


def train(config):
    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(**config["prep"]['params'])
    else:
        prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]
    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
                                transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, \
            shuffle=True, num_workers=16, drop_last=True)

    dsets["target"] = MyDataset(open(data_config["target"]["list_path"]).readlines(), transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
            shuffle=True, num_workers=16, drop_last=True)

    s_gt = np.array(dsets["source"].imgs)[:, 1]
    t_gt = np.array(dsets["target"].imgs)[:, 1]

    class_num = config["network"]["params"]["class_num"]

    data_batch_source = data_batch(s_gt, batch_size=train_bs, drop_last=False, gt_flag=True, num_class=class_num, num_batch=train_bs)
    data_batch_target = data_batch(t_gt, batch_size=train_bs, drop_last=False, gt_flag=False, num_class=class_num, num_batch=train_bs)

    dataloader_source = torch.utils.data.DataLoader(
        dataset=dsets["source"],
        batch_sampler=data_batch_source,
        shuffle=False,
        num_workers=16,
        drop_last=False)

    dataloader_target = torch.utils.data.DataLoader(
        dataset=dsets["target"],
        batch_sampler=data_batch_target,
        shuffle=False,
        num_workers=16,
        drop_last=False)

    if prep_config["test_10crop"]:
        for i in range(10):
            dsets["test"] = [ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                transform=prep_dict["test"][i]) for i in range(10)]
            dset_loaders["test"] = [DataLoader(dset, batch_size=test_bs, \
                                shuffle=False, num_workers=16) for dset in dsets['test']]
    else:
        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                transform=prep_dict["test"])
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                shuffle=False, num_workers=16)

    class_num = config["network"]["params"]["class_num"]

    ## set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()
    ad_net = network.AdversarialNetwork(base_network.output_num(), 1024)

    ad_net = ad_net.cuda()
    parameter_list = base_network.get_parameters() + ad_net.get_parameters()
 
    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, \
                    **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        ad_net = nn.DataParallel(ad_net, device_ids=[int(i) for i in gpus])
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])

    cur_up_rate = 1
    all_up_rate = 0
    for l_iter in range (args.run_id, 0, -1):
        print(l_iter)
        loadFile = open(config["load_stem"] + str(l_iter-1) + ".npy", 'rb')
        if(l_iter == args.run_id):
            tar_pseu_load = np.load(loadFile)
        else:
            tar_pseu_load += (cur_up_rate * np.load(loadFile))
        loadFile.close()
        all_up_rate += cur_up_rate
        cur_up_rate *= (1-args.update)

    if(args.run_id != 0):
        tar_pseu_load = tar_pseu_load / all_up_rate
        data_batch_target.set_probmatrix(tar_pseu_load)
        data_batch_target.get_list()

    ## train
    conf_list = torch.ones((len(dset_loaders["target"].dataset), class_num))
    conf_list *= -1
    conf_list = conf_list.cuda()

    best_acc = 0.0
    targ_iter = 0

    Cs_memory = torch.zeros(class_num, 256).cuda()
    Ct_memory = torch.zeros(class_num, 256).cuda()
    if (args.run_id != 0):
        tar_pseu_prev = torch.from_numpy(tar_pseu_load).clone()
        tar_pseu_prev = tar_pseu_prev.cuda()

    iter_source = iter(dataloader_source)
    iter_target = iter(dataloader_target)

    for i in range(config["num_iterations"]):
        if i % config["test_interval"] == config["test_interval"] - 1:
            base_network.train(False)
            temp_acc, tar_pseu_save = image_classification_test(dset_loaders, \
                base_network, test_10crop=prep_config["test_10crop"])
            tar_pseu_save = tar_pseu_save.data.cpu().numpy()
            temp_model = nn.Sequential(base_network)
            saveFile = open(config["save_labels"], 'wb')
            np.save(saveFile, tar_pseu_save)
            saveFile.close()
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = temp_model
            log_str = "iter: {:05d}, precision: {:.5f}".format(i, temp_acc)
            config["out_file"].write(log_str+"\n")
            config["out_file"].flush()
            print(log_str)
        if i % config["snapshot_interval"] == 0:
            torch.save(nn.Sequential(base_network), osp.join(config["output_path"], str(args.run_id) + "_" + str(i)+ "_model.pth.tar"))

        loss_params = config["loss"]
        targ_iter+=1

        ## train one iter
        base_network.train(True)
        ad_net.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()
        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target, sample_idx = iter_target.next()
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()

        base_network.del_gradient()
        base_network.zero_grad()
        features_source, outputs_source = base_network.bp(inputs_source)
        features_target, outputs_target = base_network.bp(inputs_target)
        features = torch.cat((features_source, features_target), dim=0)
        outputs = torch.cat((outputs_source, outputs_target), dim=0)
        outputs_target_sm = nn.Softmax(dim=1)(outputs_target)

        ### Calculate aggregate label ###
        for id_iter in range(len(sample_idx)):
            if conf_list[sample_idx[id_iter],0] < -0.5:
                conf_list[sample_idx[id_iter]] = outputs_target_sm[id_iter]
            else:
                conf_list[sample_idx[id_iter]] = args.agg_up * outputs_target_sm[id_iter] + (1-args.agg_up) * conf_list[sample_idx[id_iter]]
            outputs_target_sm[id_iter] = conf_list[sample_idx[id_iter]]
        ### Add the probabilities from previous run ###
        if(args.run_id != 0):
            pm_weight = calc_pm(i, args.pm_meth)
            pm_weight = min(pm_weight, 1.0)
            pm_weight = max(pm_weight, 0)
            outputs_target_sm = (1 - pm_weight) * tar_pseu_prev[sample_idx] + pm_weight * outputs_target_sm
        c_, pseu_labels_target = torch.max(outputs_target_sm, 1)

        t_loss = nn.CrossEntropyLoss()(outputs_target, pseu_labels_target)
        t_loss.backward(retain_graph=True)
        batch_size = outputs_target.shape[0]
        mask_target = torch.ones(batch_size, base_network.output_num())
        mask_target = mask_target.cuda()
        mask_target = mask_target * base_network.gradients[0]
        for b_cur in range(batch_size):
            mask_target[b_cur] = mask_target[b_cur] / LA.norm(mask_target[b_cur]) * args.factor
        mask_target = mask_target.detach()

        base_network.del_gradient()
        base_network.zero_grad()

        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        classifier_loss.backward(retain_graph=True)
        batch_size = outputs_source.shape[0]
        mask_source = torch.ones(batch_size, base_network.output_num())
        mask_source = mask_source.cuda()
        mask_source = mask_source * base_network.gradients[0]
        for b_cur in range(batch_size):
            mask_source[b_cur] = mask_source[b_cur] / LA.norm(mask_source[b_cur]) * args.factor

        mask_source = mask_source.detach()
        mask = torch.cat((mask_source, mask_target), dim=0)
        mask = torch.mul(features, mask)

        transfer_loss = loss.DANN(mask, ad_net)
        lam = network.calc_coeff(i)
        loss_sm, loss_sm_ss, Cs_memory, Ct_memory = loss.SM(features_source, features_target, labels_source, pseu_labels_target, Cs_memory, Ct_memory)
        transfer_loss = transfer_loss + lam * loss_sm + loss_sm_ss

        total_loss = loss_params["trade_off"] * transfer_loss
        total_loss.backward()
        optimizer.step()

        Cs_memory.detach_()
        Ct_memory.detach_()
    torch.save(best_model, osp.join(config["output_path"], "best_model.pth.tar"))
    return best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BIWAA')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50', choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13", "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"])
    parser.add_argument('--dset', type=str, default='office', choices=['office', 'domain-net', 'visda', 'office-home'], help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default='../../data/office/amazon_31_list.txt', help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='../../data/office/webcam_10_list.txt', help="The target dataset path list")
    parser.add_argument('--num_iter', type=int, default=5004, help="Iterations for a single run")
    parser.add_argument('--test_interval', type=int, default=5000, help="Interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=5000, help="Interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='san', help="Output directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--factor', type=float, default=1.0, help="Normalization factor")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--update', type=float, default=1, help="Update factor for aggregative labels for consecutive runs")
    parser.add_argument('--agg_up', type=float, default=1, help="Update factor for aggregative labels within a run")
    parser.add_argument('--run_id', type=int, default=0, help="Run ID for iterative label distribution alignment")
    parser.add_argument('--pm_meth', type=int, default=500, help="Parameter for how long bootstrap will be used")


    args = parser.parse_args()
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

    # Set random number seed.
    np.random.seed(args.seed + 100 * args.run_id)
    torch.manual_seed(args.seed + 100 * args.run_id)

    # train config
    config = {}
    config["gpu"] = args.gpu_id
    config["num_iterations"] = args.num_iter
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    config["output_path"] = "snapshot/" + args.output_dir
    if not osp.exists(config["output_path"]):
        os.system('mkdir -p '+config["output_path"])

    config["out_file"] = open(osp.join(config["output_path"], "log-" + str(args.run_id) + ".txt"), "w")
    config["save_labels"] = osp.join(config["output_path"], "logits-" + str(args.run_id) + ".npy")
    config["load_stem"] = osp.join(config["output_path"], "logits-")
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])

    config["prep"] = {"test_10crop":True, 'params':{"resize_size":256, "crop_size":224, 'alexnet':False}}
    config["loss"] = {"trade_off":1.0}
    if "AlexNet" in args.net:
        config["prep"]['params']['alexnet'] = True
        config["prep"]['params']['crop_size'] = 227
        config["network"] = {"name":network.AlexNetFc, \
            "params":{"use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    elif "ResNet" in args.net:
        config["network"] = {"name":network.ResNetFc, \
            "params":{"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    elif "VGG" in args.net:
        config["network"] = {"name":network.VGGFc, \
            "params":{"vgg_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    config["loss"]["random_dim"] = 1024

    config["optimizer"] = {"type":optim.SGD, "optim_params":{'lr':args.lr, "momentum":0.9, \
                           "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", \
                           "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75} }

    config["dataset"] = args.dset
    config["data"] = {"source":{"list_path":args.s_dset_path, "batch_size":36}, \
                      "target":{"list_path":args.t_dset_path, "batch_size":36}, \
                      "test":{"list_path":args.t_dset_path, "batch_size":36}}

    if config["dataset"] == "office":
        if ("amazon" in args.s_dset_path and "webcam" in args.t_dset_path) or \
           ("webcam" in args.s_dset_path and "dslr" in args.t_dset_path) or \
           ("webcam" in args.s_dset_path and "amazon" in args.t_dset_path) or \
           ("dslr" in args.s_dset_path and "amazon" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        elif ("amazon" in args.s_dset_path and "dslr" in args.t_dset_path) or \
             ("dslr" in args.s_dset_path and "webcam" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.0003 # optimal parameters       
        config["network"]["params"]["class_num"] = 31 
    elif config["dataset"] == "domain-net":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 40
    elif config["dataset"] == "office-home":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 65
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')
    config["out_file"].write(str(config)+"\n")
    config["out_file"].flush()
    train(config)
