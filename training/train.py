import os
os.environ["NCCL_P2P_DISABLE"] = "1"
import random
from typing import DefaultDict
import warnings
warnings.filterwarnings("ignore")
import gc
import numpy as np
import argparse
import time
import sys
import pickle
import shutil
import importlib
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#from waymo_data.collate_func import *
import io
from metricss_soft_map import soft_map
import scipy.special
import scipy.interpolate as interp
from waymo_dataset import *

parser = argparse.ArgumentParser('Interface for HDGT Training')
##### Optimizer - Scheduler
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--val_batch_size', type=int, default=128, help='batch size')
parser.add_argument('--n_epoch', type=int, default=30, help='number of epochs')
parser.add_argument('--warmup', type=float, default=1.0, help='the number of epoch for warmup')
parser.add_argument('--lr_decay_epoch', type=str, default="4-8-16-24-26", help='the index of epoch where the lr decays to lr*0.5')
parser.add_argument('--num_prediction', type=int,default=6, help='the number of modality')
parser.add_argument('--cls_weight', type=float,default=0.1, help='the weight of classification loss')
parser.add_argument('--reg_weight', type=float,default=50.0, help='the weight of regression loss')

#### Speed Up
parser.add_argument('--num_of_gnn_layer', type=int, default=6, help='the number of HDGT layer')
parser.add_argument('--hidden_dim', type=int, default=256, help='init hidden dimension')
parser.add_argument('--head_dim', type=int, default=32, help='the dimension of attention head')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout probability')
parser.add_argument('--num_worker', type=int, default=8, help='number of worker per dataloader')

#### Setting
parser.add_argument('--agent_drop', type=float, default='0.0', help='the ratio of randomly dropping agent')
parser.add_argument('--data_folder', type=str,default="hdgt_waymo", help='training set')

parser.add_argument('--refine_num', type=int, default=5, help='temporally refine the trajectory')
parser.add_argument('--output_vel', type=str, default="True", help='output in form of velocity') 
parser.add_argument('--cumsum_vel', type=str, default="True", help='cumulate velocity for reg loss')


#### Initialize
parser.add_argument('--checkpoint', type=str, default="none", help='load checkpoint')
parser.add_argument('--start_epoch', type=int, default=1, help='the index of start epoch (for resume training)')
parser.add_argument('--dev_mode', type=str, default="False", help='develop_mode')

parser.add_argument('--ddp_mode', type=str, default="False", help='False, True, multi_node')
parser.add_argument('--port', type=str, default="31243", help='DDP')

parser.add_argument('--amp', type=str, default="none", help='type of fp16')

#### Log
parser.add_argument('--val_every_train_step', type=int, default=-1, help='every number of training step to conduct one evaluation')
parser.add_argument('--name', type=str, default="hdgt_waymo_dev", help='the name of this setting')
args = parser.parse_args()
os.environ["DGLBACKEND"] = "pytorch"


class Logger():
    def __init__(self, lognames):
        self.terminal = sys.stdout
        self.logs = []
        for log_name in lognames:
            self.logs.append(open(log_name, 'w'))
    def write(self, message):
        self.terminal.write(message)
        for log in self.logs:
            log.write(message)
            log.flush()
    def flush(self):
        pass

def euclid(label, pred):
    return torch.sqrt((label[...,0]-pred[...,0])**2 + (label[...,1]-pred[...,1])**2)

def euclid_np(label, pred):
    return np.sqrt((label[...,0]-pred[...,0])**2 + (label[...,1]-pred[...,1])**2)

def cal_ADE(label, pred):
    return euclid_np(label,pred).mean()

def cal_FDE(label, pred):
    return euclid_np(label[:,-1,:], pred[:,-1,:]).mean()

def cal_ade_fde_mr(labels, preds, masks):
    if labels.shape[0] == 0:
        return None, None
    l2_norm = euclid_np(labels, preds)
    
    masks_sum = masks.sum(1)
    ade_indices = masks_sum != 0
    ade_cnt = ade_indices.sum()
    ade = ((l2_norm[ade_indices] * masks[ade_indices]).sum(1)/masks_sum[ade_indices]).mean()

    fde_indices = masks[:, -1] != 0
    fde_cnt = fde_indices.sum()
    fde = 0.0
    mr = 0.0
    if fde_cnt != 0:
        fde = l2_norm[fde_indices, -1]
        mr = (fde > 2.0).mean()
        fde = fde.mean()
    return [ade, fde, mr], [ade_cnt, fde_cnt, fde_cnt]

def cal_min6_ade_fde_mr(preds, labels, masks):
    if labels.shape[0] == 0:
        return None, None
    l2_norm = euclid_np(labels[:, np.newaxis, :, :], preds)
    ## ade6
    masks_sum = masks.sum(1)
    ade_indices = masks_sum != 0
    ade_cnt = ade_indices.sum()
    ade6 = ((l2_norm[ade_indices] * masks[ade_indices, np.newaxis, :]).sum(-1)/masks_sum[ade_indices][:, np.newaxis]).min(-1).mean()
    
    fde_indices = masks[:, -1] != 0
    fde_cnt = fde_indices.sum()
    fde6 = 0.0
    mr6 = 0.0
    if fde_cnt != 0:
        fde6 = l2_norm[fde_indices, :, -1].min(-1)
        mr6 = (fde6 > 2.0).mean()
        fde6 = fde6.mean()
    return [ade6, fde6, mr6], [ade_cnt, fde_cnt, fde_cnt]

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


def main():
    args = parser.parse_args()
    ###Distributed
    gpu_count = torch.cuda.device_count()
    global_seed = int(args.port) ## Import!!!! for coherent data splitting across process
    if gpu_count > 1:
        if args.ddp_mode == "multi_node":
            main_worker(int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"]), global_seed, args)
        else:
            mp.spawn(main_worker, nprocs=gpu_count, args=(gpu_count, global_seed, args))
    else:
        main_worker(0, gpu_count, global_seed, args)

## Running for each GPU
def main_worker(gpu, gpu_count, global_seed, args):
    if args.ddp_mode == "multi_node":
        global_rank = int(os.environ["RANK"])
        init_port = "tcp://"+os.environ["MASTER_ADDR"]+":"+os.environ["MASTER_PORT"] 
    else:
        global_rank = gpu
        init_port = "tcp://127.0.0.1:"+args.port
    print(f"Use GPU: {gpu} for training. Global Rank:{global_rank} Global World Size:{gpu_count} Init Port {init_port}")
    print("Process Id:", os.getpid())
    seed_num = random.randint(0, 1000000)
    torch.manual_seed(seed_num+global_rank)
    random.seed(seed_num+global_rank)
    np.random.seed(seed_num+global_rank)

    if gpu_count > 1:
        dist.init_process_group(backend="nccl", world_size=gpu_count, init_method=init_port, rank=global_rank)
    torch.cuda.set_device(gpu)
    device = torch.device("cuda:"+str(gpu))

    
    snapshot_dir = None
    if global_rank == 0:
        setting_name = args.name
        log_dir = "logs/" + str(setting_name+"_"+time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())))
        
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        sys.stdout = Logger([f"{setting_name}.log", os.path.join(log_dir, f"{setting_name}.log")])
        snapshot_dir = os.path.join(log_dir, "snapshot")
        if not os.path.isdir(snapshot_dir):
            os.makedirs(snapshot_dir)
        print("Log Directory:", os.path.join(log_dir, sys.argv[0]))
        print(args)
        shutil.copyfile(__file__, os.path.join(log_dir, "train.py"))
        shutil.copyfile("model.py", os.path.join(log_dir, "model.py"))
    model_module = importlib.import_module("model")
    model = model_module.HDGT_model(input_dim=11, args=args)
    model.apply(model_module.weights_init)

    print("Start Load Dataset")
    train_dataloader, val_dataloader, train_sample_num, val_sample_num = obtain_dataset(global_rank, gpu_count, global_seed, args)

    checkpoint = None
    if args.checkpoint != "none":
        print("Load:", args.checkpoint, gpu)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model = model.to(device, non_blocking=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)    
    step_per_epoch = train_sample_num // args.batch_size // gpu_count + 1
    epoch = args.n_epoch

    warmup = args.warmup
    if args.start_epoch > 1:
        warmup = 0.0

    scheduler = model_module.WarmupLinearSchedule(optimizer, step_per_epoch*warmup, step_per_epoch*epoch)
    if args.checkpoint != "none":
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr
            param_group["betas"] = (0.9, 0.95)
        print("lr")
    
    if gpu_count > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], output_device=gpu, find_unused_parameters=True)
    reg_criterion = torch.nn.SmoothL1Loss(reduction="none").to(device)
    if gpu == 0:
        print("start")
    
    if args.amp == "fp16":
        amp_data_type = torch.float16
        scaler = torch.cuda.amp.GradScaler()
        print("Use AMP with Type:", amp_data_type)
    elif args.amp == "bf16":
        amp_data_type = torch.bfloat16
        scaler = torch.cuda.amp.GradScaler()
        print("Use AMP with Type:", amp_data_type)
    else:
        amp_data_type = torch.float32 ## Not enabled
        scaler = None

    for epoch in range(args.start_epoch, args.n_epoch+1):
        run_model(dataloader=train_dataloader, num_sample=train_sample_num, model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, gpu=gpu, global_rank=global_rank, gpu_count=gpu_count, is_train=True, args=args, val_dataloader=val_dataloader, val_sample_num=val_sample_num, snapshot_dir=snapshot_dir, scaler=scaler, amp_data_type=amp_data_type)
    

def run_model(dataloader, num_sample, model, optimizer, scheduler, epoch, gpu, global_rank, gpu_count, is_train, args, val_dataloader=None, val_sample_num=None, snapshot_dir=None, scaler=None, amp_data_type=None):
    length_lis = [30, 50, 80]
    agent_type_lis = ["VEHICLE", "PEDESTRIAN", "CYCLIST"]
    metric_type_lis = ["ade", "fde", "mr", "ade6", "fde6", "mr6",]
    
    acc_metric_type_lis = ["accs", "acc6s"]
    recorder = {}
    for agent_type in agent_type_lis:
        for acc_metric_type in acc_metric_type_lis:
            recorder[agent_type+"_"+acc_metric_type] = AverageMeter()
        for length in length_lis:
            for metric_type in metric_type_lis:
                recorder[agent_type+"_"+str(length)+"_"+metric_type] = AverageMeter()
    recorder["loss"] = AverageMeter()
    recorder["reg_loss"] = AverageMeter()
    recorder["cls_loss"] = AverageMeter()
    losses_recoder, reg_losses_, cls_losses = AverageMeter(), AverageMeter(), AverageMeter()

    start_time = time.time()
    if is_train:
        model.train()
        batch_size = args.batch_size
    else:
        model.eval()
        batch_size = args.val_batch_size
        gpu_count = 1
    
    is_dev = (args.dev_mode == "True")
    is_output_vel = (args.output_vel == "True")
    is_cumsum_vel = (args.cumsum_vel == "True")


    if is_dev:
        val_every_train_step = 1
        print_freq = 1
    else:
        val_every_train_step = args.val_every_train_step
        print_freq = 100
    
    num_prediction = args.num_prediction
    refine_num = args.refine_num
    cls_weight = args.cls_weight
    reg_weight = args.reg_weight
    device = torch.device("cuda:"+str(gpu))
    
    reg_criteria = torch.nn.SmoothL1Loss(reduction="none").to(device)
    
    lr_decay_epoch = args.lr_decay_epoch.split("-")
    lr_decay_epoch = [int(_) for _ in lr_decay_epoch]

    decay_coefficient = 1.0
    for decay_epoch in lr_decay_epoch:
        if epoch >= decay_epoch:
            decay_coefficient = decay_coefficient * 0.5
    scheduler.base_lrs = [args.lr * decay_coefficient]
    for param_group in optimizer.param_groups:
        param_group["lr"] = args.lr * decay_coefficient

    use_amp = (scaler == None)
    with torch.set_grad_enabled(is_train):
        for batch_index, data in enumerate(dataloader, 0):
            data["is_train"] = is_train
            data["gpu"] = gpu
            for tensor_name in data["cuda_tensor_lis"]:
                data[tensor_name] = data[tensor_name].to("cuda:"+str(gpu), non_blocking=True)
            optimizer.zero_grad()
            num_of_sample = len(data["pred_num_lis"])

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_data_type):
                agent_reg_res, agent_cls_res, pred_indice_bool_type_lis = model(data)
                reg_labels = [data["label_lis"][pred_indice_bool_type_lis[_]] for _ in range(3)]

                auxiliary_labels = [data["auxiliary_label_lis"][pred_indice_bool_type_lis[_]] for _ in range(3)]
                auxiliary_labels_future = [data["auxiliary_label_future_lis"][pred_indice_bool_type_lis[_]] for _ in range(3)]
                label_masks = [data["label_mask_lis"][pred_indice_bool_type_lis[_]] for _ in range(3)]
                
                agent_closest_index_lis = [[] for _ in range(3)]
                loss = 0.0
                reg_loss = 0.0
                cls_loss = 0.0
                total_num_of_mask = 0.0
                total_num_of_agent = 0.0
                for agent_type_index in range(3):
                    if agent_reg_res[agent_type_index].shape[0] == 0:
                        continue
                    num_of_mask_per_agent = label_masks[agent_type_index].sum(dim=-1)
                    mask_sum = num_of_mask_per_agent.sum()
                    if mask_sum != 0:
                        dist_between_pred_label =  reg_criteria(agent_reg_res[agent_type_index], reg_labels[agent_type_index].unsqueeze(1).unsqueeze(1).repeat(1, refine_num+1, num_prediction, 1, 1)).mean(-1) ## N_Agent, N_refine, num_prediction, 80
                        dist_between_pred_label = (dist_between_pred_label * label_masks[agent_type_index].unsqueeze(1).unsqueeze(1)).sum(-1) / (num_of_mask_per_agent.unsqueeze(-1).unsqueeze(-1)+1) ## N_Agent, N_refine, num_prediction
                        agent_closest_index = dist_between_pred_label[:, -1, :].argmin(dim=-1)
                        
                        reg_loss += (dist_between_pred_label[torch.arange(agent_closest_index.shape[0]), :, agent_closest_index]).sum() / (refine_num+1)
                        
                        log_pis = agent_cls_res[agent_type_index]
                        log_pis = log_pis - torch.logsumexp(log_pis, dim=-1, keepdim=True)
                        log_pi = log_pis[torch.arange(agent_closest_index.shape[0]), agent_closest_index].sum()
                        cls_loss +=   (-log_pi)
                        agent_closest_index_lis[agent_type_index] = (agent_closest_index)
                        total_num_of_agent += len(agent_closest_index)
                        total_num_of_mask += mask_sum

                loss += (cls_loss / total_num_of_agent * cls_weight + reg_loss / total_num_of_mask * reg_weight)
                reg_loss_cnt = total_num_of_mask
                cls_loss_cnt = total_num_of_agent

            if is_train:
                if loss != 0:
                    if torch.isnan(loss):
                        print("Bad Gradients!", epoch, batch_index)
                        optimizer.zero_grad()
                        del data
                        del agent_reg_res, agent_cls_res, pred_indice_bool_type_lis
                        del loss
                        continue
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                        optimizer.step()
                    scheduler.step()
            
            if global_rank == 0:
                with torch.no_grad():
                    if loss != 0:
                        recorder["loss"].update(loss.item(), num_of_sample)
                        recorder["cls_loss"].update(cls_loss.item()/cls_loss_cnt * cls_weight, cls_loss_cnt)
                        recorder["reg_loss"].update(reg_loss.item()/reg_loss_cnt * reg_weight, reg_loss_cnt)
                        
                        neighbor_size_lis = data["pred_num_lis"]
                        cumsum_neighbor_size_lis = np.cumsum(neighbor_size_lis, axis=0).tolist()
                        cumsum_neighbor_size_lis = [0] + cumsum_neighbor_size_lis
                        for agent_type_index in range(3):
                            now_agent_cls_res = agent_cls_res[agent_type_index]
                            if now_agent_cls_res.shape[0] == 0:
                                continue
                            
                            now_agent_reg_res = agent_reg_res[agent_type_index][:, -1, ...].detach().cpu().numpy()
                            now_labels = reg_labels[agent_type_index].detach().cpu().numpy()
                            now_auxiliary_labels = auxiliary_labels[agent_type_index].detach().cpu().numpy()
                            now_auxiliary_labels_future = auxiliary_labels_future[agent_type_index].detach().cpu().numpy()
                            now_label_masks = label_masks[agent_type_index].detach().cpu().numpy()
                            now_agent_closest_index = agent_closest_index_lis[agent_type_index].detach().cpu().numpy()
                            now_cls_sorted_index = now_agent_cls_res.argsort(dim=-1, descending=True).detach().cpu().numpy()
                            now_agent_cls_res = now_agent_cls_res.detach().cpu().numpy()

                            cls_acc = 0.0
                            cls_acc6 = 0.0
                            best_preds = [0] * len(now_labels)
                            best_6preds = [0] * len(now_labels)
                            for item_index in range(len(now_labels)):
                                if now_agent_closest_index[item_index] == now_cls_sorted_index[item_index][0]:
                                    cls_acc += 1.0
                                if now_agent_closest_index[item_index] in now_cls_sorted_index[item_index][:6].tolist():
                                    cls_acc6 += 1.0
                                best_preds[item_index] = now_agent_reg_res[item_index, ...][now_cls_sorted_index[item_index][0], :, :]
                                best_6preds[item_index] = now_agent_reg_res[item_index][now_cls_sorted_index[item_index][:6], :, :]
                            cls_acc /= now_agent_reg_res.shape[0]
                            cls_acc6 /= now_agent_reg_res.shape[0]
                            recorder[agent_type_lis[agent_type_index]+"_"+"accs"].update(cls_acc, now_agent_reg_res.shape[0])
                            recorder[agent_type_lis[agent_type_index]+"_"+"acc6s"].update(cls_acc6, now_agent_reg_res.shape[0])
                            best_preds = np.stack(best_preds, axis=0)
                            best_6preds = np.stack(best_6preds, axis=0)

                            for length_indices in range(3):
                                res_lis, res_cnt_lis = cal_ade_fde_mr(best_preds[:, :length_lis[length_indices], :][:, 4::5, :], now_labels[:, :length_lis[length_indices], :][:, 4::5, :], now_label_masks[:, :length_lis[length_indices]][:, 4::5])
                                if res_lis:
                                    for metric_indices, metric_type in enumerate(["ade", "fde", "mr"]):
                                        if res_cnt_lis[metric_indices] > 0:
                                            recorder[agent_type_lis[agent_type_index]+"_"+str(length_lis[length_indices])+"_"+metric_type].update(res_lis[metric_indices], res_cnt_lis[metric_indices])
                                
                                res_lis, res_cnt_lis = cal_min6_ade_fde_mr(best_6preds[:, :, :length_lis[length_indices], :][:, :, 4::5, :], now_labels[:, :length_lis[length_indices], :][:, 4::5, :], now_label_masks[:, :length_lis[length_indices]][:, 4::5])
                                if res_lis:
                                    for metric_indices, metric_type in enumerate(["ade6", "fde6", "mr6"]):
                                        if res_cnt_lis[metric_indices] > 0:
                                            recorder[agent_type_lis[agent_type_index]+"_"+str(length_lis[length_indices])+"_"+metric_type].update(res_lis[metric_indices], res_cnt_lis[metric_indices])
                if (is_train and ((batch_index+1) % print_freq) == 0):
                    print_dic = {metric_type:0.0 for metric_type in metric_type_lis}
                    sub_print_dic = {}
                    for agent_type in agent_type_lis:
                        for metric_type in metric_type_lis:
                            sub_print_dic[agent_type + "_" + metric_type] = 0
                            for length in length_lis:
                                sub_print_dic[agent_type + "_" + metric_type] += recorder[agent_type+"_"+str(length)+"_"+metric_type].avg 

                    detail_text = ""
                    for agent_type in agent_type_lis:
                        for length in length_lis:
                            for metric_type in metric_type_lis:
                                print_dic[metric_type] += recorder[agent_type+"_"+str(length)+"_"+metric_type].avg        
                                detail_text +=  ", "+agent_type+"_"+str(length)+"_" + metric_type + " {:.4f}".format(recorder[agent_type+"_"+str(length)+"_"+metric_type].avg)
                    for agent_type in agent_type_lis:
                        for acc_metric_type in acc_metric_type_lis:
                            detail_text += ", "+agent_type+"_" + acc_metric_type + " "+str(recorder[agent_type+"_"+acc_metric_type].avg)
                    print_dic = {k:v/9.0 for k, v in print_dic.items()}
                    sub_print_dic = {k:v/3.0 for k, v in sub_print_dic.items()}

                    print_text = ' Epoch: [{0}][{1}/{2}-Batch {3}], '.format(epoch, (batch_index+1)*batch_size*gpu_count, num_sample, batch_index)
                    print_text += "Loss {:.8f}, ".format(recorder["loss"].avg)
                    print_text += "Cls Loss {:.8f}, ".format(recorder["cls_loss"].avg)
                    print_text += "Reg Loss {:.8f}, ".format(recorder["reg_loss"].avg)
                    for k, v in print_dic.items():
                        print_text += k + " {:.4f}, ".format(v)
                    for k, v in sub_print_dic.items():
                        print_text += k + " {:.4f}, ".format(v)
                    print_text += "Time(s): {:.4f}".format(time.time()-start_time)
                    print_text += detail_text

                    print_text += ", LR: {:.4e}".format(scheduler.get_last_lr()[0])
                    print(print_text, flush=True)

                    ## Reinit Train Recorder
                    recorder = {}
                    for agent_type in agent_type_lis:
                        for acc_metric_type in acc_metric_type_lis:
                            recorder[agent_type+"_"+acc_metric_type] = AverageMeter()
                        for length in length_lis:
                            for metric_type in metric_type_lis:
                                recorder[agent_type+"_"+str(length)+"_"+metric_type] = AverageMeter()
                    recorder["loss"] = AverageMeter()
                    recorder["reg_loss"] = AverageMeter()
                    recorder["cls_loss"] = AverageMeter()
                                        
                    if is_train and ((batch_index+1) % val_every_train_step == 0 or (val_every_train_step <= 0 and batch_index == (len(dataloader)-1))) and not is_dev:                    
                        if gpu_count > 1:
                            val_model = model.module
                        else:
                            val_model = model
                        val_model.eval()
                        run_model(val_dataloader, val_sample_num, val_model, optimizer, scheduler, epoch, gpu, global_rank, gpu_count, is_train=False, args=args, scaler=scaler, amp_data_type=amp_data_type)
                        model.train()
                        file_path = os.path.join(snapshot_dir, "Epoch_"+str(epoch)+"_batch"+str(batch_index)+".pt")
                        checkpoint = {}

                        if gpu_count > 1:
                            checkpoint["model_state_dict"] = model.module.state_dict()
                        else:
                            checkpoint["model_state_dict"] = model.state_dict()

                        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
                        torch.save(checkpoint, file_path)
                        print("Epoch %d Batch %d Save Model"%(epoch, batch_index))
            del data
    if (not is_train):
        print_dic = {metric_type:0.0 for metric_type in metric_type_lis}
        sub_print_dic = {}
        for agent_type in agent_type_lis:
            for metric_type in metric_type_lis:
                sub_print_dic[agent_type + "_" + metric_type] = 0
                for length in length_lis:
                    sub_print_dic[agent_type + "_" + metric_type] += recorder[agent_type+"_"+str(length)+"_"+metric_type].avg 

        detail_text = ""
        for agent_type in agent_type_lis:
            for length in length_lis:
                for metric_type in metric_type_lis:
                    print_dic[metric_type] += recorder[agent_type+"_"+str(length)+"_"+metric_type].avg        
                    detail_text +=  ", "+agent_type+"_"+str(length)+"_" + metric_type + " {:.4f}".format(recorder[agent_type+"_"+str(length)+"_"+metric_type].avg)
        for agent_type in agent_type_lis:
            for acc_metric_type in acc_metric_type_lis:
                detail_text += ", "+agent_type+"_" + acc_metric_type + " "+str(recorder[agent_type+"_"+acc_metric_type].avg)
        print_dic = {k:v/9.0 for k, v in print_dic.items()}
        sub_print_dic = {k:v/3.0 for k, v in sub_print_dic.items()}

        print_text = '****Val Epoch: [{0}][{1}/{2}], '.format(epoch, (batch_index+1)*batch_size*gpu_count, num_sample)
        print_text += "Loss {:.8f}, ".format(recorder["loss"].avg)
        print_text += "Cls Loss {:.8f}, ".format(recorder["cls_loss"].avg)
        print_text += "Reg Loss {:.8f}, ".format(recorder["reg_loss"].avg)
        for k, v in print_dic.items():
            print_text += k + " {:.4f}, ".format(v)
        for k, v in sub_print_dic.items():
            print_text += k + " {:.4f}, ".format(v)
        print_text += "Time(s): {:.4f}".format(time.time()-start_time)
        print_text += detail_text
        print(print_text, flush=True)

if __name__ == '__main__':
    main()
