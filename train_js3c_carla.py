import torch.optim as optim
from pathlib import Path
from utils import config
# cfg = get_parser()
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import importlib
import logging
import shutil
import spconv
import json
import yaml
import time
import torch
import os

from utils.evaluate_completion import get_eval_mask
from torch.utils.checkpoint import checkpoint
import models.model_utils as model_utils
from utils.np_ioueval import iouEval
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

args = config.cfg

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.float32  # Tensor type to be used
os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu']

LEARNING_RATE_CLIP = 1e-6
MOMENTUM_ORIGINAL = 0.5
MOMENTUM_DECCAY = 0.5
BN_MOMENTUM_MAX = 0.001
NUM_CLASS_SEG = args['DATA']['classes_seg']
NUM_CLASS_COMPLET = args['DATA']['classes_completion']

exp_name = args['log_dir']

seg_head = importlib.import_module('models.'+args['Segmentation']['model_name'])
seg_model = seg_head.get_model

complet_head = importlib.import_module('models.'+args['Completion']['model_name'])
complet_model = complet_head.get_model

print(args['DATA']['dataset'])
if args['DATA']['dataset'] == 'SemanticKITTI':
    dataset = importlib.import_module('kitti_dataset')

data_dir = '/media/sde1/Joey/Carla/Data/Scenes/Cartesian' # TODO: change the directory to the Carla dataset

coordinate_type = "cartesian"
cylindrical = coordinate_type=="cylindrical"
T = 1 #T is 1 because this is single sweep
B = args['TRAIN']['batch_size'] # Matching paper

model_name = 'JS3CNet' # TODO: used as the folder name

experiment_dir = "./Runs/" + model_name

# import the dataset
from dataset import CarlaDataset

carla_ds_train = CarlaDataset(directory=os.path.join(data_dir, 'Train'), device=device, num_frames=T, cylindrical=cylindrical, config=args)
carla_ds_val = CarlaDataset(directory=os.path.join(data_dir, 'Val'), device=device, num_frames=T, cylindrical=cylindrical, config=args, split='Val')

writer = SummaryWriter("./Runs/" + model_name)
with open(os.path.join(experiment_dir, 'args.txt'), 'w') as f:
    json.dump(args, f, indent=2)
file_handler = logging.FileHandler('%s/train.txt'%(experiment_dir))

logger = logging.getLogger("Model")
logger.setLevel(logging.INFO)
def log_string(str):
    logger.info(str)
    print(str)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# file_handler = logging.FileHandler('%s/train.txt'%(experiment_dir))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class J3SC_Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.seg_head = seg_model(args)
        self.complet_head = complet_model(args)
        self.voxelpool = model_utils.VoxelPooling(args)
        self.seg_sigma = nn.Parameter(torch.Tensor(1).uniform_(0.2, 1), requires_grad=True)
        self.complet_sigma = nn.Parameter(torch.Tensor(1).uniform_(0.2, 1), requires_grad=True)

    def forward(self, x):
        seg_inputs, complet_inputs, _ = x

        '''Segmentation Head'''
        seg_feature = seg_inputs['seg_features']
        # print(f'seg inputs is {seg_feature}')
        seg_output, feat = self.seg_head(seg_inputs)
        torch.cuda.empty_cache()

        '''Completion Head'''
        coords = complet_inputs['complet_coords']
        coords = coords[:, [0, 3, 2, 1]]
        

        # if args['DATA']['dataset'] == 'SemanticKITTI':
        #     coords[:, 3] += 1  # TODO SemanticKITTI will generate [256,256,31]
        # elif args['DATA']['dataset'] == 'SemanticPOSS':
        #     coords[:, 3][coords[:, 3] > 31] = 31

        # print(f'JS3CNet feat is {feat}')
        if args['Completion']['feeding'] == 'both':
            feeding = torch.cat([seg_output, feat],1)
        elif args['Completion']['feeding'] == 'feat':
            feeding = feat
        else:
            feeding = seg_output
        features = self.voxelpool(invoxel_xyz=complet_inputs['complet_invoxel_features'][:, :, :-1],
                                    invoxel_map=complet_inputs['complet_invoxel_features'][:, :, -1].long(),
                                    src_feat=feeding,
                                    voxel_center=complet_inputs['voxel_centers'])
        if self.args['Completion']['no_fuse_feat']:
            features[...] = 1
            features = features.detach()
        batch_complet = spconv.SparseConvTensor(features.float(), coords.int(), args['Completion']['full_scale'], args['TRAIN']['batch_size'])
        batch_complet = dataset.sparse_tensor_augmentation(batch_complet, complet_inputs['state'])

        if args['GENERAL']['debug']:
            model_utils.check_occupation(complet_inputs['complet_input'], batch_complet.dense())
        # print(batch_complet.dense()[0,:,:,:,:])
        complet_output = self.complet_head(batch_complet)
        torch.cuda.empty_cache()

        return seg_output, complet_output, [self.seg_sigma, self.complet_sigma]

def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum

classifier = J3SC_Net(args).cuda()
criteria = model_utils.Loss(args).cuda()

train_dataloader = DataLoader(carla_ds_train, num_workers=args['TRAIN']['train_workers'], batch_size=args['TRAIN']['batch_size'], shuffle=True, collate_fn=seg_head.Merge, pin_memory=True, drop_last=True, worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
seg_labelweights = torch.Tensor(carla_ds_train.seg_labelweights).cuda()
compl_labelweights = torch.Tensor(carla_ds_train.compl_labelweights).cuda()

val_dataloader = DataLoader(carla_ds_val, batch_size=args['TRAIN']['batch_size'], num_workers=args['TRAIN']['train_workers'], shuffle=False, collate_fn=seg_head.Merge, pin_memory=True, drop_last=True, worker_init_fn=lambda x: np.random.seed(x + int(time.time())))


training_epochs = args['TRAIN']['epochs']
# training_epoch = model_utils.checkpoint_restore(classifier, experiment_dir, True, train_from=args['TRAIN']['train_from'])
optimizer = optim.Adam(classifier.parameters(), lr=args['TRAIN']['learning_rate'], weight_decay=1e-4)

global_epoch = 0
best_iou_sem_complt = 0
best_iou_complt = 0
best_iou_seg = 0
kitti_config = yaml.safe_load(open('opt/carla.yaml', 'r')) # TODO: change this to carla_all.yaml if you want to use all labels
class_strings = kitti_config["labels"]

seg_label_to_cat = kitti_config["class_strings"]
class_inv_remap = kitti_config["learning_map_inv"]

valid_labels = np.zeros((20), dtype=np.int32)
learning_map_inv = kitti_config['learning_map_inv']
for key,value in learning_map_inv.items():
    valid_labels[key] = value

for epoch in range(training_epochs):
    classifier.train()
    log_string('\nEpoch %d (%d/%s):' % (global_epoch, epoch + 1, training_epochs))
    lr = max(args['TRAIN']['learning_rate'] * (args['TRAIN']['lr_decay'] ** (epoch // args['TRAIN']['decay_step'])), LEARNING_RATE_CLIP)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    momentum = max(MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // args['TRAIN']['decay_step'])), BN_MOMENTUM_MAX)
    if momentum < 0.01:
        momentum = 0.01
    classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
    train_loss = 0
    train_count = 0
    with tqdm(total=len(train_dataloader)) as pbar:
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            # seg_label = batch
            # print(len(batch[0]))
            seg_label = batch[0]['seg_labels']
            complet_label = batch[1]['complet_labels']
            invalid_voxels = batch[1]['complet_invalid']
            # for k in range(2):
            #     for key in batch[k].keys():
            #         if isinstance(batch[k][key], list):
            #             print(f'For {key}: len is {len(batch[k][key])}')
            #             print(batch[k][key])
            #             continue
            #         print(f' For {key}: {batch[k][key].shape}')
            #         torch.save(batch[k][key],f'./temp/{key}')
            # if i>=1:
            #     break
            # print(batch[2])
            seg_pred, complet_pred, sigma = classifier(batch)
            
            seg_label = seg_label.cuda()
            complet_label = complet_label.cuda()
            loss, loss_seg, loss_complet = criteria(seg_pred, seg_label, seg_labelweights,
                                                        complet_pred, complet_label, compl_labelweights,
                                                        invalid_voxels, sigma)

            '''Evaluation in trianing'''
            pred_choice_complet = complet_pred[-1].data.max(1)[1].to('cpu')
            complet_label = complet_label.to('cpu')
            complet_label[invalid_voxels==1] = 255
            correct_complet = pred_choice_complet.eq(complet_label.long().data).to('cpu')[(complet_label!=0)&(complet_label!=255)].sum()

            pred_choice_seg = seg_pred.data.max(1)[1].to('cpu')
            seg_label = seg_label.to('cpu')
            correct_seg = pred_choice_seg.eq(seg_label.long().data).to('cpu').sum()

            batch_loss = loss.cpu().item()
            train_loss += batch_loss

            loss.backward()
            optimizer.step()

            if i % 1000 == 0 and i > 0:
                torch.save(classifier.state_dict(), '%s/model_latest.pth' % experiment_dir)

            pbar.set_description('CLoss %.2f, SLoss %.2f, CAcc %.2f, SAcc %.2f' %
                                    (loss_complet.item(),
                                    loss_seg.item(),
                                    correct_complet.item() / float(complet_label[(complet_label!=0)&(complet_label!=255)].size()[0]),
                                    correct_seg.item() / float(seg_label.size()[0])))
            writer.add_scalar(model_name + '/Loss/Completion', loss_complet.item(), train_count)
            writer.add_scalar(model_name + '/Loss/Segmentation', loss_seg.item(), train_count)
            writer.add_scalar(model_name + '/Accuracy/Completion', correct_complet.item() / float(complet_label[(complet_label!=0)&(complet_label!=255)].size()[0]), train_count)
            writer.add_scalar(model_name + '/Accuracy/Segmentation', correct_seg.item() / float(seg_label.size()[0]), train_count)
            pbar.update(1)
            train_count += args['TRAIN']['batch_size']
            
            if args['GENERAL']['debug'] and i > 10:
                break
        
    log_string('Train Loss: %.3f' % (train_loss / len(train_dataloader)))


    with torch.no_grad():
        classifier.eval()
        complet_evaluator = iouEval(NUM_CLASS_COMPLET, [])
        seg_evaluator = iouEval(NUM_CLASS_SEG, [])
        epsilon = np.finfo(np.float32).eps

        with tqdm(total=len(val_dataloader)) as pbar:
            for i, batch in enumerate(val_dataloader):
                seg_label = batch[0]['seg_labels']
                complet_label = batch[1]['complet_labels']
                invalid_voxels = batch[1]['complet_invalid']
                seg_pred, complet_pred, _ = classifier(batch)
                try:
                    seg_pred, complet_pred, _ = classifier(batch)
                except:
                    print('Error in inference!!')
                    continue

                seg_label = seg_label.cuda()
                complet_label = complet_label.cuda()

                pred_choice_complet = complet_pred[-1].data.max(1)[1].to('cpu')
                complet_label = complet_label.to('cpu')

                pred_choice_seg = seg_pred.data.max(1)[1].to('cpu').data.numpy()
                seg_label = seg_label.to('cpu').data.numpy()

                complet_label = complet_label.data.numpy()
                pred_choice_complet = pred_choice_complet.numpy()
                invalid_voxels = invalid_voxels.data.numpy()
                masks = get_eval_mask(complet_label, invalid_voxels)

                target = complet_label[masks]
                pred = pred_choice_complet[masks]

                pred_choice_seg = pred_choice_seg[seg_label != -100]
                seg_label = seg_label[seg_label != -100]
                complet_evaluator.addBatch(pred.astype(int), target.astype(int))
                seg_evaluator.addBatch(pred_choice_seg.astype(int), seg_label.astype(int))
                pbar.update(1)

                
                if args['GENERAL']['debug'] and i > 10:
                    break

        log_string("\n  ========================== COMPLETION RESULTS ==========================  ")
        _, class_jaccard = complet_evaluator.getIoU()
        # m_jaccard = class_jaccard[1:].mean()
        m_jaccard = class_jaccard[0:].mean()

        # ignore = [0]
        ignore = []
        # print also classwise
        for i, jacc in enumerate(class_jaccard):
            if i not in ignore:
                log_string('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                    i=i, class_str=class_strings[class_inv_remap[i]], jacc=jacc*100))

        # compute remaining metrics.
        conf = complet_evaluator.get_confusion()
        precision = np.sum(conf[1:, 1:]) / (np.sum(conf[1:, :]) + epsilon)
        recall = np.sum(conf[1:, 1:]) / (np.sum(conf[:, 1:]) + epsilon)
        acc_cmpltn = (np.sum(conf[1:, 1:])) / (np.sum(conf) - conf[0, 0])
        mIoU_ssc = m_jaccard

        log_string("Precision =\t" + str(np.round(precision * 100, 2)) + '\n' +
                    "Recall =\t" + str(np.round(recall * 100, 2)) + '\n' +
                    "IoU Cmpltn =\t" + str(np.round(acc_cmpltn * 100, 2)) + '\n' +
                    "mIoU SSC =\t" + str(np.round(mIoU_ssc * 100, 2)))

        log_string("\n  ========================== SEGMENTATION RESULTS ==========================  ")
        _, class_jaccard = seg_evaluator.getIoU()
        m_jaccard = class_jaccard.mean()
        for i, jacc in enumerate(class_jaccard):
            log_string('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                i=i, class_str=seg_label_to_cat[i], jacc=jacc*100))
        log_string('Eval point avg class IoU: %f' % (m_jaccard*100))

        
        if best_iou_complt < acc_cmpltn:
            best_iou_complt = acc_cmpltn
        if best_iou_seg < m_jaccard:
            best_iou_seg = m_jaccard
        if best_iou_sem_complt < mIoU_ssc:
            best_iou_sem_complt = mIoU_ssc
            torch.save(classifier.state_dict(), '%s/model_segiou_%.4f_compltiou_%.4f_epoch%d.pth' % (experiment_dir, best_iou_seg, mIoU_ssc, epoch+1))

        log_string('\nBest segmentation IoU: %f' % (best_iou_seg * 100))
        log_string('Best semantic completion IoU: %f' % (best_iou_sem_complt * 100))
        log_string('Best completion IoU: %f' % (best_iou_complt * 100))
        writer.add_scalar(model_name + '/Val/seg_iou', m_jaccard, epoch+1)
        writer.add_scalar(model_name + '/Val/completion_iou', acc_cmpltn, epoch+1)
        writer.add_scalar(model_name + '/Val/sem_completion_iou', mIoU_ssc, epoch+1)

    global_epoch += 1
log_string('Done!')
writer.close()