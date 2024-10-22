import os
import pickle
import logging
import sys

import kornia
import torch
import numpy as np

from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import utils
# from models_new import  Enhancer
from models import  Enhancer
from utils import create_dir, logger, set_random_seed, CharbonnierLoss, create_folder_for_run
from collections import defaultdict
import dataset
from registry import DATASET_REGISTRY


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
        if val != np.nan and val != np.inf:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


from omegaconf import OmegaConf


def test():
    args = OmegaConf.load("option/test.yaml")

    this_run_folder = 'pretrained_model/sRGB'
    options_file = os.path.join(this_run_folder, 'options-and-config.pickle')
    trainOpt, dataSetOpt = utils.load_options(options_file)

    runName = "{}-{}-{}".format(dataSetOpt.name[:3], dataSetOpt.name[-3:],
                                                            dataSetOpt.version[-1:])
    checkpoint, loaded_checkpoint_file_name = utils.load_epoch_checkpoint(
        this_run_folder, runName)

    # if args.dataset.data_root is not None:
    dataSetOpt.data_root = args.args.dataset.data_root

    set_random_seed(trainOpt.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    enhancer = Enhancer(trainOpt)
    assert checkpoint is not None
    logging.info(f'Loading checkpoint from file {loaded_checkpoint_file_name}')
    utils.model_from_checkpoint(enhancer, checkpoint)
    logging.info('Begin Training......')

    val_data = DataLoader(
        DATASET_REGISTRY.get(dataSetOpt.name)(dataSetOpt, mode="test"),
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )

    logging.info('Number of test images: {:d}'.format(len(val_data)))

    val_average = defaultdict(AverageMeter)
    loopVal = tqdm(val_data, total=len(val_data))
    enhancer.model.eval()
    for i, batch in enumerate(loopVal):
        # if batch['input_name'][0]=='a2594.jpg':
            meters, (inputs, labels, outputs, inf, text) = enhancer.validate_on_batch(batch)
            for name, meter in meters.items():
                val_average[name].update(meter)
            utils.save_images_tofile((inputs, outputs, labels, inf), os.path.join(this_run_folder, 'results'),
                                     batch['input_name'][0],args.output)
            if i % trainOpt.print_each_step == 0:
                loopVal.set_description(
                    "test|  Step: {}/{} loss:{:.4f},psnr:{:.4f},ssim:{:.4f}"
                    .format(i, len(val_data),
                            val_average['loss'].avg, val_average['psnr'].avg, val_average['ssim'].avg))

    utils.log_progress(val_average)


if __name__ == '__main__':
    test()
