import shutil
import sys
sys.path.append('helpers')
import yaml
import json
import os
import glob
import argparse
from adict import adict
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from ObjectActivityCoembedding import ObjectActivityCoembeddingModule
from reader_sequential import RoutinesDataset
from encoders import TimeEncodingOptions

import random
random.seed(23435)
from numpy import random as nrandom
nrandom.seed(23435)

def run_model(data, group, cfg = {}, checkpoint_dir=None, read_ckpt=False, write_ckpt=False, tags=[], logs_dir='logs', finetune=False):


    cfg.update(data.params)
    model_configs = adict(cfg)

    
    output_dir = os.path.join(logs_dir, group,cfg['NAME'])
    if os.path.exists(output_dir):
        n = 1
        new_dir = output_dir + "_"+str(n)
        while os.path.exists(new_dir):
            n += 1
            new_dir = output_dir + "_"+str(n)
        output_dir = new_dir
    os.makedirs(output_dir)

    wandb_logger = WandbLogger(name=cfg['NAME'], log_model=True, group = group, tags = tags)
    wandb_logger.experiment.config.update(cfg)

    model = ObjectActivityCoembeddingModule(model_configs = model_configs)

    epochs = cfg['epochs'] if isinstance(cfg['epochs'],list) else [cfg['epochs']]
    done_epochs = 0

    for epoch in epochs:
        if output_dir[-2] == '_':
            output_dir_new = output_dir[:-2]+'_'+str(epoch)+'epochs'+output_dir[-2:]
        else:
            output_dir_new = output_dir+'_'+str(epoch)+'epochs'
        os.makedirs(output_dir_new)
        if write_ckpt:
            ckpt_callback = ModelCheckpoint(dirpath=output_dir_new)
            trainer = Trainer(gpus = torch.cuda.device_count(), logger=wandb_logger, max_epochs=epoch-done_epochs, log_every_n_steps=5, callbacks=[ckpt_callback])

        else:
            trainer = Trainer(gpus = torch.cuda.device_count(), logger=wandb_logger, max_epochs=epoch-done_epochs, log_every_n_steps=5)

        wandb_logger.watch(model, log='gradients', log_freq=20)

        trainer.fit(model, data.get_train_loader())
        trainer.test(model, data.get_test_loader())
        done_epochs = epoch

        # evaluation_summary = evaluate_applications_original(model, data, output_dir_new, learned_model=True, print_importance=False, confidences=cfg['action_probability_thresholds'])
        # evaluation_summary = evaluate_applications_new(model, data, output_dir_new)
        # evaluation_summary = evaluate_applications_stepahead(model, data, output_dir)


        with open (os.path.join(output_dir_new,'config.json'), 'w') as f:
            json.dump(cfg, f, indent=4)

        print('Outputs saved at ',output_dir_new)

        if write_ckpt:
            torch.save(model.state_dict(), os.path.join(output_dir_new,'weights.pt'))
    

def run(data_dir, cfg = {}, baselines=False, ckpt_dir=None, read_ckpt=False, write_ckpt=False, tags=[], train_days=None, logs_dir='logs', finetune=False):
    
    if cfg['NAME'] is None:
        cfg['NAME'] = os.path.basename(data_dir)+'_trial'

    with open(os.path.join(data_dir, 'processed_seq', 'common_data.json')) as f:
        cfg['DATA_INFO'] = json.load(f)['info']

    time_options = TimeEncodingOptions(cfg['DATA_INFO']['weeekend_days'] if 'weeekend_days' in cfg['DATA_INFO'].keys() else None)
    time_encoding = time_options(cfg['time_encoding'])

    data = RoutinesDataset(data_path=os.path.join(data_dir,'processed_seq'), 
                           time_encoder=time_encoding, 
                           batch_size=cfg['batch_size'],
                           max_routines = (train_days, None))
    
    group = os.path.basename(data_dir)

    run_model(data, group=group, cfg = cfg, checkpoint_dir=ckpt_dir, read_ckpt=read_ckpt, write_ckpt=write_ckpt, tags=tags, logs_dir=logs_dir, finetune=finetune)







if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model on routines.')
    parser.add_argument('--path', type=str, default='data/personaWithoutClothes/persona0', help='Path where the data lives. Must contain routines, info and classes json files.')
    parser.add_argument('--cfg', type=str, help='Name of config file.')
    parser.add_argument('--train_days', type=int, help='Number of routines to train on.')
    parser.add_argument('--name', type=str, default='trial', help='Name of run.')
    parser.add_argument('--tags', type=str, help='Tags for the run separated by a comma \',\'')
    parser.add_argument('--baselines', action='store_true')
    parser.add_argument('--ckpt_dir', type=str, help='Path to checkpoint file')
    parser.add_argument('--read_ckpt', action='store_true')
    parser.add_argument('--write_ckpt', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--logs_dir', type=str, default='logs_default', help='Path to store putputs.')

    args = parser.parse_args()

    with open('config/default.yaml') as f:
        cfg = yaml.safe_load(f)

    if args.cfg is not None:
        with open(os.path.join('config',args.cfg)+'.yaml') as f:
            cfg.update(yaml.safe_load(f))
    if args.name is not None:
        cfg['NAME'] = args.name

    cfg['MAX_TRAINING_SAMPLES'] = args.train_days
    tags = args.tags.split(',') if args.tags is not None else []

    print(args.logs_dir)
    run(data_dir=args.path, cfg=cfg, baselines=args.baselines, ckpt_dir=args.ckpt_dir, read_ckpt=args.read_ckpt, write_ckpt=args.write_ckpt, tags = tags, train_days=args.train_days, logs_dir=args.logs_dir, finetune=args.finetune)
