import sys
sys.path.append('helpers')
import yaml
import json
import os
import glob
import argparse
from copy import deepcopy
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

from GraphTranslatorModule import GraphTranslatorModule
from reader import RoutinesDataset, INTERACTIVE, get_cooccurence_frequency, get_spectral_components
from encoders import TimeEncodingOptions
from breakdown_evaluations import evaluate as evaluate_applications
from baselines.baselines import LastSeen, StaticSemantic, LastSeenAndStaticSemantic, Fremen, FremenStateConditioned

import random
random.seed(23435)
from numpy import random as nrandom
nrandom.seed(23435)

def run_model(data, group, checkpoint_dir=None, read_ckpt=False, write_ckpt=False, tags=[], logs_dir='logs'):
    output_dir = os.path.join(logs_dir, group,cfg['NAME'])
    if os.path.exists(output_dir):
        n = 1
        new_dir = output_dir + "_"+str(n)
        while os.path.exists(new_dir):
            n += 1
            new_dir = output_dir + "_"+str(n)
        output_dir = new_dir
    os.makedirs(output_dir)

    wandb_logger = WandbLogger(name=cfg['NAME'], log_model=True, group = group, tags = tags, mode='disabled')

    cfg['DATA_PARAM'] = data.params
    wandb_logger.experiment.config.update(cfg)
    


    if read_ckpt:
        checkpoint_file = max(glob.glob(checkpoint_dir+'/*.ckpt'), key=os.path.getctime)
        model = GraphTranslatorModule.load_from_checkpoint(checkpoint_file, 
                                                            num_nodes=data.params['n_nodes'],
                                                            node_feature_len=data.params['n_len'],
                                                            context_len=data.params['c_len'],
                                                            edge_importance=cfg['EDGE_IMPORTANCE'],
                                                            edge_dropout_prob = cfg['EDGE_DROPOUT_PROB'],
                                                            tn_loss_weight=cfg['TN_LOSS_WEIGHT'],
                                                            learned_time_periods=cfg['LEARNED_TIME_PERIODS'],
                                                            hidden_layer_size=cfg['HIDDEN_LAYER_SIZE'])
        evaluation_summary = evaluate_applications(model, data, cfg, output_dir, logger=wandb_logger.experiment, print_importance=True)

    else:
        model = GraphTranslatorModule(num_nodes=data.params['n_nodes'],
                                node_feature_len=data.params['n_len'],
                                context_len=data.params['c_len'],
                                edge_importance=cfg['EDGE_IMPORTANCE'],
                                edge_dropout_prob = cfg['EDGE_DROPOUT_PROB'],
                                tn_loss_weight=cfg['TN_LOSS_WEIGHT'],
                                learned_time_periods=cfg['LEARNED_TIME_PERIODS'],
                                hidden_layer_size=cfg['HIDDEN_LAYER_SIZE'])

        epochs = cfg['EPOCHS'] if isinstance(cfg['EPOCHS'],list) else [cfg['EPOCHS']]
        done_epochs = 0

        for epoch in epochs:
            if output_dir[-2] == '_':
                output_dir_new = output_dir[:-2]+'_'+str(epoch)+'epochs'+output_dir[-2:]
            else:
                output_dir_new = output_dir+'_'+str(epoch)+'epochs'
            os.makedirs(output_dir_new)
            if write_ckpt:
                ckpt_callback = ModelCheckpoint(dirpath=output_dir_new)
                trainer = Trainer(gpus = torch.cuda.device_count(), max_epochs=epoch-done_epochs, logger=wandb_logger, log_every_n_steps=5, callbacks=[ckpt_callback])

            else:
                trainer = Trainer(gpus = torch.cuda.device_count(), max_epochs=epoch-done_epochs, logger=wandb_logger, log_every_n_steps=5)

            trainer.fit(model, data.get_train_loader())
            trainer.test(model, data.get_test_loader())
            done_epochs = epoch

            evaluation_summary = evaluate_applications(model, data, cfg, output_dir_new, logger=wandb_logger.experiment, print_importance=True)


            with open (os.path.join(output_dir_new,'config.json'), 'w') as f:
                json.dump(cfg, f)

            print('Outputs saved at ',output_dir_new)

            if write_ckpt:
                torch.save(model.state_dict(), os.path.join(output_dir_new,'weights.pt'))
    
    
    wandb.finish()



def run(data_dir, cfg = {}, baselines=False, ckpt_dir=None, read_ckpt=False, write_ckpt=False, tags=[], train_days=None, logs_dir='logs'):
    
    if cfg['NAME'] is None:
        cfg['NAME'] = os.path.basename(data_dir)+'_trial'

    with open(os.path.join(data_dir, 'processed', 'common_data.json')) as f:
        cfg['DATA_INFO'] = json.load(f)['info']

    time_options = TimeEncodingOptions(cfg['DATA_INFO']['weeekend_days'] if 'weeekend_days' in cfg['DATA_INFO'].keys() else None)
    time_encoding = time_options(cfg['TIME_ENCODING'])

    data = RoutinesDataset(data_path=os.path.join(data_dir,'processed'), 
                           time_encoder=time_encoding, 
                           batch_size=cfg['BATCH_SIZE'],
                           only_seen_edges = cfg['ONLY_SEEN_EDGES'],
                           max_routines = (train_days, None))
    
    group = os.path.basename(data_dir)

    if baselines:
        cf = get_cooccurence_frequency(data)
        spec = get_spectral_components(data, periods_mins=[float('inf'), 60*24, 60*24/2])
        for baseline in [LastSeen(cf), StaticSemantic(cf), LastSeenAndStaticSemantic(cf), Fremen(spec), FremenStateConditioned(spec, data.params['dt'])]:
            output_dir = os.path.join(logs_dir, group,baseline.__class__.__name__)
            if os.path.exists(output_dir):
                n = 1
                new_dir = output_dir + "_"+str(n)
                while os.path.exists(new_dir):
                    n += 1
                    new_dir = output_dir + "_"+str(n)
                output_dir = new_dir
            os.makedirs(output_dir)

            wandb.init(name=baseline.__class__.__name__, dir=output_dir, group = group, tags=tags, mode='disabled')
            cfg['NAME'] = wandb.run.name
            wandb.config.update(cfg)
            for routine in data.test:
                eval, details = baseline.step(data.test.collate_fn([routine]))
            # wandb.log(baseline.log())
            _ = evaluate_applications(baseline, data, cfg, output_dir)


            with open (os.path.join(output_dir,'config.json'), 'w') as f:
                json.dump(cfg, f)

            print('Outputs saved at ',output_dir)
            wandb.finish()
    else:
        run_model(data, group=group, checkpoint_dir=ckpt_dir, read_ckpt=read_ckpt, write_ckpt=write_ckpt, tags=tags, logs_dir=logs_dir)







if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model on routines.')
    parser.add_argument('--path', type=str, default='data/Persona0219/hard_worker', help='Path where the data lives. Must contain routines, info and classes json files.')
    parser.add_argument('--architecture_cfg', type=str, help='Name of config file.')
    parser.add_argument('--cfg', type=str, help='Name of config file.')
    parser.add_argument('--train_days', type=int, help='Number of routines to train on.')
    parser.add_argument('--name', type=str, help='Name of run.')
    parser.add_argument('--tags', type=str, help='Tags for the run separated by a comma \',\'')
    parser.add_argument('--baselines', action='store_true')
    parser.add_argument('--ckpt_dir', type=str, help='Path to checkpoint file')
    parser.add_argument('--read_ckpt', action='store_true')
    parser.add_argument('--write_ckpt', action='store_true')
    parser.add_argument('--logs_dir', type=str, default='logs', help='Path to store putputs.')

    args = parser.parse_args()

    with open('config/default.yaml') as f:
        cfg = yaml.safe_load(f)

    if args.architecture_cfg is not None:
        with open(os.path.join('config',args.architecture_cfg)+'.yaml') as f:
            cfg.update(yaml.safe_load(f))

    if args.cfg is not None:
        with open(os.path.join('config',args.cfg)+'.yaml') as f:
            cfg.update(yaml.safe_load(f))
    if args.name is not None:
        cfg['NAME'] = args.name

    cfg['MAX_TRAINING_SAMPLES'] = args.train_days
    tags = args.tags.split(',') if args.tags is not None else []

    print(args.logs_dir)
    run(data_dir=args.path, cfg=cfg, baselines=args.baselines, ckpt_dir=args.ckpt_dir, read_ckpt=args.read_ckpt, write_ckpt=args.write_ckpt, tags = tags, train_days=args.train_days, logs_dir=args.logs_dir)