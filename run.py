#!./.venv/bin/python

import yaml
import json
import os
import argparse
from copy import deepcopy
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb

from GraphTranslatorModule import GraphTranslatorModule
from readerFileBased import RoutinesDataset, INTERACTIVE, get_cooccurence_frequency
from encoders import TimeEncodingOptions
from utils import visualize_unconditional_datapoint, visualize_conditional_datapoint
from applications import evaluate_applications
from baselines.baselines import LastSeen, StaticSemantic, LastSeenAndStaticSemantic, LastSeenButMostlyStaticSemantic, Slim

DEFAULT_CONFIG = 'config/default.yaml'

def run_model(data, group):
    output_dir = os.path.join('logs',cfg['NAME'])
    if os.path.exists(output_dir):
        n = 1
        new_dir = output_dir + "_"+str(n)
        while os.path.exists(new_dir):
            n += 1
            new_dir = output_dir + "_"+str(n)
        output_dir = new_dir
    os.makedirs(output_dir)

    wandb_logger = WandbLogger(name=cfg['NAME'], save_dir=output_dir, log_model=True, group = group)
    wandb_logger.experiment.config.update(cfg)

    wandb_logger.experiment.config['DATA_PARAM'] = data.params
    

    model = GraphTranslatorModule(num_nodes=data.params['n_nodes'],
                            node_feature_len=data.params['n_len'],
                            context_len=data.params['c_len'],
                            learn_nodes=cfg['LEARN_NODES'],
                            edge_importance=cfg['EDGE_IMPORTANCE'],
                            edge_dropout_prob = cfg['EDGE_DROPOUT_PROB'],
                            tn_loss_weight=cfg['TN_LOSS_WEIGHT'],
                            learn_context=cfg['LEARN_CONTEXT'])

    trainer = Trainer(gpus = torch.cuda.device_count(), max_epochs=cfg['EPOCHS'], logger=wandb_logger, log_every_n_steps=5)
    wandb_logger.watch(model, log='gradients', log_freq=20)

    trainer.fit(model, data.get_train_loader())
    trainer.test(model, data.get_test_loader())

    evaluation_summary = evaluate_applications(model, data, cfg, output_dir)
    model.log(evaluation_summary)

    print('Outputs saved at ',output_dir)
    if INTERACTIVE:
        visualize_unconditional_datapoint(model, data.test_routines, data.node_classes, use_output_nodes=cfg['LEARN_NODES'])
        visualize_conditional_datapoint(model, data.get_single_example_test_loader(), data.node_classes, use_output_nodes=cfg['LEARN_NODES'])

    wandb.finish()



def run(data_dir, cfg = {}):
    
    if cfg['NAME'] is None:
        cfg['NAME'] = os.path.basename(data_dir)

    with open(os.path.join(data_dir, 'processed', 'common_data.json')) as f:
        cfg['DATA_INFO'] = json.load(f)['info']

    time_options = TimeEncodingOptions(cfg['DATA_INFO']['weeekend_days'] if 'weeekend_days' in cfg['DATA_INFO'].keys() else None)
    time_encoding = time_options(cfg['TIME_ENCODING'])

    data = RoutinesDataset(data_path=os.path.join(data_dir,'processed'), 
                           time_encoder=time_encoding, 
                           batch_size=cfg['BATCH_SIZE'],
                           only_seen_edges = cfg['ONLY_SEEN_EDGES'])
    
    run_model(data, group = os.path.basename(data_dir))

    if cfg['RUN_BASELINES']:
        cf = get_cooccurence_frequency(data)
        for baseline in [LastSeen(cf), StaticSemantic(cf), LastSeenAndStaticSemantic(cf), LastSeenButMostlyStaticSemantic(cf)]:
            output_dir = os.path.join('logs','baselines',baseline.__class__.__name__)
            wandb.init(name=baseline.__class__.__name__, dir=output_dir, group = os.path.basename(data_dir))
            wandb.config['NAME'] = wandb.run.name
            for routine in data.test:
                eval, details = baseline.step(data.test.collate_fn([routine]))
            # wandb.log(baseline.log())
            evaluation_summary = evaluate_applications(baseline, data, cfg, output_dir)
            print(evaluation_summary)
            wandb.log(evaluation_summary)

            print('Outputs saved at ',output_dir)
            if INTERACTIVE:
                visualize_unconditional_datapoint(baseline, data.test_routines, data.node_classes, use_output_nodes=cfg['LEARN_NODES'])
                visualize_conditional_datapoint(baseline, data.get_single_example_test_loader(), data.node_classes, use_output_nodes=cfg['LEARN_NODES'])
            wandb.finish()






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model on routines.')
    parser.add_argument('--path', type=str, default='data/mcsample0207', help='Path where the data lives. Must contain routines, info and classes json files.')
    parser.add_argument('--cfg', type=str, help='Name of config file.')
    parser.add_argument('--name', type=str, help='Name of run.')

    args = parser.parse_args()

    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    if args.cfg is not None:
        with open(os.path.join('config',args.cfg)+'.yaml') as f:
            cfg.update(yaml.safe_load(f))
    if args.name is not None:
        cfg['NAME'] = args.name
    run(data_dir=args.path, cfg=cfg)