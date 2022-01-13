#!./.venv/bin/python

import yaml
import json
import os
import argparse
from copy import deepcopy
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from GraphTranslatorModule import GraphTranslatorModule
from reader import RoutinesDataset, INTERACTIVE
from encoders import TimeEncodingOptions
from utils import visualize_unconditional_datapoint, visualize_conditional_datapoint
from applications import multiple_steps, object_search

DEFAULT_CONFIG = 'config/default.yaml'

def run(cfg = {}, path = None):
    
    if path is not None:
        cfg['DATA_PATH'] = os.path.join(path, 'sample.json')
        cfg['CLASSES_PATH'] = os.path.join(path, 'classes.json')
        cfg['DATA_INFO'] = os.path.join(path, 'info.json')
        if cfg['NAME'] is None:
            cfg['NAME'] = os.path.basename(path)
    else:
        print('No path provided. Will read from config file...')

    with open(cfg['DATA_INFO']) as f:
        cfg['DATA_INFO'] = json.load(f)

    time_options = TimeEncodingOptions(cfg['DATA_INFO']['weeekend_days'] if 'weeekend_days' in cfg['DATA_INFO'].keys() else None)
    time_encoding = time_options(cfg['TIME_ENCODING'])

    if (cfg['DT'] != cfg['DATA_INFO']['dt']): print('Different dt found in config {} and data {}. The former will be used'.format(cfg['DT'], cfg['DATA_INFO']['dt']))

    data = RoutinesDataset(data_path=cfg['DATA_PATH'], 
                           classes_path=cfg['CLASSES_PATH'], 
                           time_encoder=time_encoding, 
                           dt=cfg['DT'],
                           test_perc=cfg['TEST_SPLIT'], 
                           batch_size=cfg['BATCH_SIZE'],
                           only_seen_edges = cfg['ONLY_SEEN_EDGES'])

    output_dir = os.path.join('logs',cfg['NAME'])
    if os.path.exists(output_dir):
        n = 1
        new_dir = output_dir + "_"+str(n)
        while os.path.exists(new_dir):
            n += 1
            new_dir = output_dir + "_"+str(n)
        output_dir = new_dir
    os.makedirs(output_dir)

    wandb_logger = WandbLogger(name=cfg['NAME'], save_dir=output_dir, log_model=True)
    wandb_logger.experiment.config.update(cfg)

    wandb_logger.experiment.config['DATA_PARAM'] = data.params
    

    model = GraphTranslatorModule(num_nodes=data.params['n_nodes'],
                              node_feature_len=data.params['n_len'],
                              context_len=data.params['c_len'],
                              learn_nodes=cfg['LEARN_NODES'],
                              edge_importance=cfg['EDGE_IMPORTANCE'],
                              edge_dropout_prob = cfg['EDGE_DROPOUT_PROB'])

    trainer = Trainer(max_epochs=cfg['EPOCHS'], logger=wandb_logger, log_every_n_steps=5)

    trainer.fit(model, data.get_train_loader())
    trainer.test(model, data.get_test_loader())
    
    evaluation = {}
    hit_ratios, _ = object_search(model, deepcopy(data.test_routines), cfg['DATA_INFO']['search_object_ids'], data.node_idx_from_id)
    evaluation['Search hits'] = tuple(hit_ratios)
    evaluation['Conditional accuracy drift'] = tuple(multiple_steps(model, deepcopy(data.test_routines)))
    evaluation['Un-Conditional accuracy drift'] = tuple(multiple_steps(model, deepcopy(data.test_routines), unconditional=True))
    print('Test Evaluation', evaluation)

    print('Outputs saved at ',output_dir)
    if INTERACTIVE:
        visualize_unconditional_datapoint(model, data.test_routines, data.node_classes, use_output_nodes=cfg['LEARN_NODES'])
        visualize_conditional_datapoint(model, data.get_single_example_test_loader(), data.node_classes, use_output_nodes=cfg['LEARN_NODES'])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model on routines.')
    parser.add_argument('--path', type=str, help='Path where the data lives. Must contain sample, info and classes json files.')
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
    run(cfg=cfg, path=args.path)