#!./.venv/bin/python

import yaml
import json
import os
import sys
import shutil
import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from GraphTranslatorModule import GraphTranslatorModule
from reader import RoutinesDataset, INTERACTIVE
from encoders import TimeEncodingOptions
from filters import OutputFilters
from utils import visualize_datapoint

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

    data = RoutinesDataset(data_path=cfg['DATA_PATH'], 
                           classes_path=cfg['CLASSES_PATH'], 
                           time_encoder=time_encoding, 
                           dt=cfg['DT'],
                           test_perc=cfg['TEST_SPLIT'], 
                           edges_of_interest=cfg['EDGES_OF_INTEREST'], 
                           sample_data=cfg['SAMPLE_DATA'],
                           batch_size=cfg['BATCH_SIZE'],
                           only_seen_edges = cfg['ONLY_SEEN_EDGES'],
                           tree_formuation = cfg['TREE_FORMULATION'],
                           ignore_close_edges = cfg['IGNORE_CLOSE_EDGES'])

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
                              node_class_len=data.params['n_class_len'],
                              node_state_len=data.params['n_state_len'],
                              context_len=data.params['c_len'],
                              use_spectral_loss=cfg['USE_SPECTRAL_LOSS'],
                              num_chebyshev_polys=cfg['NUM_CHEBYSHEV_POLYS'],
                              tree_formulation=cfg['TREE_FORMULATION'],
                              learn_nodes=cfg['LEARN_NODES'],
                              edges_as_attention=cfg['EDGES_AS_ATTENTION'])

    trainer = Trainer(max_epochs=cfg['EPOCHS'], logger=wandb_logger, log_every_n_steps=5)

    trainer.fit(model, data.get_train_loader())
    trainer.test(model, data.get_test_loader())
    
    print('Outputs saved at ',output_dir)
    if INTERACTIVE:
        visualize_datapoint(model, data.get_single_example_test_loader(), data.node_classes, use_output_nodes=cfg['LEARN_NODES'])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model on routines.')
    parser.add_argument('--path', type=str, help='Path where the data lives. Must contain sample, info and classes json files.')
    parser.add_argument('--cfg', type=str, help='Name of config file.')

    args = parser.parse_args()
    assert len(sys.argv) < 4, "The script can take only one argument specifying the config file name, e.g. 'sample'"

    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    if args.cfg is not None:
        with open(os.path.join('config',args.cfg)+'.yaml') as f:
            cfg.update(yaml.safe_load(f))
    run(cfg=cfg, path=args.path)