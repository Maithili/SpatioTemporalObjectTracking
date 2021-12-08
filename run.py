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
from reader import RoutinesDataset
from encoders import TimeEncodingOptions
from filters import loss_filter_options
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
                           avg_samples_per_routine=cfg['AVG_SAMPLES_PER_ROUTINE'],
                           sequential_prediction=cfg['SEQUENTIAL_PREDICTION'],
                           only_dynamic_edges = cfg['ONLY_DYNAMIC_EDGES'],
                           allow_multiple_edge_types=cfg['ALLOW_MULTIPLE_EDGE_TYPES'],
                           ignore_close_edges = cfg['IGNORE_CLOSE_EDGES'])

    tmp_path = os.path.join('logs','temp')
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    os.makedirs(tmp_path)

    wandb_logger = WandbLogger(name=cfg['NAME'], save_dir=tmp_path, log_model=True)
    wandb_logger.experiment.config.update(cfg)
    run_name = wandb_logger.experiment.name

    wandb_logger.experiment.config['DATA_PARAM'] = data.params
    
    losses = loss_filter_options(data)

    assert cfg['LOSS'] in losses, 'Loss {} specified in config is invalid'.format(cfg['LOSS'])
    train_loss = losses(cfg['LOSS'])
    logging_loss_funcs = []
    for logging_loss in cfg['LOSSES_LOG']:
        assert logging_loss in losses, f'Loss {logging_loss} specified in config should be one of {losses.keys()}'
        logging_loss_funcs.append(losses(logging_loss))

    model = GraphTranslatorModule(num_nodes=data.params['n_nodes'],
                              node_feature_len=data.params['n_len'],
                              edge_feature_len=data.params['e_len'],
                              context_len=data.params['c_len'],
                              train_analyzer=train_loss, 
                              logging_analyzers=logging_loss_funcs,
                              use_spectral_loss=cfg['USE_SPECTRAL_LOSS'],
                              num_chebyshev_polys=cfg['NUM_CHEBYSHEV_POLYS'],
                              allow_multiple_edge_types=cfg['ALLOW_MULTIPLE_EDGE_TYPES'])

    trainer = Trainer(max_epochs=cfg['EPOCHS'], logger=wandb_logger, log_every_n_steps=5)

    trainer.fit(model, data.get_train_loader())
    trainer.test(model, data.get_test_loader())
    
    try:
        output_dir = os.path.join('logs',run_name)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.rename(tmp_path, output_dir)
        print('Outputs saved at ',output_dir)
    except Exception as e:
        print(e)
    finally:
        # visualize_datapoint(model, data.get_test_loader(), data.node_classes, data.edge_keys)
        pass



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