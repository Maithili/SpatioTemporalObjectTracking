#!/home/maithili/repos/GraphTrans/.venv/bin/python

import yaml
import os
import sys
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from GraphTranslatorModule import GraphTranslatorModule
from reader import RoutinesDataset
from encoders import time_encoding_options
from analyzers import loss_options

DEFAULT_CONFIG = 'config/default.yaml'

def run(cfg_in):
    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg.update(cfg_in)

    try:
        wandb_logger = WandbLogger(name=cfg['NAME'])
    except:
        wandb_logger = WandbLogger()
    wandb_logger.experiment.config.update(cfg)
    
    assert cfg['TIME_ENCODING'] in time_encoding_options, 'Time encoding {} specified in config should be one of {}'.format(cfg['TIME_ENCODING'], time_encoding_options.keys())
    time_encoding = time_encoding_options[cfg['TIME_ENCODING']]

    data = RoutinesDataset(data_path=cfg['DATA_PATH'], 
                           classes_path=cfg['CLASSES_PATH'], 
                           time_encoder=time_encoding, 
                           test_perc=cfg['TEST_SPLIT'], 
                           edges_of_interest=cfg['EDGES_OF_INTEREST'], 
                           sample_data=cfg['SAMPLE_DATA'],
                           sampling_ratio=cfg['SAMPLING_RATIO'])

    wandb_logger.experiment.config['DATA_PARAM'] = data.params
    
    losses = loss_options(data)

    assert cfg['LOSS'] in losses, 'Loss {} specified in config should be one of {}'.format(cfg['LOSS'], losses.keys())
    train_loss = losses(cfg['LOSS'])
    logging_loss_funcs = []
    for logging_loss in cfg['LOSSES_LOG']:
        assert logging_loss in losses, 'Loss {} specified in config should be one of {}'.format(logging_loss, losses.keys())
        logging_loss_funcs.append(losses(logging_loss))

    model = GraphTranslatorModule(num_nodes=data.params['n_nodes'],
                              node_feature_len=data.params['n_len'],
                              edge_feature_len=data.params['e_len'],
                              context_len=data.params['c_len'],
                              train_analyzer=train_loss, 
                              logging_analyzers=logging_loss_funcs)

    trainer = Trainer(max_epochs=cfg['EPOCHS'], logger=wandb_logger)

    trainer.fit(model, data.get_train_loader())
    trainer.test(model, data.get_test_loader())

    wandb_logger.experiment.finish(exit_code=0)

def run_from_config(config_filename):
    with open(os.path.join('config',config_filename)+'.yaml') as f:
        cfg = yaml.safe_load(f)
    run(cfg)

if __name__ == '__main__':
    assert len(sys.argv) == 2, "The script requires exactly one argument specifying the config file name, e.g. 'sample'"
    run_from_config(sys.argv[1])
