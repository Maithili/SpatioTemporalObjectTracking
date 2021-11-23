#!/home/maithili/repos/GraphTrans/.venv/bin/python

import yaml
import json
import os
import sys
import shutil
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from GraphTranslatorModule import GraphTranslatorModule
from reader import RoutinesDataset
from encoders import TimeEncodingOptions
from analyzers import loss_options
from utils import visualize_datapoint

DEFAULT_CONFIG = 'config/default.yaml'

def run(cfg_in = {}):
    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg.update(cfg_in)
    cfg['DATA_INFO'] = {}
    if cfg_in['DATA_INFO'] is not None:
        with open(cfg_in['DATA_INFO']) as f:
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
                           only_dynamic_edges = cfg['ONLY_DYNAMIC_EDGES'])

    run_name = None
    try:
        run_name = cfg['NAME']
    except:
        pass

    tmp_path = os.path.join('logs','temp')
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    os.makedirs(tmp_path)

    wandb_logger = WandbLogger(name=run_name, save_dir=tmp_path, log_model=True)
    wandb_logger.experiment.config.update(cfg)
    run_name = wandb_logger.experiment.name

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
                              logging_analyzers=logging_loss_funcs,
                              use_spectral_loss=cfg['USE_SPECTRAL_LOSS'],
                              num_chebyshev_polys=cfg['NUM_CHEBYSHEV_POLYS'])

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
        visualize_datapoint(model, data.get_test_loader(), data.node_classes, data.edge_keys)
    

def run_from_config(config_filename):
    with open(os.path.join('config',config_filename)+'.yaml') as f:
        cfg = yaml.safe_load(f)
    run(cfg)

if __name__ == '__main__':
    assert len(sys.argv) < 3, "The script can take only one argument specifying the config file name, e.g. 'sample'"
    if len(sys.argv) > 1:
        run_from_config(sys.argv[1])
    else:
        run()
