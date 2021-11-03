import yaml
import os
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from GraphTranslatorModule import GraphTranslatorModule
from reader import RoutinesDataset
from encoders import time_encoding_options
from analyzers import loss_options

def run(cfg):
    try:
        wandb_logger = WandbLogger(name=cfg['MODEL']['NAME'])
    except:
        wandb_logger = WandbLogger()
    wandb_logger.experiment.config.update(cfg)
    
    assert cfg['MODEL']['TIME_ENCODING'] in time_encoding_options, 'Time encoding {} specified in config should be one of {}'.format(cfg['MODEL']['TIME_ENCODING'], time_encoding_options.keys())
    time_encoding = time_encoding_options[cfg['MODEL']['TIME_ENCODING']]

    data = RoutinesDataset(data_path=cfg['DATA_PATH']['GRAPHS'], classes_path=cfg['DATA_PATH']['CLASSES'], time_encoder=time_encoding, test_perc=cfg['TRAIN']['TEST_SPLIT'])
    
    losses = loss_options(data)

    assert cfg['TRAIN']['LOSS'] in losses, 'Loss {} specified in config should be one of {}'.format(cfg['TRAIN']['LOSS'], losses.keys())
    train_loss = losses(cfg['TRAIN']['LOSS'])
    logging_loss_funcs = []
    for logging_loss in cfg['TRAIN']['LOSSES_LOG']:
        assert logging_loss in losses, 'Loss {} specified in config should be one of {}'.format(logging_loss, losses.keys())
        logging_loss_funcs.append(losses(logging_loss))

    model = GraphTranslatorModule(num_nodes=data.n_nodes, 
                              node_feature_len=data.n_len, 
                              edge_feature_len=data.e_len, 
                              context_len=data.c_len, 
                              train_analyzer=train_loss, 
                              logging_analyzers=logging_loss_funcs)

    trainer = Trainer(max_epochs=cfg['TRAIN']['EPOCHS'], logger=wandb_logger)

    trainer.fit(model, data.get_train_loader())
    trainer.test(model, data.get_test_loader())

    wandb_logger.experiment.finish()

def run_from_config(config_filename):
    with open(os.path.join('config',config_filename)+'.yaml') as f:
        cfg = yaml.safe_load(f)
    run(cfg)

