import sys
sys.path.append('helpers')
import yaml
import os
import shutil
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from GraphTranslatorModule import GraphTranslatorModule
from reader_dummy import DummyDataset

import random
random.seed(23435)
from numpy import random as nrandom
nrandom.seed(23435)

output_dir = 'logs_default/dummy/'
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

data = DummyDataset()
with open('config/default.yaml') as f:
    cfg = yaml.safe_load(f)

model = GraphTranslatorModule(num_nodes=data.params['n_nodes'],
                                node_feature_len=data.params['n_len'],
                                context_len=data.params['c_len'],
                                edge_importance=cfg['EDGE_IMPORTANCE'],
                                edge_dropout_prob = cfg['EDGE_DROPOUT_PROB'],
                                learned_time_periods=cfg['LEARNED_TIME_PERIODS'],
                                hidden_layer_size=cfg['HIDDEN_LAYER_SIZE'],
                                learn_node_embeddings=cfg['LEARN_NODE_EMBEDDINGS'],
                                preprocess_context=True,
                                num_activities = 5,
                                context_type_to_use='time')


wandb_logger = WandbLogger(name='Dummy', log_model=True, save_dir='wandb_logs')
wandb_logger.watch(model, log="all", log_freq=1)
wandb_logger.experiment.config.update(cfg)
epoch = 500
output_dir_new = output_dir+'_'+str(epoch)+'epochs'
os.makedirs(output_dir_new)
trainer = Trainer(gpus = torch.cuda.device_count(), max_epochs=epoch, logger=wandb_logger, log_every_n_steps=1)
trainer.fit(model, data.get_train_loader())
trainer.test(model, data.get_test_loader())