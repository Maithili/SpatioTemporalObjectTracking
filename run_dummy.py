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
                                preprocess_context=cfg['PREPROCESS_CONTEXT'])

epochs = cfg['EPOCHS'] if isinstance(cfg['EPOCHS'],list) else [cfg['EPOCHS']]
done_epochs = 0

wandb_logger = WandbLogger(name='Dummy', log_model=True, save_dir='wandb_logs')
epoch = 20
output_dir_new = output_dir+'_'+str(epoch)+'epochs'
os.makedirs(output_dir_new)
trainer = Trainer(gpus = torch.cuda.device_count(), max_epochs=epoch, logger=wandb_logger, log_every_n_steps=5)
trainer.fit(model, data.get_train_loader())
trainer.test(model, data.get_test_loader())