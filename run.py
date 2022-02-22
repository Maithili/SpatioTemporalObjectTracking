#!./.venv/bin/python

from gc import callbacks
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
from readerFileBased import RoutinesDataset, INTERACTIVE, get_cooccurence_frequency, get_spectral_components
from encoders import TimeEncodingOptions
from utils import visualize_unconditional_datapoint, visualize_conditional_datapoint
# from applications import evaluate_applications
from breakdown_evaluations import evaluate as evaluate_applications
from baselines.baselines import LastSeen, StaticSemantic, LastSeenAndStaticSemantic, LastSeenButMostlyStaticSemantic, Fremen, FremenStateConditioned, Slim

def run_model(data, group, checkpoint_dir=None, read_ckpt=False, write_ckpt=False, tags=[]):
    output_dir = os.path.join('logs',cfg['NAME'])
    if os.path.exists(output_dir):
        n = 1
        new_dir = output_dir + "_"+str(n)
        while os.path.exists(new_dir):
            n += 1
            new_dir = output_dir + "_"+str(n)
        output_dir = new_dir
    os.makedirs(output_dir)

    wandb_logger = WandbLogger(name=cfg['NAME'], log_model=True, group = group, tags = tags)
    wandb_logger.experiment.config.update(cfg)

    wandb_logger.experiment.config['DATA_PARAM'] = data.params
    

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

    else:

        model = GraphTranslatorModule(num_nodes=data.params['n_nodes'],
                                node_feature_len=data.params['n_len'],
                                context_len=data.params['c_len'],
                                edge_importance=cfg['EDGE_IMPORTANCE'],
                                edge_dropout_prob = cfg['EDGE_DROPOUT_PROB'],
                                tn_loss_weight=cfg['TN_LOSS_WEIGHT'],
                                learned_time_periods=cfg['LEARNED_TIME_PERIODS'],
                                hidden_layer_size=cfg['HIDDEN_LAYER_SIZE'])
    
        if write_ckpt:
            ckpt_callback = ModelCheckpoint(dirpath=checkpoint_dir)
            trainer = Trainer(gpus = torch.cuda.device_count(), max_epochs=cfg['EPOCHS'], logger=wandb_logger, log_every_n_steps=5, callbacks=[ckpt_callback])

        else:
            trainer = Trainer(gpus = torch.cuda.device_count(), max_epochs=cfg['EPOCHS'], logger=wandb_logger, log_every_n_steps=5)

        wandb_logger.watch(model, log='gradients', log_freq=20)

        trainer.fit(model, data.get_train_loader())
        trainer.test(model, data.get_test_loader())

    if cfg['LEARNED_TIME_PERIODS'] : print('Time Periods learned (hrs)',model.period_in_days*24)

    evaluation_summary = evaluate_applications(model, data, cfg, output_dir, logger=wandb_logger.experiment)

    print('Outputs saved at ',output_dir)
    if INTERACTIVE:
        visualize_unconditional_datapoint(model, data.test_routines, data.node_classes, use_output_nodes=cfg['LEARN_NODES'])
        visualize_conditional_datapoint(model, data.get_single_example_test_loader(), data.node_classes, use_output_nodes=cfg['LEARN_NODES'])

    wandb.finish()



def run(data_dir, cfg = {}, baselines=False, ckpt_dir=None, read_ckpt=False, write_ckpt=False, tags=[]):
    
    if cfg['NAME'] is None:
        cfg['NAME'] = os.path.basename(data_dir)+'_trial'

    with open(os.path.join(data_dir, 'processed', 'common_data.json')) as f:
        cfg['DATA_INFO'] = json.load(f)['info']

    time_options = TimeEncodingOptions(cfg['DATA_INFO']['weeekend_days'] if 'weeekend_days' in cfg['DATA_INFO'].keys() else None)
    time_encoding = time_options(cfg['TIME_ENCODING'])

    data = RoutinesDataset(data_path=os.path.join(data_dir,'processed'), 
                           time_encoder=time_encoding, 
                           batch_size=cfg['BATCH_SIZE'],
                           only_seen_edges = cfg['ONLY_SEEN_EDGES'])
    
    if baselines:
        cf = get_cooccurence_frequency(data)
        spec = get_spectral_components(data, periods_mins=[float('inf'), 60*24, 60*24/2])
        # for baseline in [LastSeen(cf), StaticSemantic(cf), LastSeenAndStaticSemantic(cf), LastSeenButMostlyStaticSemantic(cf), Fremen(spec), FremenStateConditioned(spec, data.params['dt']), FremenStateConditionedFastDecay(spec, data.params['dt'])]:
        for baseline in [LastSeen(cf), StaticSemantic(cf), LastSeenAndStaticSemantic(cf), Fremen(spec), FremenStateConditioned(spec, data.params['dt'])]:
            output_dir = os.path.join('logs','baselines',baseline.__class__.__name__)
            wandb.init(name=baseline.__class__.__name__, dir=output_dir, group = os.path.basename(data_dir), tags=tags)
            cfg['NAME'] = wandb.run.name
            wandb.config.update(cfg)
            for routine in data.test:
                eval, details = baseline.step(data.test.collate_fn([routine]))
            # wandb.log(baseline.log())
            _ = evaluate_applications(baseline, data, cfg, output_dir)

            print('Outputs saved at ',output_dir)
            if INTERACTIVE:
                visualize_unconditional_datapoint(baseline, data.test_routines, data.node_classes, use_output_nodes=cfg['LEARN_NODES'])
                visualize_conditional_datapoint(baseline, data.get_single_example_test_loader(), data.node_classes, use_output_nodes=cfg['LEARN_NODES'])
            wandb.finish()
    else:
        run_model(data, group = os.path.basename(data_dir), checkpoint_dir=ckpt_dir, read_ckpt=read_ckpt, write_ckpt=write_ckpt, tags=tags)







if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model on routines.')
    parser.add_argument('--path', type=str, default='data/Persona0219/hard_worker', help='Path where the data lives. Must contain routines, info and classes json files.')
    parser.add_argument('--architecture_cfg', type=str, help='Name of config file.')
    parser.add_argument('--cfg', type=str, help='Name of config file.')
    parser.add_argument('--name', type=str, help='Name of run.')
    parser.add_argument('--tags', type=str, help='Tags for the run separated by a comma \',\'')
    parser.add_argument('--baselines', action='store_true')
    parser.add_argument('--ckpt_dir', type=str, help='Path to checkpoint file')
    parser.add_argument('--read_ckpt', action='store_true')
    parser.add_argument('--write_ckpt', action='store_true')

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

    tags = args.tags.split(',') if args.tags is not None else []

    run(data_dir=args.path, cfg=cfg, baselines=args.baselines, ckpt_dir=args.ckpt_dir, read_ckpt=args.read_ckpt, write_ckpt=args.write_ckpt, tags = tags)