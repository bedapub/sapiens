'''Joint training of a mention detection and entity linking component'''
import argparse
import json
import numpy as np
import torch
import tomli
from os import mkdir
from dataclasses import dataclass
from os.path import isfile, isdir, join
from typing import List, Callable
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR
from torch.utils.tensorboard.writer import SummaryWriter
from sapiens.cnn import ResCNN, ResCNNConfig
from sapiens.utils import Dotdict, EarlyStop, RetrievalEvaluator


# --- Train & validation steps -----------------------------


def train() -> torch.Tensor:
    '''Train step 
    ---
    returns training loss
    '''
    loss = 
    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss


def validate() -> torch.Tensor:
    '''Validation step
    ---
    returns validation loss
    '''
    model.eval()
    with torch.no_grad():

    return val_loss


# --- Utility classes -------------------------------------


def verify_args(args, config):
    '''verifies logging and checkpoint directories'''
    # assertions
    assert config.TRAIN.optim in ["Adam", "SGD"]

    # logging directory for tensorboard
    i = 0
    if not isdir(args.logdir):
        mkdir(args.logdir)
        mkdir(join(args.logdir, f"run{str(i)}"))
    else:
        while isdir(join(args.logdir,f"run{str(i)}")):
            i += 1
        mkdir(join(args.logdir, f"run{str(i)}"))

    # save training config in logging directory
    tc = config.TRAIN
    with open(join(args.logdir, f"run{str(i)}", "config.info"), "w") as f:
        for key,val in tc.items():
            f.write(f"{key}: {val}\n")

    # checkpoint directory
    j = 0
    if not isdir(args.checkpoint):
        mkdir(args.checkpoint)
        mkdir(join(args.checkpoint, f"run{str(j)}"))
    else:
        while isdir(join(args.checkpoint,f"run{str(j)}")):
            j += 1
        mkdir(join(args.checkpoint, f"run{str(j)}"))

    # define vars
    logdir = join(args.logdir, f"run{str(i)}") 
    checkpointdir = join(args.checkpoint, f"run{str(j)}")

    return logdir, checkpointdir


def print_config(config: Dotdict):
    '''Prints configuration parameters'''
    print("[DATA CONFIG]")
    for key,val in config.DATA.items():
        print(f"{key} = {val}", flush=True)
    print("")
    print("[TRAIN CONFIG]")
    for key,val in config.TRAIN.items():
        print(f"{key} = {val}", flush=True)
    print("")
    print("[MODEL CONFIG]")
    for key,val in config.MODEL.items():
        print(f"{key} = {val}", flush=True)


def save_model_checkpoint(
    epoch, model, optimizer, loss, cp_path
    ):
    '''Saves a checkpoint of model'''
    model_id = f"checkpoint_e{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, f"{join(cp_path, model_id)}")


# --- Main ------------------------------------------------


def main(config: Dotdict, cp_path: str, logdir: str, verbose:bool):
    '''Trains ResCNN model'''
    #--- SETUP TRAIN --------------------------------------

    # config
    if verbose: print_config(config)
    dc = config.DATA
    tc = config.TRAIN
    mc = config.MODEL

    # load train, val
    train = Data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configpath",
        default="configs/train_config.toml",
        help="toml config for model training"
    )
    parser.add_argument(
        "--checkpoint",
        default="resources/checkpoints",
        help="path to dir where model checkpoints are saved"
    )
    parser.add_argument(
        "--logdir",
        default="resources/logdir",
        help="path to dir where training logs are saved"
    )
    parser.add_argument(
        "--verbose",
        default=True
    )
    args = parser.parse_args()
    config = tomli.load(open(args.configpath, "rb"))
    config = Dotdict(config)
    logdir, checkpointdir = verify_args(args, config)
    main(config, checkpointdir, logdir, args.verbose)
