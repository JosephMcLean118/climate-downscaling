import argparse
import os
import xarray as xr 
import numpy as np 
import torch 
from torch.utils.data import Dataset, DataLoader 
import torch.nn as nn 
import torch.nn.functional as F 
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs 
import cartopy.feature as cfeature 
import math 
from einops import rearrange 
import sys 
sys.path.insert(0, "../u-cast") 
import sys 
sys.path.append('../evaluation') 
import indices, diagnostics 
from ucast.ucast import UNetBlock, DhariwalUNet 
from ucast import training 
from train import train, prepare_data



def main():
    """
    Entry point of program. Any model type can be created here.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--domain", type=str, default="ALL")
    parser.add_argument("--stochasticity", type=str, default="dropout")
    parser.add_argument("--var_target", type=str, default="tasmax")
    parser.add_argument("--blocksparse_attention", action="store_true", default=False)
    parser.add_argument("--training_experiment", type=str, default="ESD_pseudo_reality")
    parser.add_argument("--base", type=str, default="convolutional")
    parser.add_argument("--stages", type=str, default="ALL")
    args = parser.parse_args()

    train(
        block_sparse_attention = args.blocksparse_attention,
        base=args.base,
        model_name=args.model_name,
        domain=args.domain,
        stochasticity=args.stochasticity,
        var_target=args.var_target,
        training_experiment=args.training_experiment,
        stage=args.stages
        
    )

if __name__ == "__main__":
    main()