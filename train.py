import os
import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from ucast.utils import enable_inference_dropout 
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import indices, diagnostics
from ucast.training import train_stage1, train_stage2

class EmulationTrainingDataset(Dataset):
        def __init__(self, x_data, y_data):
            if not isinstance(x_data, torch.Tensor):
                x_data = torch.tensor(x_data)
            if not isinstance(y_data, torch.Tensor):
                y_data = torch.tensor(y_data)
            self.x_data, self.y_data = x_data, y_data
    
        def __len__(self):
            return len(self.x_data)
    
        def __getitem__(self, idx):
            x_sample, y_sample = self.x_data[idx, :], self.y_data[idx, :]
            return x_sample, y_sample
    
class EmulationTestDataset(Dataset):
    def __init__(self, x_data):
        if not isinstance(x_data, torch.Tensor):
            x_data = torch.tensor(x_data)
        self.x_data = x_data
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        return self.x_data[idx, :]

def plot_training_history(stage1_history, stage2_history, model_name, domain, save_dir="./results"):
    """
    Plot loss of model as it trains, with both training stages highlighted
    """
    os.makedirs(save_dir, exist_ok=True)

    stage1_epochs = list(range(1, len(stage1_history) + 1))
    stage2_epochs = list(range(len(stage1_history) + 1, len(stage1_history) + len(stage2_history) + 1))
    plt.figure(figsize=(10, 6))

    # Stage 1
    plt.plot(
        stage1_epochs,
        stage1_history,
        marker="o",
        label="Stage 1 — Weighted MAE",
        color="tab:blue",
    )

    # Stage 2
    plt.plot(
        stage2_epochs,
        stage2_history,
        marker="o",
        label="Stage 2 — CRPS",
        color="tab:orange",
    )

    # Highlight stages
    if stage1_epochs:
        plt.axvspan(
            0.5,
            len(stage1_history) + 0.5,
            color="tab:blue",
            alpha=0.08,
        )

    if stage2_epochs:
        plt.axvspan(
            len(stage1_history) + 0.5,
            len(stage1_history) + len(stage2_history) + 0.5,
            color="tab:orange",
            alpha=0.08,
        )

    # Stage boundary
    plt.axvline(
        len(stage1_history) + 0.5,
        color="black",
        linestyle="--",
        alpha=0.7,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} — {domain} Training")

    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    path = os.path.join(
        save_dir,
        f"{model_name}/training/training.png"
    )

    plt.savefig(path, dpi=200)
    plt.close()

    print(f"Training graph saved to: {path}")
    return path

def prepare_data(domain="NZ", training_experiment="ESD_pseudo_reality", var_target="tasmax"):
    """
    Download and prepare data based on domain

    Input: domain, training_experiment
    Output: 
    """

    DATA_PATH = '/Bench-data'
    MODELS_PATH = './models'
    
    os.makedirs(MODELS_PATH, exist_ok=True)
    
    # Set the period
    if training_experiment == 'ESD_pseudo_reality':
        period_training = '1961-1980'
    elif training_experiment == 'Emulator_hist_future':
        period_training = '1961-1980_2080-2099'
    else:
        raise ValueError('Provide a valid date')
    print(domain)
    # Set the GCM
    if domain == 'ALPS':
        gcm_name = 'CNRM-CM5'
    elif (domain == 'NZ') or (domain == 'SA'):
        gcm_name = 'ACCESS-CM2'
   
    predictor_filename = f'.{DATA_PATH}/{domain}/{domain}_domain/train/{training_experiment}/predictors/{gcm_name}_{period_training}.nc'
    predictor = xr.open_dataset(predictor_filename)
    
    
    predictand_filename = f'.{DATA_PATH}/{domain}/{domain}_domain/train/{training_experiment}/target/pr_tasmax_{gcm_name}_{period_training}.nc'
    predictand = xr.open_dataset(predictand_filename)
    predictand = predictand[[var_target]]
    
    if training_experiment == 'ESD_pseudo_reality':
        years_train = list(range(1961, 1975))
        years_test = list(range(1975, 1980+1))
    elif training_experiment == 'Emulator_hist_fut':
        years_train = list(range(1961, 1980+1)) + list(range(2080, 2090))
        years_test = list(range(2090, 2099+1))
    
    x_train = predictor.sel(time=np.isin(predictor['time'].dt.year, years_train))
    y_train = predictand.sel(time=np.isin(predictand['time'].dt.year, years_train))
    
    x_test = predictor.sel(time=np.isin(predictor['time'].dt.year, years_test))
    y_test = predictand.sel(time=np.isin(predictand['time'].dt.year, years_test))
    mean_train = x_train.mean('time')
    std_train = x_train.std('time')
    
    x_train_stand = (x_train - mean_train) / std_train
    x_test_stand = (x_test - mean_train) / std_train
    
    if domain == 'ALPS':
        spatial_dims = ('y', 'x')
    elif (domain == 'NZ') or (domain == 'SA'):
        spatial_dims = ('lat', 'lon')
    
    y_train_stack = y_train.stack(gridpoint=spatial_dims)
    y_test_stack = y_test.stack(gridpoint=spatial_dims)
    
    x_train_stand_array = torch.from_numpy(x_train_stand.to_array().transpose("time", "variable", "lat", "lon").values)
    y_train_stack_array = torch.from_numpy(y_train_stack.to_array()[0, :].values)
    
    x_test_stand_array = torch.from_numpy(x_test_stand.to_array().transpose("time", "variable", "lat", "lon").values)

    dataset_training = EmulationTrainingDataset(x_data=x_train_stand_array, y_data=y_train_stack_array)
    dataloader_train = DataLoader(dataset=dataset_training,
                              batch_size=32, shuffle=True)
    dataset_test = EmulationTestDataset(x_data=x_test_stand_array)
    test_dataloader = DataLoader(dataset=dataset_test,
                             batch_size=32, shuffle=False)
    

    return {"y_test": y_test, 
            "y_test_stack": y_test_stack, 
            "x_test_stand_array": x_test_stand_array,
            "dataset_training": dataset_training,
            "dataloader_train": dataloader_train,
            "test_dataloader": test_dataloader
           }

def train(model: nn.Module, var_target="tasmax", training_experiment='ESD_pseudo_reality', epochs=100, domain='NZ', model_name="model", stochasticity="dropout"):
    """
    Train given downscaling model and save state dictionary
    
    Input: downscaling model
    Output: state dictionary 
    """    

    if domain == "ALL":
        print("ALL")
        data_all_domains = {domain: prepare_data(domain, training_experiment, var_target) for domain in ["NZ", "SA", "ALPS"]}
    else:
        data_all_domains = {domain: prepare_data(domain, training_experiment, var_target) for domain in [domain]}
    
    directory_name = "results"

    try:
        os.mkdir(f"models/{model_name}")
    except FileExistsError:
        print("A model with this name already exists")
        
    is_first=True
    torch.cuda.reset_peak_memory_stats()
    
    for domain, data in data_all_domains.items():
        os.mkdir(f"models/{model_name}/{domain}")
        device = ('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        is_ucast_model = True
        ckpt_dir = "./models"
        if is_first:
            final_stage1_ckpt, stage1_history = train_stage1(
                model, data["dataloader_train"], data["dataset_training"], device,
                lat_weights=None, ckpt_dir=ckpt_dir, domain=domain, model_name=model_name
            )
            final_stage2ckpt, prev_domain, stage2_history = train_stage2(
                model, data["dataloader_train"], data["dataset_training"], device,
                stage1_ckpt_path=final_stage1_ckpt, lat_weights=None,
                ckpt_dir=ckpt_dir, domain=domain, model_name=model_name, stochasticity=stochasticity
            )
            is_first=False
        else:
            final_stage1_ckpt, stage1_history = train_stage1(
                model, data["dataloader_train"], data["dataset_training"], device,
                lat_weights=None, ckpt_dir=ckpt_dir, domain=domain, model_name=model_name, prev_domain=prev_domain
            )
            final_stage2_ckpt, prev_domain, stage2_history = train_stage2(
                model, data["dataloader_train"], data["dataset_training"], device,
                stage1_ckpt_path=final_stage1_ckpt, lat_weights=None,
                ckpt_dir=ckpt_dir, domain=domain, model_name=model_name, stochasticity=stochasticity
            )
        os.makedirs(f"results/{model_name}/training", exist_ok=True)
        plot_training_history(stage1_history, stage2_history, model_name=model_name, domain=domain)
        