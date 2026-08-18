import os
import xarray as xr
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from ucast.utils import enable_inference_dropout 
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import indices, diagnostics
from train import prepare_data

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



def plot_data_map(data, var_name, domain, vmin, vmax, map_name, model_name,
                  fig_title='', figsize=(5,5), cmap='viridis'):
    
    central_longitude = 180 if domain == 'NZ' else 0 if domain == 'ALPS' else None
    
    plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=central_longitude))

    if (domain == 'NZ') or (domain == 'SA'):
        data[var_name].plot(ax=ax, transform=ccrs.PlateCarree(),
                            vmin=vmin, vmax=vmax,
                            cmap=cmap)
    elif domain == 'ALPS':
        cs = ax.pcolormesh(data[var_name]['lon'], data[var_name]['lat'],
                           data[var_name],
                           transform=ccrs.PlateCarree(),
                           vmin=vmin, vmax=vmax,
                           cmap=cmap)
    
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    plt.title(fig_title)
    plt.savefig(f"results/{model_name}/{domain}/plots/{map_name}.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_psd(psd_target, psd_pred, map_name,model_name, domain):
    plt.loglog(psd_target.wavenumber, psd_target, label="Target")
    plt.loglog(psd_pred.wavenumber, psd_pred, label="Prediction")
    plt.xlabel("Wavenumber")
    plt.title("Power Spectral Density")
    plt.legend()
    plt.savefig(f"results/{model_name}/{domain}/plots/{map_name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_model(model: nn.Module, state_path: str, var_target: str, training_experiment: str, is_probabilistic: bool, domain="NZ", model_name="model", stochasticity="dropout", attention="dense"):
    """
    Evaluate models peformance across all metrics provided from diagnostics.py
    
    Input: Model and dataset used to train, validate and evaluate it.
    Output: CSV file containing report of models performance.    
    """

    if domain == "ALL":
        data_all_domains = {domain: prepare_data(domain, training_experiment, var_target) for domain in ["NZ", "SA", "ALPS"]}
    else:
        data_all_domains = {domain: prepare_data(domain, training_experiment, var_target) for domain in [domain]}
    
    directory_name = "results"
    cordex_rows = []
    model_rows = []
    
    for domain, data in data_all_domains.items():
        # Generate models predictions on test dataset
        torch.cuda.reset_peak_memory_stats()
        if is_probabilistic:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
            
            ckpt = torch.load(state_path, map_location="cpu", weights_only=False)
            state_dict = ckpt["ema"] if "ema" in ckpt else ckpt["model"]
            
            model.load_state_dict(state_dict)
            model = model.to(device)   
        
            Y_MEAN = ckpt["y_mean"]
            Y_STD = ckpt["y_std"]
            y_test = data["y_test"]
            y_test_stack = data["y_test_stack"]
            x_test_stand_array = data["x_test_stand_array"]
            test_dataloader = data["test_dataloader"]
        
            model.eval()
    
            # Enable mc dropout for inference
            enable_inference_dropout(model)  
        
            N_ENSEMBLE_MEMBERS = 10  
            all_member_predictions = []
        
            with torch.no_grad():
                for member_idx in range(N_ENSEMBLE_MEMBERS):
                    member_predictions = []
                    for batch_x in test_dataloader:
                        batch_x = batch_x.to(next(model.parameters()).device)
                        with torch.amp.autocast('cuda'):
                            outputs = model(batch_x)  
                        outputs_denormed = outputs.float() * Y_STD + Y_MEAN  
                        member_predictions.append(outputs_denormed.cpu().numpy())
                    member_predictions = np.concatenate(member_predictions, axis=0)  
                    all_member_predictions.append(member_predictions)
        
            all_member_predictions = np.stack(all_member_predictions, axis=0)  
        
            # Ensemble mean
            predictions = all_member_predictions.mean(axis=0)  
            predictions = predictions.reshape(predictions.shape[0], -1)
    
        else:
            state_dict = torch.load("./models/DeepESD_U-Cast_ESD_pseudo_reality_NZ_tasmax.pt", weights_only=False)
            model.load_state_dict(state_dict)
            model.eval()
            predictions = []
            with torch.no_grad():
                for batch_x in test_dataloader:
                    batch_x = batch_x.to(next(model.parameters()).device)
                    outputs = model(batch_x)
                    predictions.append(outputs.cpu().numpy())
            predictions = np.concatenate(predictions, axis=0)
            predictions = predictions.reshape(predictions.shape[0], -1)
        
        y_pred_stack = y_test_stack.copy(deep=True)
        y_pred_stack[var_target].values = predictions
        y_pred = y_pred_stack.unstack()
    
    
        # Obtain diagnostic metrics
    
        rmse = diagnostics.rmse(x0=y_test, x1=y_pred, var=var_target, dim='time')
        mean_rmse = rmse[var_target].mean().values.item()
        num_params = sum(p.numel() for p in model.parameters())
    
        #bias_index = diagnostics.bias_index(x0=y_test, x1=y_pred,var=var_target)
        bias_mean = diagnostics.bias_index(x0=y_test, x1=y_pred,index_fn=indices.mean,var=var_target)
        bias_p98 = diagnostics.bias_index(x0=y_test, x1=y_pred,index_fn=indices.quantile,q=0.98,var=var_target)
    
        #bias_multivariable_correlation = diagnostics.bias_multivariable_correlation(y_test, y_pred, var_x="tasmax", var_y="pr")
    
        #ratio_mean = diagnostics.ratio_index(y_test, y_pred, index_fn=indices._mean)
        
        psd_target, psd_pred = diagnostics.psd(x0=y_test, x1=y_pred,var=var_target)
        psd_difference = psd_target - psd_pred
        
        ralsd = diagnostics.ralsd(psd_target, psd_pred)
        txx = indices.txx(x=y_pred, var=var_target)
        mean_txx = txx["tasmax"].mean(dim=("lat", "lon")).item()
        iav = (indices.interannual_var(x=y_pred, var=var_target))["tasmax"].mean(dim=("lat", "lon")).item()
        pss = indices.pss(x0=y_test, x1=y_pred, var=var_target)
        #wasserstein_difference = diagnostics.wasserstein_differnce()
        peak_mb = torch.cuda.max_memory_allocated() / 1024**3

        try:
            ralsd_value = float(ralsd)
        except:
            ralsd_value = float(np.asarray(ralsd).mean())
        
        cordex_rows.append({
            "model": model_name,
            "domain": domain,
            "paramaters": num_params,
            "stochasticity": (stochasticity if is_probabilistic else "deterministic"),
            "attention" : attention,
            "RMSE": round(mean_rmse, 4),
            "TXx": round(mean_txx, 4),
            "RALSD": round(ralsd_value, 4),
            "PSS": round(pss, 4),
            "IAV": round(iav, 4),
            "Inference Memory Usage (GB)": peak_mb,
            "Mean_Bias": round(float(bias_mean[var_target].mean().values), 4),
            "P98_Bias": round(float(bias_p98[var_target].mean().values), 4)
        })



        
        try:
            os.mkdir(f"results/{model_name}/{domain}")
            os.mkdir(f"results/{model_name}/{domain}/data")
            os.mkdir(f"results/{model_name}/{domain}/plots")
        except FileExistsError:
            print(f"Directory 'results/{model_name}' already exists.")
    
        rmse.to_netcdf(f"results/{model_name}/{domain}/data/rmse.nc")
        bias_mean.to_netcdf(f"results/{model_name}/{domain}/data/bias_mean.nc")
        bias_p98.to_netcdf(f"results/{model_name}/{domain}/data/bias_p98.nc")
        psd_difference.to_netcdf(f"results/{model_name}/{domain}/data/psd_difference.nc")
    
        plot_data_map(data=rmse, var_name=var_target, domain=domain, vmin=0, vmax=5, fig_title='RMSE', cmap='Reds', map_name="rmse_map", model_name=model_name)
        plot_data_map(data=bias_mean, var_name=var_target, domain=domain, vmin=-2, vmax=2, fig_title='Bias Mean', cmap='RdBu_r', map_name="bias_mean_map", model_name=model_name)
        plot_data_map(data=bias_p98, var_name=var_target, domain=domain, vmin=-2, vmax=2, fig_title='Bias 98th Percentile', cmap='RdBu_r', map_name="bias_98p_map", model_name=model_name)
        plot_psd(psd_target=psd_target, psd_pred=psd_pred, map_name="psd_map", model_name=model_name, domain=domain)

    print(f"RMSE: {mean_rmse}")
    csv_path = f"results/model_performance.csv"
    new_results = pd.DataFrame(cordex_rows)

    
    
    if os.path.exists(csv_path):
        old_results = pd.read_csv(csv_path)
        new_results = pd.concat(
            [old_results, new_results],
            ignore_index=True
        )
    
    new_results.to_csv(csv_path, index=False)
    
    print(f"Results written to {csv_path}")
    return 
    