import os
import xarray as xr
import numpy as np
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from ucast.utils import enable_inference_dropout 
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import indices, diagnostics
import time
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


def evaluate_model(model: nn.Module, state_path: str, var_target: str, training_experiment: str, is_probabilistic: bool, domain="NZ", model_name="model", stochasticity="dropout", attention="dense", epoch_num=None):
    torch.manual_seed(42)
    np.random.seed(42)
    """
    Evaluate models peformance across all metrics provided from diagnostics.py
    
    Input: Model and dataset used to train, validate and evaluate it.
    Output: CSV file containing report of models performance.    
    """

    if domain == "ALL":
        data_all_domains = {domain: prepare_data(domain, training_experiment, var_target) for domain in ["ALPS", "NZ", "SA"]}
    else:
        data_all_domains = {domain: prepare_data(domain, training_experiment, var_target) for domain in [domain]}
    
    directory_name = "results"
    cordex_rows = []
    model_rows = []
    extra_epochs = []
    if var_target == "tasmax":
        domain_results = {
            "RMSE": [], 
            "TXx": [],
            "RALSD": [],
            "PSS": [],
            "IAV": [],
            "P98_Bias": [],
            "Average Inference Time (ms)": [],
            "Inference Memory Usage (GB)": [],
            "clim_mean": []
            }
    else:
        domain_results = {
            "RMSE": [], 
            "SDII": [],
            "Rx1day": [],
            "RALSD": [],
            "LHD": [],
            "P98_Bias": [],
            "Average Inference Time (ms)": [],
            "Inference Memory Usage (GB)": [],
            "clim_mean": []
            }   
    
    for domain, data in data_all_domains.items():        
        # Generate models predictions on test dataset
        torch.cuda.reset_peak_memory_stats()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

        # Test domain on corresponding model for domain         # CHANGE TO STAGE TWO
        ckpt = torch.load(f"models/{model_name}/{domain}/stage_one/downscale.pt", map_location="cpu", weights_only=False)
        print(f"Domain: {domain}, State Path: models/{model_name}/{domain}/stage_one/downscale.pt")
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

        #Enable mc dropout for inference
        #enable_inference_dropout(model)  
    
        N_ENSEMBLE_MEMBERS = 10  
        all_member_predictions = []

        # Warmup GPU for timing
        with torch.no_grad():
            for batch_x in test_dataloader:
                batch_x = batch_x.to(next(model.parameters()).device)
        
                with torch.amp.autocast("cuda"):
                    _ = model(batch_x, stochasticity=stochasticity)
        
                break
        
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        predictions = []
        with torch.no_grad():
            for batch_x in test_dataloader:
                batch_x = batch_x.to(next(model.parameters()).device)
                with torch.amp.autocast('cuda'):
                    outputs = model(batch_x, stochasticity=stochasticity)  
                outputs_denormed = outputs.float() * Y_STD + Y_MEAN  
                predictions.append(outputs_denormed.cpu().numpy())

        # Ensemble mean
        predictions = np.concatenate(predictions, axis=0)
        predictions = predictions.reshape(predictions.shape[0], -1)
            
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # Time per 10-member ensemble = time for all ensembles / total data (69 batches of 32 pieces of data)
        elapsed = (elapsed / (69*32))*1000 #ms
    
        y_pred_stack = y_test_stack.copy(deep=True)
        y_pred_stack[var_target].values = predictions
        y_pred = y_pred_stack.unstack()
        
    
        # Obtain diagnostic metrics
        peak_mb = torch.cuda.max_memory_allocated() / 1024**3
        rmse = diagnostics.rmse(x0=y_test, x1=y_pred, var=var_target, dim='time')
        mean_rmse = rmse[var_target].mean().values.item()
        num_params = sum(p.numel() for p in model.parameters())
    
        #bias_index = diagnostics.bias_index(x0=y_test, x1=y_pred,var=var_target)
        bias_mean = diagnostics.bias_index(x0=y_test, x1=y_pred,index_fn=indices.mean,var=var_target)
        bias_p98 = diagnostics.bias_index(x0=y_test, x1=y_pred,index_fn=indices.quantile,q=0.98,var=var_target)
        
        psd_target, psd_pred = diagnostics.psd(x0=y_test, x1=y_pred,var=var_target)
        psd_difference = psd_target - psd_pred
        
        ralsd = diagnostics.ralsd(psd_target, psd_pred)

        #compute clim mean
        y_test_clim = y_test.mean("time")
        y_pred_clim = y_pred.mean("time")

        # RMSE of climatological mean TXx
        if var_target == "tasmax":
            txx_pred = indices.txx(x=y_pred, var=var_target)
            txx_true = indices.txx(x=y_test, var=var_target)
            if domain != "ALPS":
                txx_rmse = diagnostics.rmse(x0=txx_true, x1=txx_pred, var="tasmax", dim=("lat", "lon"))
                iav = (indices.interannual_var(x=y_pred, var=var_target))["tasmax"].mean(dim=("lat", "lon")).item()
                clim_rmse = diagnostics.rmse(x0=y_test_clim, x1=y_pred_clim, var=var_target, dim=("lat", "lon"))
            else:
                txx_rmse = diagnostics.rmse(x0=txx_true, x1=txx_pred, var="tasmax", dim=("x", "y"))
                iav = (indices.interannual_var(x=y_pred, var=var_target))["tasmax"].mean(dim=("x", "y")).item()
                clim_rmse = diagnostics.rmse(x0=y_test_clim, x1=y_pred_clim, var=var_target, dim=("x", "y"))
    
            mean_txx = txx_rmse["tasmax"].item()
            clim_rmse = clim_rmse["tasmax"].item()
            pss = indices.pss(x0=y_test, x1=y_pred, var=var_target)
        

        if var_target == "pr":
            rx1day_pred = indices.rx1day(x=y_pred, var=var_target)
            rx1day_true = indices.rx1day(x=y_test, var=var_target)
            if domain != "ALPS":
                rx1day_rmse = diagnostics.rmse(x0=rx1day_true, x1=rx1day_pred, var="pr", dim=("lat", "lon"))
                sdii = (indices.sdii(x=y_pred, var=var_target))["pr"].mean(dim=("lat", "lon")).item()
                clim_rmse = diagnostics.rmse(x0=y_test_clim, x1=y_pred_clim, var=var_target, dim=("lat", "lon"))
            else:
                rx1day_rmse = diagnostics.rmse(x0=rx1day_true, x1=rx1day_pred, var="pr", dim=("x", "y"))
                sdii = (indices.sdii(x=y_pred, var=var_target))["pr"].mean(dim=("x", "y")).item()
                clim_rmse = diagnostics.rmse(x0=y_test_clim, x1=y_pred_clim, var=var_target, dim=("x", "y"))

            mean_rx1day = rx1day_rmse["pr"].item()
            lhd = diagnostics.wasserstein_distance(x0=y_test, x1=y_pred, var=var_target)
            clim_rmse = clim_rmse["pr"].item()
            
            
        peak_mb = torch.cuda.max_memory_allocated() / 1024**3

        try:
            ralsd_value = float(ralsd)
        except:
            ralsd_value = float(np.asarray(ralsd).mean())
        
        try:
            os.makedirs(f"results/{model_name}det/{domain}/data")
            os.makedirs(f"results/{model_name}det/{domain}/plots")
        except FileExistsError:
            print(f"Directory 'results/{model_name}' already exists.")

        
        rmse.to_netcdf(f"results/{model_name}/{domain}/data/rmse.nc")
        bias_p98.to_netcdf(f"results/{model_name}/{domain}/data/bias_p98.nc")
        psd_difference.to_netcdf(f"results/{model_name}/{domain}/data/psd_difference.nc")
        
        
        plot_data_map(data=rmse, var_name=var_target, domain=domain, vmin=0, vmax=5, fig_title='RMSE', cmap='Reds', map_name="rmse_map", model_name=model_name)
        plot_data_map(data=bias_p98, var_name=var_target, domain=domain, vmin=-2, vmax=2, fig_title='Bias 98th Percentile', cmap='RdBu_r', map_name="bias_98p_map", model_name=model_name)
        plot_psd(psd_target=psd_target, psd_pred=psd_pred, map_name="psd_map", model_name=model_name, domain=domain)
        
        if var_target == "tasmax":
            domain_results["TXx"].append(mean_txx)
            domain_results["PSS"].append(pss)
            domain_results["IAV"].append(iav)

        else:
            domain_results["SDII"].append(sdii)
            domain_results["Rx1day"].append(mean_rx1day)
            domain_results["LHD"].append(lhd)

        domain_results["RMSE"].append(mean_rmse)
        domain_results["RALSD"].append(ralsd_value)
        domain_results["Average Inference Time (ms)"].append(elapsed)
        domain_results["Inference Memory Usage (GB)"].append(peak_mb)
        domain_results["P98_Bias"].append(bias_p98)
        domain_results["clim_mean"].append(clim_rmse)
        
        
    print(f"RMSE: {mean_rmse}")
    clim_mean = round(np.mean(domain_results["clim_mean"]), 4)
    print(f"model: {model_name}")
    print(f"Clim Mean: {clim_mean}")
    
    if var_target == "tasmax":
        csv_path = f"results/model_performance.csv"
        cordex_rows.append({
            "model": epoch_num,
            "paramaters": num_params,
            "stochasticity": (stochasticity if is_probabilistic else "deterministic"),
            "attention" : attention,
            "RMSE": round(np.mean(domain_results["RMSE"]), 4),
            "TXx": round(np.mean(domain_results["TXx"]), 4),
            "RALSD": round(np.mean(domain_results["RALSD"]), 4),
            "PSS": round(np.mean(domain_results["PSS"]), 4),
            "IAV": round(np.mean(domain_results["IAV"]), 4),
            "Average Inference Time (ms)": round(np.mean(domain_results["Average Inference Time (ms)"]), 4),
            "Inference Memory Usage (GB)": round(peak_mb, 4),
            "P98_Bias": round(float(np.mean([bias[var_target].mean().values for bias in domain_results["P98_Bias"]])),4)        
        })
    else:
        csv_path = f"results/model_performance_pr.csv"
        cordex_rows.append({
            "model": model_name,
            "paramaters": num_params,
            "stochasticity": (stochasticity if is_probabilistic else "deterministic"),
            "attention" : attention,
            "RMSE": round(np.mean(domain_results["RMSE"]), 4),
            "SDII": round(np.mean(domain_results["SDII"]), 4),
            "Rx1day": round(np.mean(domain_results["Rx1day"]), 4),
            "RALSD": round(np.mean(domain_results["RALSD"]), 4),
            "LHD": round(np.mean(domain_results["LHD"]), 4),
            "Average Inference Time (ms)": round(np.mean(domain_results["Average Inference Time (ms)"]), 4),
            "Inference Memory Usage (GB)": round(peak_mb, 4),
            "P98_Bias": round(float(np.mean([bias[var_target].mean().values for bias in domain_results["P98_Bias"]])),4)        
        })


    new_results = pd.DataFrame(cordex_rows)

    if os.path.exists(csv_path):
        old_results = pd.read_csv(csv_path)
        new_results = pd.concat(
            [old_results, new_results],
            ignore_index=True
        )
    
    new_results.to_csv(csv_path, index=False)
    
    print(f"Results written to {csv_path}")
    new_results.to_csv(csv_path, index=False)

    print("Wrote CSV:", os.path.abspath(csv_path))
    print("Rows written:", len(new_results))
    
    # Verify the file actually contains them
    check = pd.read_csv(csv_path)
    check = pd.read_csv(csv_path)

    print("CSV path:", os.path.abspath(csv_path))
    print("DataFrame rows:", len(new_results))
    print("CSV rows:", len(check))
    print(check.to_string(index=False))
    print("Rows actually in file:", len(check))
    print(check.tail())
    print("--- END CSV DEBUG ---\n")
    return 
