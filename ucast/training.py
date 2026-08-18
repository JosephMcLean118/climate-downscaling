import torch
import os
from .utils import handle_fp16, fair_crps_loss, enable_inference_dropout, weighted_mae_loss, EMA, build_optimizer, LinearWarmupCosineAnnealingLR
from .muon import Muon, get_muon_momentum, muon_update, zeropower_via_newtonschulz5
import math

def train_stage1(model, dataloader_train, dataset_training, device, prev_domain="start",
                  max_epochs=100, lat_weights=None, ckpt_dir="./models", use_muon=True, domain="NZ", model_name="model"): 
    """"
    Deterministic pre-training stage.
    Input is normalised, passed through model, weighted-mae is calculated
    """

    history = []
    model = model.to(device)
    os.mkdir(f"models/{model_name}/{domain}/stage_one")

    # Load state dict from previous domain
    if prev_domain != "start":
        ckpt = torch.load(f"models/{model_name}/{prev_domain}/stage_two/downscale.pt", map_location="cpu", weights_only=False)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt["state_dict"]
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
        
    # Target normalization stats
    y_sum, y_sq_sum, y_count = 0.0, 0.0, 0
    for _, batch_y in dataloader_train:
        y_sum += batch_y.sum().item()
        y_sq_sum += (batch_y ** 2).sum().item()
        y_count += batch_y.numel()
    Y_MEAN = y_sum / y_count
    Y_STD = (y_sq_sum / y_count - Y_MEAN ** 2) ** 0.5
    print(f"Y_MEAN={Y_MEAN:.4f}, Y_STD={Y_STD:.4f}", flush=True)

    # Optimizers
    optimizers = build_optimizer(model, adamw_lr=3e-4, use_muon=use_muon, muon_lr=0.003, muon_wd=0.03, muon_momentum=0.95)
    adamw_opt, muon_opt = optimizers["adamw"], optimizers["muon"]

    steps_per_epoch = len(dataloader_train)
    total_steps = max_epochs * steps_per_epoch
    WARMUP_STEPS = 1500

    # Decide learning rate
    scheduler_adamw = LinearWarmupCosineAnnealingLR(adamw_opt, warmup_steps=WARMUP_STEPS, max_steps=total_steps)
    scheduler_muon = LinearWarmupCosineAnnealingLR(muon_opt, warmup_steps=WARMUP_STEPS, max_steps=total_steps) if muon_opt else None

    ema = EMA(model, decay=0.9999)
    scaler = torch.amp.GradScaler('cuda')

    last_ckpt_path = None
    global_step = 0
    mom_max = 0.95
    mom_min = mom_max - 0.1
    cooldown = WARMUP_STEPS // 10

    # Training loop
    for epoch in range(max_epochs):
        epoch_loss = 0.0

        for batch_x, batch_y in dataloader_train:
            # Load, reshape and normalise each batch
            batch_size = batch_x.shape[0]
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Reshape target so it's not a flattened 1D vector
            batch_y_reshaped = batch_y.view(batch_y.size(0), -1, 128, 128)
            batch_y_normed = (batch_y_reshaped - Y_MEAN) / Y_STD

            adamw_opt.zero_grad()
            if muon_opt is not None:
                muon_opt.zero_grad()
                current_momentum = get_muon_momentum(
                    global_step, total_steps,
                    muon_warmup_steps=WARMUP_STEPS, muon_cooldown_steps=cooldown,
                    momentum_min=mom_min, momentum_max=mom_max,
                )
                for group in muon_opt.param_groups:
                    group["momentum"] = current_momentum

            with torch.amp.autocast('cuda'):
                outputs = model(batch_x)
                loss_batch = weighted_mae_loss(outputs, batch_y_normed, lat_weights=lat_weights)

            # Handle scaling issues with fp16
            handle_fp16(scaler, loss_batch, model, adamw_opt, muon_opt, scheduler_adamw, scheduler_muon)
                
            ema.update(model)
            epoch_loss += batch_size * loss_batch.item()
            global_step += 1

        epoch_loss /= len(dataset_training)
        print(f"[Stage 1] Epoch {epoch+1}/{max_epochs} - Loss: {epoch_loss:.6f}", flush=True)
        history.append(epoch_loss)

        last_ckpt_path = f"{ckpt_dir}/{model_name}/{domain}/stage_one/downscale.pt"
        torch.save({"model": model.state_dict(), "ema": ema.shadow,
                   "y_mean": Y_MEAN, "y_std": Y_STD}, last_ckpt_path)

    return last_ckpt_path, history

def train_stage2(model, dataloader_train, dataset_training, device,
                        stage1_ckpt_path, lat_weights=None,
                        max_epochs=8, early_stop_epoch=8, 
                        num_training_ensemble_members=2, ckpt_dir="./models", use_muon=True, domain="NZ", model_name="model", stochasticity="dropout"):
    """
    Probabilistic fine tuning. We now enable MC dropout and generate an ensemble for predictions.
    CRPS will be used as optimiser. MC dropout is selected by default however learnable perturbations
    is also an option.
    """
    # Load in output from stage one
    ckpt = torch.load(stage1_ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt["state_dict"]

    history = []
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    os.mkdir(f"models/{model_name}/{domain}/stage_two")
    
    Y_MEAN = ckpt["y_mean"]
    Y_STD = ckpt["y_std"]

    # Enable dropout for mc ensembling
    if stochasticity == "dropout":
        enable_inference_dropout(model)

    optimizers = build_optimizer(model, adamw_lr=7e-5, use_muon=use_muon, muon_lr=0.007, muon_wd=0, muon_momentum=0.95)
    adamw_opt, muon_opt = optimizers["adamw"], optimizers["muon"]
    
    steps_per_epoch = len(dataloader_train)
    total_steps = max_epochs * steps_per_epoch  

    # Warmup steps to total steps ratio was 0.09375
    WARMUP_STEPS = total_steps * 0.09375

    scheduler_adamw = LinearWarmupCosineAnnealingLR(adamw_opt, warmup_steps=WARMUP_STEPS, max_steps=total_steps, warmup_start_lr=1e-8, eta_min=1e-8)
    scheduler_muon = LinearWarmupCosineAnnealingLR(muon_opt, warmup_steps=WARMUP_STEPS, max_steps=total_steps, warmup_start_lr=1e-8, eta_min=1e-8) if muon_opt else None

    ema = EMA(model, decay=0.9999)
    scaler = torch.amp.GradScaler('cuda')

    mom_max = 0.95
    mom_min = mom_max - 0.1
    cooldown = WARMUP_STEPS // 10
    last_ckpt_path = None
    global_step = 0   
    
    for epoch in range(early_stop_epoch): 
        epoch_loss = 0

        for idx, (batch_x, batch_y) in enumerate(dataloader_train):
            batch_size = batch_x.shape[0]
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Normalise to match stage one
            batch_y_reshaped = batch_y.view(batch_y.size(0), -1, 128, 128)
            batch_y_normed = (batch_y_reshaped - Y_MEAN) / Y_STD

            adamw_opt.zero_grad()
            if muon_opt is not None:
                muon_opt.zero_grad()
                current_momentum = get_muon_momentum(
                    global_step, total_steps,
                    muon_warmup_steps=WARMUP_STEPS, muon_cooldown_steps=cooldown,
                    momentum_min=mom_min, momentum_max=mom_max,
                )
                for group in muon_opt.param_groups:
                    group["momentum"] = current_momentum
            
            with torch.amp.autocast('cuda'):
                # M stochastic forward passes. Use either dropout or perturbations
                preds = torch.stack([model(batch_x, stochasticity) for _ in range(num_training_ensemble_members)], dim=0)  # (M, B, C, 128, 128)
                loss_batch = fair_crps_loss(preds, batch_y_normed, lat_weights=None)

            if torch.isnan(loss_batch) or torch.isinf(loss_batch):
                print(f"epoch {epoch}, batch {idx}: BAD loss")
                # dump grad norms of key params to see if it's exploding gradients, not activations
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
                print(f"grad norm at failure: {total_norm}")
                break

            # Handle scaling issues with fp16
            handle_fp16(scaler, loss_batch, model, adamw_opt, muon_opt, scheduler_adamw, scheduler_muon)
            ema.update(model)

            epoch_loss += batch_size * loss_batch.item()
            global_step += 1

        epoch_loss /= len(dataset_training)
        print(f"[Stage 2] Epoch {epoch+1}/{early_stop_epoch} (of {max_epochs} configured) - CRPS Loss: {epoch_loss:.6f}", flush=True)
        history.append(epoch_loss)
        last_ckpt_path = f"{ckpt_dir}/{model_name}/{domain}/stage_two/downscale.pt"
        torch.save({"model": model.state_dict(), "ema": ema.shadow,
                    "y_mean": Y_MEAN, "y_std": Y_STD},
                   last_ckpt_path)

    return (last_ckpt_path, domain, history)