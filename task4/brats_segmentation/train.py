import os
import time
import numpy as np
import torch
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from torch.nn import functional as F
from tqdm import tqdm
from dataloader import get_dataloaders
import wandb


def create_model(dataloader):
    sample = next(iter(dataloader))
    sample_image = sample["image"]
    sample_label = sample["label"]
    assert sample_image.shape[2:] == sample_label.shape[2:], "Shape mismatch"
    assert sample_label.shape[0] == sample_image.shape[0], "Batch size mismatch"

    model = UNet(
        spatial_dims=3,
        in_channels=sample_image.shape[1],
        out_channels=sample_label.shape[1],
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
    )
    print(model)
    return model


def train():
    # Initialize W&B
    wandb.init(
        project="brats_segmentation",
        config={
            "roi_size": [128, 128, 128],
            "batch_size": 16,
            "epochs": 50,
            "learning_rate": 0.01,
            "num_workers": 1,
            "split_dir": "./splits/split2",
            "data_dir": "/work/projects/ai_imaging_class/dataset",
            "early_stop_limit": 10,
            "save_path": "models",
        },
    )
    config = wandb.config

    # Load dataloaders
    train_loader, val_loader, _ = get_dataloaders(
        split_dir=config.split_dir,
        roi_size=config.roi_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    # Initialize model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(train_loader).to(device)

    loss_function = DiceLoss(to_onehot_y=False, softmax=False, include_background=False)
    # loss_function = BCEWithLogitsLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=1e-5
    )

    # Initialize metrics
    dice_metric = DiceMetric(
        # include_background=False, reduction="mean", get_not_nans=True
        include_background=False,
        reduction="none",
        get_not_nans=True,
    )
    # post_transform = Compose([
    #     Activations(softmax=True),
    #     AsDiscrete(argmax=True)
    # ])

    # Early stopping
    best_dice = 0.0
    early_stop_counter = 0
    os.makedirs(config.save_path, exist_ok=True)

    wandb.watch(model, log="all", log_freq=100)
    for epoch in range(config.epochs):
        start_time = time.time()
        model.train()
        epoch_loss = 0
        dice_metric.reset()

        for batch_idx, batch in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        ):
            inputs, labels = batch["image"].to(device), batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            softmax_outputs = F.softmax(outputs, dim=1)
            # print(softmax_outputs[0, :, 0, 0, 0], labels[0, :, 0, 0, 0])
            # print(outputs.shape, labels.shape)
            # exit()
            # loss = loss_function(outputs, labels)
            loss = loss_function(softmax_outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            dice_metric(y_pred=softmax_outputs, y=labels)
            wandb.log({"train_loss_step": loss.item()})

        # Aggregate and log training Dice scores
        print(f"{dice_metric=}")
        dice_scores, _ = dice_metric.aggregate()
        dice_scores = dice_scores.cpu().numpy()
        dice_metric.reset()

        dice_scores_arr = np.nanmean(np.array(dice_scores), axis=0)
        print(f"Epoch {epoch + 1}: Train Loss: {epoch_loss}, Dice: {dice_scores_arr}")
        wandb.log(
            {
                "train_loss": epoch_loss / len(train_loader),
                **{
                    f"train_dice_class_{i}": score
                    for i, score in enumerate(dice_scores_arr)
                },
            }
        )

        # Validation step
        model.eval()
        val_loss = 0.0
        dice_metric.reset()

        with torch.no_grad():
            for val_batch in val_loader:
                val_inputs, val_labels = val_batch["image"].to(device), val_batch[
                    "label"
                ].to(device)

                # Perform sliding window inference
                # val_outputs = sliding_window_inference(
                #     inputs=val_inputs,
                #     roi_size=config.roi_size,
                #     sw_batch_size=1,
                #     predictor=model,
                #     overlap=0.5,
                # )
                val_outputs = model(val_inputs)
                val_outputs = F.softmax(val_outputs, dim=1)

                # Post-process predictions
                # val_outputs = post_transform(val_outputs)

                # Ensure shape compatibility
                assert (
                    val_outputs.shape == val_labels.shape
                ), f"Shape mismatch: {val_outputs.shape} vs {val_labels.shape}"

                # Update Dice metric
                # val_outputs_decoupled = decollate_batch(val_outputs)
                # val_labels_decoupled = decollate_batch(val_labels)

                try:
                    # dice_metric(y_pred=val_outputs_decoupled, y=val_labels_decoupled)
                    dice_metric(y_pred=val_outputs, y=val_labels)
                except Exception as e:
                    print(f"[ERROR] Dice metric computation failed: {e}")
                    continue

                # Compute validation loss
                # loss = loss_function(val_outputs, val_labels)
                # val_loss += loss.item()

        # Aggregate and log validation metrics
        dice_scores, _ = dice_metric.aggregate()
        dice_scores = dice_scores.cpu().numpy()
        mean_dice = np.nanmean(dice_scores)

        wandb.log(
            {
                "val_loss": val_loss / len(val_loader),
                **{f"val_dice_class_{i}": score for i, score in enumerate(dice_scores)},
                "mean_dice": mean_dice,
            }
        )

        print(f"Mean Validation Dice: {mean_dice:.4f}")

        # Save best model
        if mean_dice > best_dice:
            best_dice = mean_dice
            early_stop_counter = 0
            torch.save(
                model.state_dict(), os.path.join(config.save_path, "best_model.pth")
            )
            print(f"Saved Best Model at Epoch {epoch + 1}: Mean Dice: {mean_dice:.4f}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= config.early_stop_limit:
                print("Early stopping triggered.")
                break

        wandb.log({"epoch_time": time.time() - start_time})


if __name__ == "__main__":
    train()
