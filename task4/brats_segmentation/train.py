import os
import time
import torch
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete, Compose
from dataloader import get_dataloaders
import wandb

def train():
    # Initialize W&B
    wandb.init(project="brats_segmentation", config={
        "roi_size": [128, 128, 128],
        "batch_size": 2,
        "epochs": 50,
        "learning_rate": 1e-4,
        "num_workers": 4,
        "split_dir": "./splits/split1",
        "data_dir": "/work/projects/ai_imaging_class/dataset",
        "early_stop_limit": 10,
        "save_path": "models"
    })
    config = wandb.config

    # Load dataloaders based on splits
    train_loader, val_loader, _ = get_dataloaders(
        split_dir=config.split_dir,
        roi_size=config.roi_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )

    # Initialize model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        spatial_dims=3,
        in_channels=4,  # Multi-modal inputs (T1, T1ce, T2, FLAIR)
        out_channels=4,  # Multi-class outputs (background, edema, tumor core, enhancing tumor)
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
    ).to(device)

    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Initialize metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=True)
    post_transform = AsDiscrete(argmax=True)

    # Variables to track the best validation Dice score and early stopping
    best_dice = 0.0
    early_stop_counter = 0
    os.makedirs(config.save_path, exist_ok=True)

    # Log the model architecture
    wandb.watch(model, log="all", log_freq=100)

    # Training loop
    for epoch in range(config.epochs):
        start_time = time.time()
        model.train()
        epoch_loss = 0.0
        train_class_dice_scores = []

        dice_metric.reset()
        for batch_idx, batch in enumerate(train_loader):
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()

            # Log gradient norms
            grad_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in model.parameters() if p.grad is not None]))
            wandb.log({"gradient_norm": grad_norm.item()})

            optimizer.step()
            epoch_loss += loss.item()

            # Update training Dice metric
            dice_metric(y_pred=outputs, y=labels)

            # Log step-wise metrics
            step = epoch * len(train_loader) + batch_idx
            wandb.log({"train_loss_step": loss.item(), "step": step})

        train_batch_dice_scores = dice_metric.aggregate()
        dice_metric.reset()

        # Log per-class Dice scores for training
        wandb.log({
            **{f"train_dice_class_{i}": score.item() for i, score in enumerate(train_batch_dice_scores)},
            "train_loss": epoch_loss / len(train_loader)
        })

        # Validation step
        model.eval()
        val_loss = 0.0
        class_dice_scores = []

        with torch.no_grad():
            dice_metric.reset()
            for val_batch in val_loader:
                val_inputs, val_labels = val_batch["image"].to(device), val_batch["label"].to(device)
                val_outputs = sliding_window_inference(val_inputs, config.roi_size, 4, model)
                val_outputs = post_transform(val_outputs)
                loss = loss_function(val_outputs, val_labels)
                val_loss += loss.item()
                dice_metric(y_pred=val_outputs, y=val_labels)

                # Log predictions every epoch
                if val_batch["image"].shape[0] > 0:
                    examples = []
                    for idx in range(min(2, val_batch["image"].shape[0])):  # Log up to 2 examples
                        examples.append(
                            wandb.Image(
                                val_inputs[idx, 0].cpu().numpy(),
                                caption=f"Prediction vs Ground Truth (epoch {epoch})",
                                masks={
                                    "prediction": {"mask_data": val_outputs[idx].cpu().argmax(dim=0).numpy()},
                                    "ground_truth": {"mask_data": val_labels[idx, 0].cpu().numpy()}
                                }
                            )
                        )
                    wandb.log({"examples": examples})

            dice_scores = dice_metric.aggregate()  # Tensor with per-class scores
            mean_dice = dice_scores.mean().item()
            dice_metric.reset()

            wandb.log({
                **{f"val_dice_class_{i}": score.item() for i, score in enumerate(dice_scores)},
                "val_mean_dice": mean_dice,
                "val_loss": val_loss / len(val_loader)
            })

        # Save the model if validation Dice improves
        if mean_dice > best_dice:
            best_dice = mean_dice
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(config.save_path, "best_model.pth"))
            print(f"Best model updated: Epoch {epoch + 1}, Val Dice: {mean_dice:.4f}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= config.early_stop_limit:
                print("Early stopping triggered.")
                break

        # Log epoch time
        epoch_time = time.time() - start_time
        wandb.log({"epoch_time": epoch_time})

        # Print progress
        print(f"Epoch {epoch + 1}/{config.epochs}")
        print(f"  Train Loss: {epoch_loss / len(train_loader):.4f}")
        print(f"  Train Dice Scores: {train_class_dice_scores}")
        print(f"  Val Dice Scores: {class_dice_scores}")
        print(f"  Mean Val Dice: {mean_dice:.4f}")
        print(f"  Val Loss: {val_loss / len(val_loader):.4f}")
        print(f"  Epoch Time: {epoch_time:.2f}s")


if __name__ == "__main__":
    train()
