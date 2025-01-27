import os
import torch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete
from monai.networks.nets import UNet

from train import create_model
from dataloader import visualize_image_and_label, get_dataloader


def test():
    # Configuration
    roi_size = [128, 128, 128]
    batch_size = 1
    num_workers = 4
    split_dir = "./splits/split1"
    checkpoint_path = "models/best_model.pth"  # Path to the best model

    # Ensure checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test dataset
    _, _, test_loader = get_dataloaders(
        split_dir=split_dir,
        roi_size=roi_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Load model
    model = UNet(
        spatial_dims=3,
        in_channels=4,  # Multi-modal inputs
        out_channels=4,  # Multi-class outputs
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
    ).to(device)

    # Load checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Metrics
    dice_metric = DiceMetric(
        include_background=False, reduction="mean_batch", get_not_nans=True
    )

    # Post-processing (apply softmax and argmax)
    post_trans = Compose([Activations(softmax=True), AsDiscrete(argmax=True)])

    # Testing loop
    print("Starting testing...")
    with torch.no_grad():
        dice_metric.reset()
        class_dice_scores = []
        for i, batch in enumerate(test_loader):
            print(f"Processing batch {i + 1}/{len(test_loader)}...")
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            outputs = sliding_window_inference(inputs, roi_size, 4, model)
            outputs = post_trans(outputs)

            # Calculate Dice score
            dice_metric(y_pred=outputs, y=labels)

        # Aggregate metrics
        dice_scores = dice_metric.aggregate()  # Tensor with per-class Dice scores
        mean_dice = dice_scores.mean().item()

        # Reset metric for clean logging
        dice_metric.reset()

    # Log results
    print("\nTesting Completed.")
    print(f"Mean Dice Score on Test Set: {mean_dice:.4f}")
    for i, score in enumerate(dice_scores):
        print(f"Class {i} Dice Score: {score.item():.4f}")


def visualize_result():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = get_dataloader(
        split_dir="./splits/split1",
        name="test",
        roi_size=[128, 128, 128],
        batch_size=1,
        num_workers=1,
    )
    model = create_model(test_loader).to(device)
    model.load_state_dict(torch.load("models_tmp/best_model.pth", weights_only=True))
    sample = next(iter(test_loader))
    inputs, labels = sample["image"].to(device), sample["label"].to(device)
    outputs = model(inputs)
    outputs = torch.argmax(outputs, dim=1)
    inputs = inputs.cpu().numpy().squeeze()
    labels = labels.cpu().numpy().squeeze()
    outputs = outputs.cpu().numpy().squeeze()
    # visualize_image_and_label(inputs, labels)
    visualize_image_and_label(inputs, outputs)


if __name__ == "__main__":
    # test()
    visualize_result()
