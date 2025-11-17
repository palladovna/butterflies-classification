import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from src.data.dataloader import get_dataloaders
from src.utils.train_loop import train

from src.models.myalexnet import create_custom_alexnet
from src.models.resnet import create_resnet18
from src.models.vgg import create_vgg16
from src.models.efficientnet import create_efficientnet_b0


# ============================================================


def build_model(model_name, num_classes, pretrained, freeze_backbone):

    if model_name == "alexnet_custom":
        return create_custom_alexnet(num_classes=num_classes)

    elif model_name == "resnet18":
        return create_resnet18(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )

    elif model_name == "vgg16":
        return create_vgg16(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )

    elif model_name == "efficientnet_b0":
        return create_efficientnet_b0(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}")


# ============================================================


def main(config_path):

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    wandb.init(
        project="butterflies-classification",
        config=config,
        name=config["model"]["name"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(
        train_dir=config["data"]["train_dir"],
        val_dir=config["data"]["val_dir"],
        test_dir=config["data"]["test_dir"],
        batch_size=config["training"]["batch_size"],
        image_size=config["training"]["image_size"],
        num_workers=config["training"]["num_workers"]
    )

    model = build_model(
        model_name=config["model"]["name"],
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        freeze_backbone=config["model"]["freeze_backbone"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["learning_rate"]
    )

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=config["training"]["epochs"],
        save_path=config["training"]["save_path"]
    )

    print("Training completed!")


# ============================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML")
    args = parser.parse_args()

    main(args.config)
