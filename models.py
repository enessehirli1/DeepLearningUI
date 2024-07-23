import torch
import torchvision.models
import torch.nn as nn

def createEffNetB7(num_classes: int, seed: int = 42):
    effnetb7_weights = torchvision.models.EfficientNet_B7_Weights.DEFAULT
    effnetb7_transforms = effnetb7_weights.transforms()
    model = torchvision.models.efficientnet_b7(weights=effnetb7_weights)

    for param in model.parameters():
        param.requires_grad = False

    torch.manual_seed(seed)
    model.classifier = nn.Sequential(
        nn.Linear(in_features=2560, out_features=num_classes, bias=True)
    )

    return model, effnetb7_transforms

def createEffNetB2(num_classes: int, seed: int = 42):
    effnetb2_weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    effnetb2_transforms = effnetb2_weights.transforms()
    model = torchvision.models.efficientnet_b2(weights=effnetb2_weights)

    for param in model.parameters():
        param.requires_grad = False

    torch.manual_seed(seed)
    model.classifier = nn.Sequential(
        nn.Linear(in_features=1408, out_features=num_classes, bias=True)
    )

    return model, effnetb2_transforms

def createViTB16(num_classes: int, seed: int = 42):
    vitb16_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    vitb16_transforms = vitb16_weights.transforms()
    model = torchvision.models.vit_b_16(weights=vitb16_weights)

    for param in model.parameters():
        param.requires_grad = False

    torch.manual_seed(seed)
    model.heads = nn.Sequential(
        nn.Linear(in_features=768, out_features=num_classes, bias=True)
    )

    return model, vitb16_transforms

def createResNet50(num_classes: int, seed: int = 42):
    resnet50_weights = torchvision.models.ResNet50_Weights.DEFAULT
    resnet50_transforms = resnet50_weights.transforms()
    model = torchvision.models.resnet50(weights=resnet50_weights)

    for param in model.parameters():
        param.requires_grad = False

    torch.manual_seed(seed)
    model.fc = nn.Sequential(
        nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    )

    return model, resnet50_transforms
