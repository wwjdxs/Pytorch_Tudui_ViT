"""训练与测试共用模型工厂，确保权重、分类头和预处理匹配。"""
import torch
from torch import nn
from torchvision import models, transforms
from ViTmodel import cifar10
from ViTdata import get_transform as custom_transform


def build_model(model_type, config, pretrained=False):
    if model_type == "custom":
        return cifar10(**config)
    if model_type != "torchvision_vit_b_16":
        raise ValueError(f"未知模型类型：{model_type}")
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vit_b_16(weights=weights)
    model.heads.head = nn.Linear(model.heads.head.in_features, config["num_classes"])
    return model


def get_transform(training=False, model_type="custom", normalize=True):
    if model_type == "custom":
        return custom_transform(training=training, normalize=normalize)
    if model_type != "torchvision_vit_b_16":
        raise ValueError(f"未知模型类型：{model_type}")
    # 显式固定 V1 的推理配方；创建变换不会下载权重。
    evaluation = models.ViT_B_16_Weights.IMAGENET1K_V1.transforms()
    if not training:
        return evaluation
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        evaluation,
    ])


def make_optimizer(model, lr, pretrained=False):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        no_decay = (parameter.ndim == 1 or any(token in name for token in
                    ("pos_embed", "pos_embedding", "cls_token", "class_token")))
        head = pretrained and name.startswith("heads.")
        key = (head, no_decay)
        groups.setdefault(key, []).append(parameter)
    return torch.optim.AdamW([
        dict(params=params, lr=lr * (10 if head else 1),
             weight_decay=0.0 if no_decay else 0.05)
        for (head, no_decay), params in groups.items()
    ], lr=lr, foreach=False)
