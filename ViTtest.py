import os
import torch
import torchvision
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from ViTmodel import cifar10
from ViTdata import get_transform


def main():
    # 默认测试新版最佳权重；若改为旧models_vit_gpu/vit9.pth，会自动使用8头、无标准化。
    current_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.environ.get("VIT_CHECKPOINT", os.path.join(current_path, "models_vit_optimized", "best.pth"))
    data_path = os.environ.get("CIFAR10_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    img_path = os.environ.get("VIT_IMAGE", "")  # 留空测试整个测试集，填写路径做单图预测。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    # 新版权重带少量配置，旧版只有state_dict；两者预处理和头数不同，不能混用。
    is_new = "model_state" in checkpoint
    model = cifar10(**checkpoint["model_config"]) if is_new else cifar10()
    model.load_state_dict(checkpoint["model_state"] if is_new else checkpoint)
    model = model.to(device).eval()
    transform = get_transform(normalize=checkpoint.get("normalize", False) if is_new else False)

    if img_path:
        with Image.open(img_path) as image:
            image = image.convert("RGB").resize((32, 32))
            image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            index = model(image).argmax(1).item()
        classes = ("airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck")
        print("预测类别：{} ({})".format(classes[index], index))
    else:
        test_data = torchvision.datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        total_loss, correct = 0.0, 0
        with torch.no_grad():
            for images, targets in DataLoader(test_data, batch_size=32, num_workers=0):
                images, targets = images.to(device), targets.to(device)
                output = model(images)
                total_loss += nn.functional.cross_entropy(output, targets, reduction="sum").item()
                correct += (output.argmax(1) == targets).sum().item()
        result = "Test samples={} accuracy={:.4f} mean_loss={:.4f}".format(
            len(test_data), correct/len(test_data), total_loss/len(test_data))
        print(result)
        if is_new:
            # 写在训练日志末尾，只记录最终评估，不据此修改最佳验证权重。
            with open(os.path.join(os.path.dirname(model_path), "training.log"), "a", encoding="utf-8") as log:
                log.write(result+"\n")


if __name__ == "__main__":
    main()

