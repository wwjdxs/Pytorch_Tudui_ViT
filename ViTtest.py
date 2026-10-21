import os
import argparse
from pathlib import Path
from ViTanalysis import export_analysis
import torch
import torchvision
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from ViTmodel import cifar10
from ViTexperiment import build_model, get_transform


def main():
    parser = argparse.ArgumentParser(description="评估CIFAR-10 ViT及导出错误分析")
    root = Path(__file__).resolve().parent
    parser.add_argument("--checkpoint", type=Path, default=Path(os.environ.get("VIT_CHECKPOINT", root / "models_vit_optimized" / "best.pth")))
    parser.add_argument("--data-path", default=os.environ.get("CIFAR10_DATA", str(root / "data")))
    parser.add_argument("--image", default=os.environ.get("VIT_IMAGE", ""))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--no-analysis", action="store_true", help="只评估，不导出错误图片")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size 必须为正数")
    if not args.checkpoint.is_file():
        parser.error(f"未找到权重：{args.checkpoint}；请训练或通过--checkpoint指定已有权重")
    model_path, data_path, img_path = str(args.checkpoint), args.data_path, args.image
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    # 新版权重带少量配置，旧版只有state_dict；两者预处理和头数不同，不能混用。
    is_new = "model_state" in checkpoint
    model_type = checkpoint.get("model_type", "custom") if is_new else "custom"
    model = build_model(model_type, checkpoint["model_config"]) if is_new else cifar10()
    model.load_state_dict(checkpoint["model_state"] if is_new else checkpoint)
    model = model.to(device).eval()
    transform = get_transform(model_type=model_type, normalize=checkpoint.get("normalize", False) if is_new else False)

    if img_path:
        with Image.open(img_path) as image:
            image = image.convert("RGB")
            if model_type == "custom":
                image = image.resize((32, 32))
            image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            index = model(image).argmax(1).item()
        classes = ("airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck")
        print("预测类别：{} ({})".format(classes[index], index))
    else:
        test_data = torchvision.datasets.CIFAR10(data_path, train=False, download=args.download, transform=transform)
        total_loss, correct = 0.0, 0
        predictions, confidences = [], []
        with torch.no_grad():
            for images, targets in DataLoader(test_data, batch_size=args.batch_size, num_workers=0):
                images, targets = images.to(device), targets.to(device)
                output = model(images)
                total_loss += nn.functional.cross_entropy(output, targets, reduction="sum").item()
                correct += (output.argmax(1) == targets).sum().item()
                confidence, prediction = output.softmax(1).max(1)
                predictions.extend(prediction.cpu().tolist())
                confidences.extend(confidence.cpu().tolist())
        result = "Test samples={} accuracy={:.4f} mean_loss={:.4f}".format(
            len(test_data), correct/len(test_data), total_loss/len(test_data))
        print(result)
        if not args.no_analysis:
            export_analysis(test_data, predictions, confidences, model_path,
                            total_loss / len(test_data), checkpoint.get("epoch") if is_new else None)
        if is_new:
            # 写在训练日志末尾，只记录最终评估，不据此修改最佳验证权重。
            with open(os.path.join(os.path.dirname(model_path), "training.log"), "a", encoding="utf-8") as log:
                log.write(result+"\n")


if __name__ == "__main__":
    main()
