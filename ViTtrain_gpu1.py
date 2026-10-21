import os
import time
import argparse
import json
from datetime import datetime
from pathlib import Path
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from ViTdata import CIFARSubset
from ViTexperiment import build_model, get_transform, make_optimizer


def main(device=None):
    parser = argparse.ArgumentParser(description="CIFAR10 ViT 分阶段准确率实验")
    parser.add_argument("--experiment", choices=["long", "wide", "pretrained"], default="long")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--data-path", default=os.environ.get("CIFAR10_DATA", str(Path(__file__).resolve().parent / "data")))
    parser.add_argument("--download", action="store_true", help="下载CIFAR-10到数据目录")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "experiments")
    args = parser.parse_args()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_path = args.data_path
    pretrained = args.experiment == "pretrained"
    batch_size = args.batch_size if args.batch_size is not None else (8 if pretrained else 32)
    epoch = args.epochs if args.epochs is not None else (30 if pretrained else 150)
    if epoch < 2 or batch_size < 1:
        parser.error("epochs 必须至少为 2，batch-size 必须为正数")
    # long 先单独检验训练预算；wide 与 long 仅宽度不同。
    warmup_epoch = min(3 if pretrained else 5, epoch - 1)
    learning_rate = 3e-5 if pretrained else 5e-4
    model_type = "torchvision_vit_b_16" if pretrained else "custom"
    model_config = dict(num_classes=10) if pretrained else dict(
        embed_dim=96 if args.experiment == "wide" else 48,
        depth=6, num_head=3, patch_size=4, drop=0.1)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
    torch.set_num_threads(4)
    print("训练设备：{}".format(device))

    # 2. 只加载一份uint8训练图片，不缓存整套float32图像，也不复制一份做验证集。
    # 每类4500张训练、500张验证，与CNN实验使用同样的分层划分。
    dataset = torchvision.datasets.CIFAR10(data_path, train=True, download=args.download)
    labels = torch.tensor(dataset.targets)
    generator = torch.Generator().manual_seed(args.seed)
    train_indices, val_indices = [], []
    for label in range(10):
        indices = torch.where(labels == label)[0]
        indices = indices[torch.randperm(len(indices), generator=generator)]
        val_indices.extend(indices[:500].tolist())
        train_indices.extend(indices[500:].tolist())
    train_data = CIFARSubset(dataset, sorted(train_indices), get_transform(training=True, model_type=model_type))
    val_data = CIFARSubset(dataset, sorted(val_indices), get_transform(model_type=model_type))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    # workers=0，不启用固定内存预取，优先减少Windows进程和CPU内存占用。

    # 预训练分支训练全部参数；骨干与分类头使用不同学习率。
    model = build_model(model_type, model_config, pretrained=pretrained).to(device)
    train_loss_fun = nn.CrossEntropyLoss(label_smoothing=0.1)
    eval_loss_fun = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, learning_rate, pretrained)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epoch)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epoch-warmup_epoch, eta_min=1e-7 if pretrained else 1e-5)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epoch])

    # 4. CUDA启用FP16混合精度，参数本身仍保留FP32；Scaler防止小梯度下溢。
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    save_path = os.path.join(args.output_dir, args.experiment, run_name)
    os.makedirs(save_path, exist_ok=False)
    run_config = dict(experiment=args.experiment, model_type=model_type,
                      model_config=model_config, epochs=epoch, warmup_epochs=warmup_epoch,
                      batch_size=batch_size, seed=args.seed, data_path=data_path,
                      backbone_lr=learning_rate, head_lr=learning_rate * 10 if pretrained else learning_rate,
                      weight_decay=0.05, label_smoothing=0.1,
                      pretrained_weights="IMAGENET1K_V1" if pretrained else None,
                      device=str(device), torch_version=str(torch.__version__),
                      torchvision_version=str(torchvision.__version__))
    with open(os.path.join(save_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)
    print("本次实验目录：", save_path)
    best_accuracy = -1.0
    log_path = os.path.join(save_path, "training.log")
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("seed={}, train={}, validation={}, batch={}, epochs={}, heads={}, AMP={}\n".format(
            args.seed, len(train_data), len(val_data), batch_size, epoch, model_config.get("num_head", 12), use_amp))
    print("参数量：{}；训练/验证：{}/{}".format(
        sum(p.numel() for p in model.parameters()), len(train_data), len(val_data)))

    # 5. 训练流程保持：前向 -> loss -> 清梯度 -> 反向 -> 更新。
    for i in range(epoch):
        start = time.perf_counter()
        model.train()
        train_loss, train_correct = 0.0, 0
        for step, (images, targets) in enumerate(train_loader, 1):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)  # 释放旧梯度引用，减少常驻占用。
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                output = model(images)
                loss = train_loss_fun(output, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                     # 裁剪前先恢复真实梯度。
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * targets.size(0)
            train_correct += (output.argmax(1) == targets).sum().item()
            if step % 400 == 0:
                print("Epoch {} batch {}/{} loss={:.4f}".format(i+1, step, len(train_loader), loss.item()), flush=True)

        # 6. 验证不增强、不更新梯度，按样本数统计loss。测试集不参与选模型。
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                output = model(images)  # 验证使用FP32，测试时也使用相同口径。
                val_loss += eval_loss_fun(output, targets).item() * targets.size(0)
                val_correct += (output.argmax(1) == targets).sum().item()
        accuracy = val_correct / len(val_data)
        peak_mb = torch.cuda.max_memory_allocated(device)/1024**2 if use_amp else 0.0
        message = ("Epoch {}/{} lr={:.6f} train_loss={:.4f} train_acc={:.2%} "
                   "val_loss={:.4f} val_acc={:.2%} peak_GPU={:.1f}MiB seconds={:.1f}").format(
            i+1, epoch, optimizer.param_groups[0]["lr"], train_loss/len(train_data),
            train_correct/len(train_data), val_loss/len(val_data), accuracy,
            peak_mb, time.perf_counter()-start)
        print(message, flush=True)
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(message+"\n")
        # 保存模型类型和配置；测试不需要重新下载预训练权重。
        checkpoint = dict(model_state=model.state_dict(), model_config=model_config,
                          model_type=model_type, experiment=args.experiment,
                          pretrained_weights=run_config["pretrained_weights"],
                          normalize=True, epoch=i+1, val_accuracy=accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(checkpoint, os.path.join(save_path, "best.pth"))
        scheduler.step()
    # 不保存每一轮，更不保存庞大的优化器状态；只留最佳和最后两个权重。
    torch.save(checkpoint, os.path.join(save_path, "last.pth"))
    with open(os.path.join(save_path, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(dict(status="completed", best_val_accuracy=best_accuracy,
                       last_val_accuracy=accuracy, epochs=epoch,
                       test_accuracy=None), f, indent=2)
    print("最佳验证准确率：{:.2%}，模型目录：{}".format(best_accuracy, save_path))


if __name__ == "__main__":
    main()
