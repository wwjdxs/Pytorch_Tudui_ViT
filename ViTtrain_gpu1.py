import os
import time
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from ViTmodel import cifar10
from ViTdata import CIFARSubset, get_transform


def main(device=None):
    # 1. 参数集中放在这里。CPU入口复用本函数，不再复制另一整套训练循环。
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_path = os.environ.get("CIFAR10_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    batch_size = 32             # 原来64，减小batch可降低中间特征占用。
    epoch = 50                 # 增加训练时间，不增加模型或峰值显存大小。
    warmup_epoch = 5
    learning_rate = 5e-4
    model_config = dict(embed_dim=48, depth=6, num_head=3, patch_size=4, drop=0.1)
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats(device)
    torch.set_num_threads(4)
    print("训练设备：{}".format(device))

    # 2. 只加载一份uint8训练图片，不缓存整套float32图像，也不复制一份做验证集。
    # 每类4500张训练、500张验证，与CNN实验使用同样的分层划分。
    dataset = torchvision.datasets.CIFAR10(data_path, train=True, download=True)
    labels = torch.tensor(dataset.targets)
    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = [], []
    for label in range(10):
        indices = torch.where(labels == label)[0]
        indices = indices[torch.randperm(len(indices), generator=generator)]
        val_indices.extend(indices[:500].tolist())
        train_indices.extend(indices[500:].tolist())
    train_data = CIFARSubset(dataset, sorted(train_indices), get_transform(training=True))
    val_data = CIFARSubset(dataset, sorted(val_indices), get_transform())
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0,
                              generator=torch.Generator().manual_seed(42))
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    # workers=0，不启用固定内存预取，优先减少Windows进程和CPU内存占用。

    # 3. 维度和层数不变。3头时每头16维；头数改变不增加qkv/投影层参数。
    model = cifar10(**model_config).to(device)
    train_loss_fun = nn.CrossEntropyLoss(label_smoothing=0.1)  # 减少对训练标签的过度自信。
    eval_loss_fun = nn.CrossEntropyLoss()                     # 评估仍统计普通交叉熵。
    # 位置编码、CLS、bias和LayerNorm不做weight decay，避免直接衰减这些参数。
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.ndim == 1 or "pos_embed" in name or "cls_token" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": decay}, {"params": no_decay, "weight_decay": 0.0}
    ], lr=learning_rate, weight_decay=0.05, foreach=False)
    # foreach=False减少优化器临时张量；这里不改变AdamW算法。
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epoch)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epoch-warmup_epoch, eta_min=1e-5)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epoch])

    # 4. CUDA启用FP16混合精度，参数本身仍保留FP32；Scaler防止小梯度下溢。
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_vit_optimized")
    os.makedirs(save_path, exist_ok=True)
    best_accuracy = -1.0
    log_path = os.path.join(save_path, "training.log")
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("seed=42, train={}, validation={}, batch={}, epochs={}, heads={}, AMP={}\n".format(
            len(train_data), len(val_data), batch_size, epoch, model_config["num_head"], use_amp))
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
        # 元数据很小，但能避免测试时误用旧8头模型或忘记标准化。
        checkpoint = dict(model_state=model.state_dict(), model_config=model_config,
                          normalize=True, epoch=i+1, val_accuracy=accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(checkpoint, os.path.join(save_path, "best.pth"))
        scheduler.step()
    # 不保存每一轮，更不保存庞大的优化器状态；只留最佳和最后两个权重。
    torch.save(checkpoint, os.path.join(save_path, "last.pth"))
    print("最佳验证准确率：{:.2%}，模型目录：{}".format(best_accuracy, save_path))


if __name__ == "__main__":
    main()

