# CIFAR10轻量ViT

本机已实际完成50轮训练：最佳验证准确率 **73.58%**（第49轮），完整测试准确率 **72.93%**，旧权重复核为58.49%，提高 **14.44个百分点**。实际训练PyTorch张量峰值约 **60.7MiB**。详情见[优化验证记录](优化验证记录.md)和[训练日志](models_vit_optimized/training.log)。

保持原来的48维、6层、4×4 patch，不增大网络。原8头模型和新3头模型都为175,738个参数。训练流程按数据、模型、优化器、训练、验证、保存的顺序写在ViTtrain_gpu1.py中；CPU入口只复用这个流程，避免重复维护两份代码。

## 修改内容

|项目|原配置|新配置|
|---|---|---|
|训练轮数|10|50（增加时间，不增加峰值显存）|
|batch|64|32|
|注意力头数|8，每头6维|3，每头16维；参数量不变|
|注意力实现|手动建立注意力矩阵|PyTorch SDPA，能使用融合内核时自动选择|
|精度|FP32|CUDA训练使用FP16混合精度，参数仍为FP32|
|数据|ToTensor、不打乱|训练随机裁剪、翻转、打乱；所有新数据标准化至约[-1,1]|
|学习率|固定3e-4|最高5e-4，5轮预热，再余弦下降至1e-5|
|正则化|所有参数都做weight decay|排除CLS、位置编码、bias、LayerNorm；训练标签平滑0.1|
|权重保存|每轮一个文件|仅best.pth和last.pth|

ViT本来就有LayerNorm，因此不直接照搬CNN的BatchNorm。额外加入梯度裁剪，防止偶发大梯度；zero_grad(set_to_none=True)释放旧梯度引用；AdamW设置foreach=False减少临时张量。实际计算峰值仍取决于批次大小、内核和环境。

## 内存控制

注意力分数的元素数与batch×头数×token数平方成正比：当前token数仍为65，仅batch与头数变化就将这部分的元素数降为原来的18.75%；整个模型的实际峰值还包括MLP、梯度等，不能按这个比例直接推算。

训练/验证共用一份CIFAR10原始uint8图片，约146.5MiB，不复制另一份图片做验证，不缓存整套float32图像。ViTdata.py只有一个小型数据视图类和预处理函数，两个子集分别应用训练/验证变换。num_workers=0，不使用固定内存预取。进程总内存还包括Python、PyTorch、驱动及临时加载开销，不能只看数据数组大小。

合成批次预热后实测：旧batch64/8头/FP32约234.89MiB，新batch32/3头/FP16约60.71MiB，均为PyTorch max_memory_allocated，降低约74.2%。保持batch64/8头/FP32、采用SDPA及新版优化器设置时约234.40MiB，差别很小；这不是严格的单因素对照，也不能将新配置的全部节省归因于SDPA。实际收益依赖批次、头数、精度和可用内核。详见[优化验证记录](优化验证记录.md)。驱动上下文与缓存另计，nvidia-smi可能显示更高数值。

## 训练

选择安装了PyTorch和torchvision的环境，例如本机pytorch2.14环境。本版使用torch.amp.GradScaler和SDPA，本机已验证PyTorch 2.15.0.dev20260913+cu130。无TensorBoard、pandas、命令行参数解析或大型实验框架。

在ViTtrain_gpu1.py顶部直接修改data_path、batch_size、epoch等参数。发布版默认将数据放在仓库的data目录，首次运行自动下载；也可以通过CIFAR10_DATA环境变量指定已有数据目录。

```powershell
# 自动使用GPU，无CUDA时回退CPU
python ViTtrain_gpu1.py

# 显式使用CPU（完整训练会慢）
python ViTtrain.py
```

每类抽取4500张训练、500张验证，seed=42；与CNN分层划分相同。每轮只看验证集，按最佳验证准确率保存best.pth，最后保存last.pth。训练集准确率是在增强、Dropout和参数更新中统计的，不能直接当作固定模型对干净训练集的准确率。

新权重和简短training.log放在本目录models_vit_optimized，不覆盖旧models_vit_gpu。再次运行会覆盖这个优化目录的同名文件，需要保留多个实验时请修改save_path。当前不保存优化器状态，不支持从中断处恢复训练。

## 测试

```powershell
python ViTtest.py
```

默认读取新best.pth，用完整官方测试集做最终评估。模型文件中保存少量配置，包括头数和标准化标识，测试自动使用相同设置。设置ViTtest.py中的img_path可预测图片，也可用VIT_IMAGE环境变量指定图片。

复核旧8头、只用ToTensor的权重：

```powershell
$env:VIT_CHECKPOINT = "你的旧权重目录\vit9.pth"
python ViTtest.py
Remove-Item Env:VIT_CHECKPOINT
```

旧版纯state_dict会按原默认8头、无标准化加载，新版按文件中的配置加载。虽然不同头数的参数形状相同，计算方式仍不同，所以不能忽略头数配置。原权重复核为58.49%，平均测试loss为1.1425。

仓库仅附带最佳权重和训练日志；last.pth和旧models_vit_gpu未上传。旧权重复核命令需要自行提供原文件。请先执行 `pip install -r requirements.txt`，GPU版PyTorch请按官方安装指引选择与你设备匹配的版本。

本轮优化同时改变了训练预算、数据与优化设置，属于组合方案，不是各因素的独立消融实验；准确率提升幅度需以完成后的实测为准。不要根据官方测试集反复调参。CNN目录保持不变。

实现参考：[PyTorch SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)、[混合精度训练](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html)。

