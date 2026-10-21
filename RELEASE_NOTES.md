# 实验发布说明

新增150轮训练、96维宽模型、图像预训练微调、错误分析和已执行报告。发布前仅对验证选出的best.pth补测，未依据测试结果调参。

本地release_artifacts/准备三个最佳权重附件：small150-best.pth（78.43%）、wide150-best.pth（82.30%）、pretrained30-best.pth（97.66%），及SHA-256清单。
附件不提交Git。343MB预训练权重需通过Releases等大文件渠道上传；尚未发布附件前，不提供未经验证的下载链接。
