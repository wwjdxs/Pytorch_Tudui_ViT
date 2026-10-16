"""CPU入口：复用同一套训练代码，避免两份流程产生差异。"""
from ViTtrain_gpu1 import main

if __name__ == "__main__":
    main(device="cpu")
