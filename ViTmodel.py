import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_c=3, embed_dim=48, norm_layer=None):
        super().__init__()
        if img_size <= 0 or patch_size <= 0 or img_size % patch_size != 0:
            raise ValueError("img_size必须能被正整数patch_size整除")
        img_size=(img_size,img_size)#二维元组
        patch_size=(patch_size,patch_size)
        self.img_size=img_size
        self.patch_size=patch_size
        self.grid_size=(img_size[0]//patch_size[0],img_size[1]//patch_size[1])#32/4=8
        self.num_patch=self.grid_size[0]*self.grid_size[1]#8*8=64
        num_patch=self.num_patch

        #按patch拆分
        self.proj=nn.Conv2d(
            in_channels=in_c,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            )#[B,3,32,32]->[B,48,8,8]

        #有norm_layer则使用，没有则保持原样
        self.norm=norm_layer(embed_dim) if norm_layer else nn.Identity()
        # 顺序：卷积切块、展平、可选归一化、拼接CLS、加位置编码。

        #设置可学习的分类标记cls_token: shape[1,1,D], forward时expand成[B,1,D]
        self.cls_token=nn.Parameter(torch.zeros(1,1,embed_dim))

        #设置可学习的绝对位置编码：shape[1,num_patch+1,D]
        #[B,64+1,48]
        self.pos_embed=nn.Parameter(torch.zeros(1,num_patch+1,embed_dim))#不需要再batch_size上扩充，因为广播机制

        #对加了位置编码后的序列做dropout(正则化)
        self.pos_drop=nn.Dropout(p=0.1)#p的设置再找一些参考



    def forward(self,x):
        B,C,H,W=x.shape  #获取输入张量的性质

        #判断张量大小与模型是否匹配
        assert H==self.img_size[0] and W==self.img_size[1],\
        f"输入图像大小{H}*{W}与模型期望大小{self.img_size[0]}*{self.img_size[1]}不符合"

        #[B,C,H,W]->[B,N,D]
        #[B,48,8,8]->flatten[2,3位置]->[B,48,64]->transpose(1,2)交换1,2位置的维度->[B,64,48]
        x=self.proj(x).flatten(2).transpose(1,2)

        #传入norm_layer,如果有则用
        x=self.norm(x)

        #cls_token.expand [1,1,D]->[B,1,D]
        cls_token=self.cls_token.expand(x.shape[0],-1,-1) #x.shape[0]=batch_size,-1=保持不变

        #加上cls_token[B,N,D]->[B,N+1,D]
        x=torch.cat((cls_token,x),dim=1)#[B,64,48]->[B,65,48]

        #加位置编码
        x=self.pos_drop(x+self.pos_embed)

        return x

#多头注意力机制
class attention(nn.Module):
    def __init__(self, 
                 dim, #输入的token维度：48
                 num_head=8, # 注意力头数，必须整除dim；48/8=6
                 qkv_bias=False, #qkv线性层是否带bia
                 qk_scale=None, #可选：自定义缩放因子。如果为None，则使用1/sqrt(head_dim)
                 attn_drop_ratio=0, #attention权重的dropout的比率，防止过拟合
                 proj_drop_ratio=0 #最终投影层的dropout比例
                ):
        super(attention,self).__init__()
    
        if num_head <= 0 or dim % num_head != 0:
            raise ValueError("dim必须能被正整数num_head整除")
        self.num_head=num_head #注意力头数
        head_dim=dim // num_head  #每个注意力头的维度dk
        self.scale = head_dim**(-0.5) if qk_scale is None else qk_scale  #qk的缩放因子
        
        
        #一次线性层同时生成Q,K,V
        self.qkv=nn.Linear(dim,3*dim,bias=qkv_bias)
        self.attn_drop=nn.Dropout(attn_drop_ratio)#dropout:自动失活一些神经元，防止过拟合

        #多头concat后再做一次输出投影 W0
        self.proj=nn.Linear(dim,dim)
        self.proj_drop=nn.Dropout(proj_drop_ratio)


    def forward(self,x):
        #x=[B,N+1,D]
        B,N,D=x.shape

        #1)生成qkv：[B,N+1,3D]
        #2)reshape:[B,N+1,3,h,d] d=D/h
        #3)permute:[3,B,h,N+1,d] 方便拆出qkv
        qkv=self.qkv(x).reshape(B,N,3,self.num_head, D//self.num_head).permute(2,0,3,1,4)#[索引位置]

        # q,k,v=[B,h,N+1,d]，其中d=D/h
        q,k,v=qkv[0],qkv[1],qkv[2]


        #加权求和得到每个token的新表示：
        #attn @ v: [B,h,N+1,d]
        #transpose:[B,N+1,h,d]
        #reshape:[B,N+1,D] h*d=D(把多头拼回去)

        # 计算仍为softmax(QK转置/sqrt(d)) @ V。
        # SDPA在硬件和数据类型支持时使用融合内核；否则自动回退，实际显存需测量。
        # 注意：SDPA不会自动根据eval关闭dropout，所以评估时必须明确传0。
        x = F.scaled_dot_product_attention(
            q, k, v, scale=self.scale,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(B, N, D)  # 合并各头，恢复[B,N,D]
        x=self.proj(x)
        x=self.proj_drop(x)

        return x


class MLP(nn.Module):
    def __init__(self,
                 in_features, #输入维度D
                 hidden_features=None, #隐藏层维度
                 out_features=None, #输出维度，默认等于输入维度
                 act_layer=nn.GELU, #激活函数：ViT默认用GELU):
                 drop=0):
        super().__init__()

        #默认保持输入输出维度一致（残差连接要求）
        out_features=out_features if out_features is not None else in_features
        hidden_features=hidden_features  or in_features

        #第一层：升纬度（D->hidden）
        self.fc1=nn.Linear(in_features,hidden_features)
        #激活：GELU（VIT,BERT常用）
        self.act=act_layer()
        #第二层：降维（hidden->D）
        self.fc2=nn.Linear(hidden_features,out_features)
        #drop_out:在fc1,fc2后面都用一遍
        self.drop=nn.Dropout(drop)


    def forward(self,x):
        # 每个token独立通过同一个MLP，只改变特征维度。
        x=self.fc1(x)  # [B,N,D] -> [B,N,hidden]
        x=self.act(x)
        x=self.drop(x)
        x=self.fc2(x)  # [B,N,hidden] -> [B,N,out_features]
        return self.drop(x)


class Block(nn.Module):
    """Pre-Norm编码块：注意力混合token信息，MLP变换特征。"""
    def __init__(self, dim, num_head=8, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = attention(dim, num_head, qkv_bias=True,
                              attn_drop_ratio=drop, proj_drop_ratio=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        # 两次残差连接保留原始信息，并为梯度提供直接传播路径。
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """适用于32×32 CIFAR10的小型ViT；默认8头兼容旧权重，轻量训练显式用3头。"""
    def __init__(self, img_size=32, patch_size=4, in_c=3, num_classes=10,
                 embed_dim=48, depth=6, num_head=8, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        if depth <= 0 or mlp_ratio <= 0:
            raise ValueError("depth和mlp_ratio必须为正数")
        self.patch_embed = PatchEmbed(img_size, patch_size, in_c, embed_dim)
        self.patch_embed.pos_drop.p = drop
        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_head, mlp_ratio, drop) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)
        # CLS和位置编码使用小随机数初始化，它们会随训练更新。
        nn.init.trunc_normal_(self.patch_embed.cls_token, std=0.02)
        nn.init.trunc_normal_(self.patch_embed.pos_embed, std=0.02)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.patch_embed(x)  # [B,3,32,32] -> [B,65,48]
        x = self.blocks(x)       # 保持[B,65,48]，CLS逐层汇总图像信息
        x = self.norm(x)
        x = x[:, 0]             # 仅取CLS特征：[B,48]
        return self.head(x)      # [B,10]原始logits，交叉熵内部处理softmax


# 兼容原训练代码使用的模型名称。
cifar10 = VisionTransformer


if __name__ == "__main__":
    model = cifar10()
    images = torch.randn(2, 3, 32, 32)
    logits = model(images)
    nn.CrossEntropyLoss()(logits, torch.tensor([0, 1])).backward()
    print("输出形状：", logits.shape)
    print("参数量：", sum(p.numel() for p in model.parameters()))
