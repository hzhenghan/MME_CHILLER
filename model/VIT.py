from functools import partial
from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("S", type=int, default=16, help="input dimension")
# parser.add_argument("N", type=int, default=3, help="input dimension")
# parser.add_argument("--variable_num", type=int, default=64, help="train data second dimension")
# opt = parser.parse_args()

S = 16
N = 4

def drop_path(x, drop_prob: float = 0., training: bool = False):

    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class PatchEmbed(nn.Module):

    def __init__(self, img_size=S, embed_dim=N * S, norm_layer=None):
        #in_c为输入通道数，RGB图像所以三个通道
        #embed_dim为输入嵌入之后的维度
        super().__init__()

        self.img_size = img_size

        self.proj = nn.Linear(img_size, embed_dim)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):

        # B, C, H, W = x.shape
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x) #.flatten(2).transpose(1, 2)
        # print(x.shape)
        #x = self.norm(x)
        return x

class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim 3*16
                 num_heads=4,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.1,
                 proj_drop_ratio=0.1):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) #dim * 3有助于并行化提高训练速度
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim] 会加一个class token所以加1
        B, N, C = x.shape #C=32

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        #print(qkv.shape)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale  #q @ k.transpose(-2, -1)表示Q乘K的转置，-2 -1表示最后两个维度
        #print(attn.shape) #[200, 4, 16, 16]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = attn @ v
        head = x.transpose(0, 1)
        head1, head2, head3, head4 = head[0], head[1], head[2], head[3]
        # print(head1.shape) #[200, 16, 8]
        # print(x.shape) #[200, 4, 16, 8]
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=F.gelu, drop=0.):
        super().__init__()
        # out_features = out_features or in_features
        # hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, 256)
        self.act = act_layer#()
        self.fc2 = nn.Linear(256, N * S)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        #x = self.drop(x)
        return x

class Block(nn.Module):  #1个Encoder块
    def __init__(self,
                 dim, #每个token的dimension 3*16
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=F.gelu,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x))
        x1 = self.attn(self.norm1(x)) #, head1, head2, head3, head4
        x = x + x1 #self.attn(self.norm1(x))
        y = self.mlp(self.norm1(x))
        x = x + self.mlp(self.norm1(x))
        return x #, head1, head2, head3, head4

class VisionTransformer(nn.Module):
    def __init__(self, img_size=64, patch_size=4, in_c=1, num_classes=8,
                 embed_dim=N * S, out_channel=64 / S, depth=4, num_heads=4, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, S=S):

        super(VisionTransformer, self).__init__()
        self.S = S
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1 #embed_dim=1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or F.gelu

        self.patch_embed = embed_layer(img_size=S, embed_dim=embed_dim, norm_layer=None)

        self.cls_token = nn.Parameter(torch.rand(1, 1, embed_dim)) #embed_dim=32  zeros
        #self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None #不用管默认是None
        #self.pos_embed = nn.Parameter(torch.rand(1, out_channel, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(self.num_features, num_classes), ) #nn.LogSoftmax()

        nn.init.normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        x = x.view(-1, 4, 16)
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        #x = self.pos_drop(x + self.pos_embed)
        #x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)  #[B, num_heads, embedd_dim]

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x[:, 0, :]
        x = x.unsqueeze(1)
        # x = self.classifier(x)
        # print('shape of classifier is {}'.format(x.shape))
        return x

def _init_vit_weights(m):

    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
    elif isinstance(m, nn.Conv1d):
        #
        n = m.kernel_size[0] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()

def vit_base_patch16_224(num_classes: int = 1000):

    model = VisionTransformer(img_size=64,
                              patch_size=4,
                              embed_dim=N * S,
                              depth=1,
                              num_heads=4,
                              representation_size=None,
                              num_classes=num_classes)
    return model
