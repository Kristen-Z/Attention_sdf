import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

class CrossAttention(nn.Module):
    def __init__(self, dim_q, dim_kv, heads=4, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(dim_q, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_kv, inner_dim * 2, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(inner_dim, dim_kv)
        self.layer_norm = nn.LayerNorm(dim_kv, eps=1e-6)

    def forward(self, q, kv):
        h = self.heads

        q = self.to_q(q)
        k, v = self.to_kv(kv).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        attn = sim.softmax(dim=-1)
        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        # add & norm
        out = self.dropout(self.fc(out)) 
        out += out

        return self.layer_norm(out)

class Decoder(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
    ):
        super(Decoder, self).__init__()

        def make_sequence():
            return []

        dims = [latent_size] + dims + [1]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, input):

        if input.shape[1] > 3 and self.latent_dropout:
            x = F.dropout(x, p=0.2, training=self.training)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], dim=-1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x= self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x


class Attention_SDF(nn.Module):
    def __init__(
        self,
        latent_size,
        cross_atten_num,
        decoder_dims,
        pre_norm=False,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=True,
        xyz_in_all=False,
        use_tanh=False,
        latent_dropout=False,
    ):
        super(Attention_SDF, self).__init__()

        crossattention = CrossAttention(dim_q=3,dim_kv=latent_size)
        self.attn_layers = nn.ModuleList([])
        self.pre_norm = pre_norm
        for i in range(cross_atten_num):
            self.attn_layers.append(crossattention)
        self.decoder = Decoder(latent_size,decoder_dims,dropout,dropout_prob,norm_layers,latent_in,weight_norm,xyz_in_all,use_tanh,latent_dropout)
    
    def forward(self, latent_codes, coords):
        if self.pre_norm:
            latent_codes = F.normalize(latent_codes, dim=-1)
            coords = F.normalize(coords, dim=-1)
        latent_codes = rearrange(latent_codes, 'b ... d -> b (...) d')
        coords = rearrange(coords, 'b ... d -> b (...) d')
        for crossattn in self.attn_layers:
            latent_emb = crossattn(coords,latent_codes)
        
        sdf = self.decoder(latent_emb)

        return sdf.squeeze(dim=1)
    
# latent = torch.randn(10, 256)
# coords = torch.randn(10, 3)
# model = Attention_SDF(256,4,[ 512, 512, 512, 512, 512, 512, 512, 512 ],dropout=[0, 1, 2, 3, 4, 5, 6, 7],dropout_prob=0.2,norm_layers=[0, 1, 2, 3, 4, 5, 6, 7],latent_in=[4])
# print(model)
# res = model(latent,coords)
# print(res.shape)
# print(res)