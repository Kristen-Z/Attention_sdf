import torch
from torch import nn, einsum
import torch.nn.functional as F
import math
from einops import rearrange

PI = math.pi
class Decoder(nn.Module):
    def __init__(
        self,
        input_dim,
        coord_dim,
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
        
        dims = [input_dim] + dims + [1]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.coord_dim = coord_dim
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

    # input: N x (L)
    def forward(self, input):
        xyz = input[:, -self.coord_dim:]

        if input.shape[1] > 3 and self.latent_dropout:
            latent_vecs = input[:, :-self.coord_dim]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
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
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x
    

# Positional encoding like nerf
def positional_encoding(x, scale=None, l=10):
    ''' Implements positional encoding on the given coordinates.
    
    Differentiable wrt to x.
    
    Args:
        x: torch.Tensor(n, dim)  - input coordinates
        scale: torch.Tensor(2, dim) or None 
            scale along the coords for normalization
            If None - scale inferred from x
        l: int - number of modes in the encoding
    Returns:
        torch.Tensor(n, dim + 2 * dim * l) - positional encoded vector.
    '''

    if scale is None:
        scale = torch.vstack([x.min(axis=0)[0], x.max(axis=0)[0]]).T

    x_normed = 2 * (x - scale[:, 0]) / (scale[:, 1] - scale[:, 0]) - 1

    if l > 0:
        sinuses = torch.cat([torch.sin( (2 ** p) * PI * x_normed) for p in range(l) ], axis=1)
        cosines = torch.cat([torch.cos( (2 ** p) * PI * x_normed) for p in range(l) ], axis=1)

        pos_enc = torch.cat([x_normed, sinuses, cosines], axis=1)
    else:
        pos_enc = x_normed
    return pos_enc

# class Embedder:
#     def __init__(self, **kwargs):
#         self.kwargs = kwargs
#         self.create_embedding_fn()
        
#     def create_embedding_fn(self):
#         embed_fns = []
#         d = self.kwargs['input_dims']
#         out_dim = 0
#         if self.kwargs['include_input']:
#             embed_fns.append(lambda x : x)
#             out_dim += d
            
#         max_freq = self.kwargs['max_freq_log2']
#         N_freqs = self.kwargs['num_freqs']
        
#         if self.kwargs['log_sampling']:
#             freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
#         else:
#             freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
#         for freq in freq_bands:
#             for p_fn in self.kwargs['periodic_fns']:
#                 embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
#                 out_dim += d
                    
#         self.embed_fns = embed_fns
#         self.out_dim = out_dim
        
#     def embed(self, inputs):
#         return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


# def get_embedder(pos_emb_freq, if_pos_emb=True):
#     if not if_pos_emb:
#         return nn.Identity(), 3
    
#     embed_kwargs = {
#                 'include_input' : True,
#                 'input_dims' : 3,
#                 'max_freq_log2' : pos_emb_freq-1,
#                 'num_freqs' : pos_emb_freq,
#                 'log_sampling' : True,
#                 'periodic_fns' : [torch.sin, torch.cos],
#     }
    
#     embedder_obj = Embedder(**embed_kwargs)
#     embed = lambda x, eo=embedder_obj : eo.embed(x)
#     return embed, embedder_obj.out_dim



class PreNorm(nn.Module):
    def __init__(self, query_dim, fn, latent_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(query_dim)
        self.norm_latent = nn.LayerNorm(latent_dim) if latent_dim else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if self.norm_latent:
            context = kwargs['z']
            normed_context = self.norm_latent(context)
            kwargs.update(z = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, kv_dim=None, selfatt=True, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        if selfatt:
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        else:
            self.to_q = nn.Linear(dim, inner_dim, bias=False)
            self.to_kv = nn.Linear(kv_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, z=None):
        h = self.heads
        if z is None:
            qkv = self.to_qkv(x).chunk(3, dim=-1)
        else:
            q = self.to_q(x)
            k, v = self.to_kv(z).chunk(2, dim=-1)
            qkv = (q, k, v)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), qkv)
        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        attn = sim.softmax(dim=-1)
        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, kv_dim=None, heads=8, dim_head=64, selfatt=True, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, kv_dim, selfatt, heads=heads, dim_head=dim_head,
                                       dropout=dropout),kv_dim),
                PreNorm(dim, FeedForward(dim, dropout=dropout),kv_dim)
            ]))

    def forward(self, x, z=None):
        for attn, ff in self.layers:
            x = attn(x, z=z) + x
            x = ff(x) + x
        return x
    
class Scene_MLP(nn.Module):
    def __init__(self, latent_size, hidden_dims, do_pos_enc=False,pos_enc_freq=10,input_dim=3,norm_layers=(0,1,2,3), weight_norm=True,):
        super(Scene_MLP, self).__init__()
        
        self.norm_layers = [range(len(hidden_dims))]
        self.weight_norm = weight_norm
        self.do_pos_enc = do_pos_enc
        if do_pos_enc:
            input_dim *= (2*pos_enc_freq+1)
        layers = []
        dims = [input_dim] + hidden_dims + [latent_size]
        self.num_layers = len(dims)
        for layer in range(self.num_layers-1):
            if weight_norm and layer in self.norm_layers:
                layers.append(nn.utils.weight_norm(nn.Linear(dims[layer], dims[layer+1])))
            else:
                layers.append(nn.Linear(dims[layer], dims[layer+1]))

            if not weight_norm and self.norm_layers and layer in self.norm_layers:
                layers.append(nn.LayerNorm(dims[layer+1]))
        
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        if self.do_pos_enc:
            x = positional_encoding(x)
        x = self.layers(x)
        return x
    
class Attention_SDF(nn.Module):
    def __init__(
        self,
        latent_size,
        cross_atten_num,
        decoder_dims,
        self_attn,
        if_pos_emb=True,
        if_coord_query=True,
        pre_norm=False,
        pos_emb_freq=5,
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

        self.pre_norm = pre_norm
        self.self_attn = self_attn
        self.if_coord_query = if_coord_query
        self.if_pos_emb = if_pos_emb
        self.coord_dim = 3 if not if_pos_emb else (2*pos_emb_freq+1)*3
        self.pos_emb_freq = pos_emb_freq
        #self.embed_fn, self.coord_dim = get_embedder(pos_emb_freq, if_pos_emb)
        if not self_attn:
            q_dim = self.coord_dim if if_coord_query else latent_size
            kv_dim = latent_size if if_coord_query else self.coord_dim
        else:
            kv_dim = None
            q_dim = self.coord_dim+latent_size
        self.transformer = Transformer(q_dim,cross_atten_num,kv_dim,selfatt=self_attn)

        self.decoder = Decoder(q_dim,self.coord_dim,decoder_dims,dropout,dropout_prob,norm_layers,latent_in,weight_norm,xyz_in_all,use_tanh,latent_dropout)
    
    def forward(self, latent_codes, coords):
        if self.pre_norm:
            #latent_codes = F.normalize(latent_codes, dim=-1)
            coords = F.normalize(coords, dim=-1)
        if self.if_pos_emb:
            coords = positional_encoding(coords,l=self.pos_emb_freq)
        latent_codes = rearrange(latent_codes, 'b ... d -> b (...) d')
        coords = rearrange(coords, 'b ... d -> b (...) d')
        # self attention - input the concatenation
        if self.self_attn:
            concat = torch.cat([latent_codes,coords],dim=-1)
            latent_codes = self.transformer(concat)
        else:
            if self.if_coord_query:
                latent_codes = self.transformer(coords,latent_codes)
            else:
                latent_codes = self.transformer(latent_codes,coords)
        
        latent_codes = latent_codes.squeeze(dim=1)
        
        sdf = self.decoder(latent_codes)

        return sdf
    
# latent = torch.randn(10, 256)
# coords = torch.randn(10, 3)
# model = Attention_SDF(256,2,[ 512, 512, 512, 512],True,False,dropout=[0, 1, 2, 3],dropout_prob=0.2,norm_layers=[0, 1, 2, 3],latent_in=[2])
# scene_mlp = Scene_MLP(256,[256,256,256,256])
# latent = scene_mlp(coords)
# res = model(latent,coords)
# print(res.shape)
# print(res)
