# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


"""
Clay models for multi-sensor Earth observation data.

This module contains the Clay-MAE architecture implementation, a multi-sensor foundation
model for Earth observation that can handle different satellite sensors and modalities.
"""

import math
from typing import Any, Dict, Optional, Tuple
import timm
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch import nn, Tensor
from torchvision.transforms import v2
from torchvision.models._api import Weights, WeightsEnum


class WavesTransformer(nn.Module):
    """Transformer for processing wavelength information."""

    def __init__(
        self,
        wave_dim: int,
        output_dim: int,
        num_latent_tokens: int,
        embed_dim: int,
        is_decoder: bool,
        num_heads: int = 2,
        num_layers: int = 1,
    ) -> None:
        """Initialize WavesTransformer.

        Args:
            wave_dim: Wavelength embedding dimension
            output_dim: Output dimension
            num_latent_tokens: Number of learnable tokens
            embed_dim: Embedding dimension
            is_decoder: Whether this is used in decoder
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
        """
        super().__init__()
        self.num_latent_tokens = num_latent_tokens
        self.is_decoder = is_decoder

        # Ensure wave_dim is divisible by num_heads
        self.wave_dim = wave_dim
        if self.wave_dim % num_heads != 0:
            # Round up wave_dim to nearest multiple of num_heads
            self.wave_dim = ((wave_dim + num_heads - 1) // num_heads) * num_heads
            self.input_proj = nn.Linear(wave_dim, self.wave_dim)
        else:
            self.input_proj = nn.Identity()

        layer = nn.TransformerEncoderLayer(
            d_model=self.wave_dim,
            nhead=num_heads,
            dim_feedforward=self.wave_dim * 4,
            activation="gelu",
            dropout=0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)

        self.fc_weight = nn.Linear(self.wave_dim, output_dim)
        self.fc_bias = None if self.is_decoder else nn.Linear(self.wave_dim, embed_dim)

        self.weight_tokens = nn.Parameter(
            torch.randn(self.num_latent_tokens, self.wave_dim) * 0.02
        )
        self.bias_token = nn.Parameter(torch.randn(1, self.wave_dim) * 0.02)

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass.

        Args:
            x: Input tensor of shape (N, D)

        Returns:
            Tuple of (weights tensor, optional bias tensor)
        """
        x = self.input_proj(x)
        x = torch.cat([self.weight_tokens, x, self.bias_token], dim=0)
        out = self.encoder(x)
        weights = self.fc_weight(
            out[self.num_latent_tokens:-1] + x[self.num_latent_tokens:-1]
        )
        bias = None if self.is_decoder else self.fc_bias(out[-1])
        return weights, bias

class Transformer(nn.Module):
    """Transformer encoder module."""

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        fused_attn: bool = True,
    ) -> None:
        """Initialize the transformer.

        Args:
            dim: Input dimension
            depth: Number of transformer layers
            heads: Number of attention heads
            dim_head: Dimension of each attention head
            mlp_dim: MLP hidden dimension
            fused_attn: Whether to use fused attention
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, fused_attn=fused_attn),
                        FeedForward(dim, mlp_dim),
                    ]
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, N, D)

        Returns:
            Output tensor of shape (B, N, D)
        """
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class Attention(nn.Module):
    """Multi-head self-attention module."""

    def __init__(
        self,
        dim: int, 
        heads: int = 8, 
        dim_head: int = 64, 
        fused_attn: bool = True
    ) -> None:
        """Initialize attention module.

        Args:
            dim: Input dimension
            heads: Number of attention heads
            dim_head: Dimension of each attention head
            fused_attn: Whether to use fused attention
        """

        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.fused_attn = fused_attn

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, N, D)

        Returns:
            Output tensor of shape (B, N, D)
        """
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = attn.softmax(dim=-1)
            x = torch.matmul(attn, v)

        x = rearrange(x, "b h n d -> b n (h d)")
        return self.to_out(x)

class FeedForward(nn.Module):
    """Feed-forward network module."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        """Initialize feed-forward network.

        Args:
            dim: Input dimension
            hidden_dim: Hidden dimension
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        return self.net(x)

class DynamicEmbedding(nn.Module):
    """Dynamic embedding module for handling different sensor modalities."""

    def __init__(
        self,
        wave_dim: int,
        num_latent_tokens: int,
        patch_size: int,
        embed_dim: int,
        is_decoder: bool = False,
    ) -> None:
        """Initialize dynamic embedding.

        Args:
            wave_dim: Wavelength embedding dimension
            num_latent_tokens: Number of learnable tokens
            patch_size: Size of image patches
            embed_dim: Output embedding dimension
            is_decoder: Whether this is used in decoder
        """
        super().__init__()
        self.wave_dim = wave_dim
        self.num_latent_tokens = num_latent_tokens
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.is_decoder = is_decoder
        self.output_dim = (patch_size**2) * embed_dim if is_decoder else embed_dim

        self.weight_generator = WavesTransformer(
            wave_dim=wave_dim,
            output_dim=self.output_dim,
            num_latent_tokens=num_latent_tokens,
            embed_dim=embed_dim,
            is_decoder=is_decoder,
        )

    def forward(self, x: Tensor, waves: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W) or (B, L, D)
            waves: Wavelength information tensor of shape (N,)

        Returns:
            Tuple of (embedded tensor, processed waves)
        """
        # Process wavelength information
        waves = waves.float()  # Ensure float type
        waves = (waves - waves.min()) / (waves.max() - waves.min())  # Normalize to [0,1]
        waves = posemb_sincos_1d(waves, self.wave_dim)
        
        # Generate weights and bias
        weights, bias = self.weight_generator(waves)

        if self.is_decoder:
            # For decoder: (cin, k1*k2*cout) -> (cin*k1*k2, cout)
            dynamic_weight = rearrange(
                weights,
                'cin (k1 k2 cout) -> (cin k1 k2) cout',
                k1=self.patch_size,
                k2=self.patch_size,
                cout=self.embed_dim,
            )
            if bias is not None:
                bias = rearrange(bias, 'b -> (b)')
            out = F.linear(x, dynamic_weight * 0.02, bias=bias)
        else:
            # For encoder: handle input shape
            if len(x.shape) == 3:
                B, L, D = x.shape
                C = D // (self.patch_size * self.patch_size)  # Calculate original channels
                H = W = int(math.sqrt(L))  # Calculate original H, W
                x = rearrange(
                    x,
                    'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                    h=H, w=W,
                    p1=self.patch_size, p2=self.patch_size,
                    c=C
                )

            # Reshape weights for convolution
            dynamic_weight = rearrange(
                weights,
                'cin (cout k1 k2) -> cout cin k1 k2',
                k1=self.patch_size,
                k2=self.patch_size,
            )
            if bias is not None:
                bias = rearrange(bias, 'b -> (b)')

            # Apply convolution
            out = F.conv2d(
                x,
                dynamic_weight * 0.02,
                bias=bias,
                stride=self.patch_size
            )
            
            # Reshape output back to sequence form
            B, C, H, W = out.shape
            out = rearrange(out, 'b c h w -> b (h w) c')

        return out, waves

class Encoder(nn.Module):
    """Encoder module for the Clay-MAE model."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the encoder.

        Args:
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        self.mask_ratio = kwargs.get("mask_ratio", 0.75)
        self.patch_size = kwargs.get("patch_size", 8)
        self.shuffle = kwargs.get("shuffle", True)

        self.metadata = kwargs.get("metadata", None)

        self.encoder = Transformer(
            dim=kwargs.get("dim", 768),
            depth=kwargs.get("depth", 12),
            heads=kwargs.get("heads", 12),
            dim_head=kwargs.get("dim_head", 64),
            mlp_dim=kwargs.get("mlp_dim", 3072),
            fused_attn=kwargs.get("fused_attn", True),
        )
        
        self.dynamic_embedding = DynamicEmbedding(
            wave_dim=kwargs.get("wave_dim", 10),
            num_latent_tokens=kwargs.get("num_latent_tokens", 16),
            patch_size=kwargs.get("patch_size", 8),
            embed_dim=kwargs.get("embed_dim", 768),
            is_decoder=False,
        ) 
        
        self.encoder_norm = nn.LayerNorm(kwargs.get("dim", 768))

    def forward(self, x: Tensor, waves: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, N, D)
            waves: Wavelength information tensor

        Returns:
            Encoded tensor
        """
        # Embed patches
        x, waves = self.dynamic_embedding(x, waves)
        
        # Add positional embeddings
        x = x + self.pos_embedding[:, :x.size(1)]
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        return x
    
class Decoder(nn.Module):   
    """Decoder module for the Clay-MAE model."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the decoder.

        Args:       
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        self.mask_ratio = kwargs.get("mask_ratio", 0.75)
        self.patch_size = kwargs.get("patch_size", 8)
        self.shuffle = kwargs.get("shuffle", True)  
        self.metadata = kwargs.get("metadata", None)
        
        self.decoder = Transformer(
            dim=kwargs.get("dim", 768),
            depth=kwargs.get("depth", 12),
            heads=kwargs.get("heads", 12),
            dim_head=kwargs.get("dim_head", 64),
            mlp_dim=kwargs.get("mlp_dim", 3072),
            fused_attn=kwargs.get("fused_attn", True),
        )  
        
        self.decoder_norm = nn.LayerNorm(kwargs.get("dim", 768))
        
    def forward(self, x: Tensor, waves: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: Input tensor from encoder
            waves: Wavelength information tensor

        Returns:
            Tuple of (decoded tensor, processed waves)
        """
        # Project to decoder dimension
        x = self.decoder_norm(x)
        
        # Add positional embeddings
        x = x + self.pos_embedding[:, :x.size(1)]
        
        # Apply transformer blocks
        x = self.decoder(x)
        
        # Final projection to patch dimension
        x = self.pred(x)
        
        return x, waves



class ClayMAE(nn.Module):
    """Clay-MAE model architecture."""

    def __init__(
        self,
        mask_ratio: float = 0.75,
        patch_size: int = 8,
        norm_pix_loss: bool = False,
        shuffle: bool = True,
        metadata: Optional[Dict] = None,
        teacher: str = "vit_base_patch16_224",
        dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        dim_head: int = 64,
        mlp_ratio: float = 4.0,
        decoder_dim: int = 512,
        decoder_depth: int = 4,
        decoder_heads: int = 4,
        decoder_dim_head: int = 64,
        decoder_mlp_ratio: float = 4.0,
        wave_dim: int = 10,
        num_latent_tokens: int = 16,
        **kwargs: Any,
    ) -> None:
        """Initialize Clay-MAE model.

        Args:
            mask_ratio: Ratio of patches to mask
            patch_size: Size of patches
            norm_pix_loss: Whether to normalize pixels in loss calculation
            shuffle: Whether to shuffle patches
            metadata: Sensor metadata dictionary
            teacher: Name of teacher model
            dim: Encoder embedding dimension
            depth: Encoder transformer depth
            heads: Number of encoder attention heads
            dim_head: Dimension of encoder attention heads
            mlp_ratio: Encoder MLP ratio
            decoder_dim: Decoder embedding dimension
            decoder_depth: Decoder transformer depth
            decoder_heads: Number of decoder attention heads
            decoder_dim_head: Dimension of decoder attention heads
            decoder_mlp_ratio: Decoder MLP ratio
            wave_dim: Wavelength embedding dimension
            num_latent_tokens: Number of learnable tokens
        """
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss
        self.shuffle = shuffle
        self.metadata = metadata

        # Initialize teacher model
        self.teacher = timm.create_model(teacher, pretrained=True, num_classes=0)
        self.teacher_chip_size = 518
        self.teacher_resize = v2.Resize(size=(self.teacher_chip_size, self.teacher_chip_size))
        self.proj = nn.Linear(dim, self.teacher.num_features)

        # Initialize encoder and decoder
        self.encoder = Encoder(
            mask_ratio=mask_ratio,
            patch_size=patch_size,
            shuffle=shuffle,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_ratio=mlp_ratio,
            wave_dim=wave_dim,
            num_latent_tokens=num_latent_tokens,
        )

        self.decoder = Decoder(
            mask_ratio=mask_ratio,
            patch_size=patch_size,
            encoder_dim=dim,
            dim=decoder_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            dim_head=decoder_dim_head,
            mlp_ratio=decoder_mlp_ratio,
        )

        self.freeze_teacher()

    def freeze_teacher(self) -> None:
        """Freeze teacher model parameters."""
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

    def forward(self, datacube: Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass.

        Args:
            datacube: Dictionary containing input tensors

        Returns:
            Tuple of (total loss, reconstruction loss, representation loss)
        """
        x = datacube["pixels"]  # Shape: [B, C, H, W]
        waves = datacube["waves"]
        B, C, H, W = x.shape

        # Handle channel dropout during training
        if self.training and torch.rand(1).item() > 0.9:  # 10% chance of dropout
            channel_mask = torch.rand(B, C, 1, 1, device=x.device) > 0.1
            x = x * channel_mask

        # Process teacher input first
        with torch.no_grad():
            # Handle channel count for teacher
            teacher_input = x
            if C != 3:
                if C > 3:
                    teacher_input = x[:, :3]
                else:
                    teacher_input = torch.cat([x] * (3 // C) + [x[:, :(3 % C)]], dim=1)
            
            teacher_tokens = self.teacher.forward_features(teacher_input)
            teacher_tokens = self.teacher.norm(teacher_tokens)

        # Get patches for student model
        patches = rearrange(
            x,
            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
            p1=self.patch_size,
            p2=self.patch_size
        )

        # Process through encoder
        tokens = self.encoder(patches, waves)
        
        # Random masking
        num_patches = tokens.shape[1]
        num_masked = int(self.mask_ratio * num_patches)
        
        # Generate random indices for masking
        if self.shuffle:
            noise = torch.rand(B, num_patches, device=x.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
        else:
            ids_shuffle = torch.arange(num_patches, device=x.device).expand(B, -1)
            ids_restore = ids_shuffle

        # Keep visible tokens for encoder
        ids_keep = ids_shuffle[:, :num_patches-num_masked]
        tokens_visible = torch.gather(
            tokens,
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, tokens.shape[-1])
        )

        # Pass through encoder-decoder
        encoded = self.encoder_norm(tokens_visible)
        decoded = self.decoder(encoded, waves)
        decoded = self.decoder_norm(decoded)

        # Compute reconstruction loss
        if self.norm_pix_loss:
            mean = patches.mean(dim=-1, keepdim=True)
            var = patches.var(dim=-1, keepdim=True)
            patches = (patches - mean) / (var + 1.e-6)**.5

        # Reconstruction loss for visible patches
        reconstruction_loss = F.mse_loss(
            decoded,
            patches.gather(1, ids_restore.unsqueeze(-1).repeat(1, 1, patches.shape[-1]))
        )

        # Representation loss between teacher and student
        student_tokens = tokens_visible.mean(dim=1)  # global average pooling
        representation_loss = F.mse_loss(student_tokens, teacher_tokens[:, 0])  # compare with [CLS] token

        # Total loss
        loss = reconstruction_loss + 0.1 * representation_loss

        return loss, reconstruction_loss, representation_loss

class ClayMAEBase_Weights(WeightsEnum):
    """Clay-MAE Base weights."""

    PRETRAIN = Weights(
        url="",
        transforms=None,
        meta={
            "architecture": "Clay-MAE",
            "publication_name": "Clay-MAE: A Multi-Sensor Foundation Model for Earth Observation",
            "num_params": 86_000_000,
        },
    )

def clay_mae_base(
    weights: Optional[ClayMAEBase_Weights] = None,
    progress: bool = True,
    **kwargs: Any,
) -> ClayMAE:
    """Create a Clay-MAE base model.

    Args:
        weights: Optional pretrained weights
        progress: Whether to show progress bar when downloading weights
        **kwargs: Additional arguments to pass to ClayMAE

    Returns:
        ClayMAE model
    """
    args = {
        "dim": 768,
        "depth": 12,
        "heads": 12,
        "dim_head": 64,
        "mlp_ratio": 4,
        "decoder_dim": 512,
        "decoder_depth": 4,
        "decoder_heads": 4,
        "decoder_dim_head": 64,
        "decoder_mlp_ratio": 4,
        "wave_dim": 12,  # Changed from 10 to 12 to be divisible by num_heads=2
        "num_latent_tokens": 16,
        "patch_size": 8,
    }
    args.update(kwargs)
    model = ClayMAE(**args)

    if weights is not None:
        state_dict = weights.get_state_dict(progress=progress)
        model.load_state_dict(state_dict, strict=False)

    return model

def clay_mae_large(
    weights: Optional[ClayMAEBase_Weights] = None,
    progress: bool = True,
    **kwargs: Any,
) -> ClayMAE:
    """Create a Clay-MAE Large model.

    Args:
        weights: The pretrained weights to use
        progress: If True, displays a progress bar of the download to stderr
        **kwargs: Additional arguments to pass to the model

    Returns:
        A Clay-MAE Large model
    """
    args = {
        "dim": 1024,
        "depth": 24,
        "heads": 16,
        "dim_head": 64,
        "mlp_ratio": 4,
        "decoder_dim": 512,
        "decoder_depth": 4,
        "decoder_heads": 4,
        "decoder_dim_head": 64,
        "decoder_mlp_ratio": 4,
    }
    args.update(kwargs)

    model = ClayMAE(**args)

    if weights is not None:
        state_dict = weights.get_state_dict(progress=progress)
        model.load_state_dict(state_dict, strict=False)

    return model

def posemb_sincos_1d(x: Tensor, dim: int) -> Tensor:
    """Create sinusoidal positional embeddings.

    Args:
        x: Input tensor to embed
        dim: Embedding dimension

    Returns:
        Tensor with positional embeddings
    """
    half_dim = dim // 2
    emb = math.log(10000) / half_dim
    emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
    emb = x.unsqueeze(-1) * emb.unsqueeze(0)
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
    
    # Handle odd dimensions
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb