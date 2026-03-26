"""Neural network components and SB3 policies for the trading agent.

Includes CNN/Transformer feature extractors, relative position attention blocks,
and Actor-Critic policy definitions used by PPO training.
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class RelativePositionBias(nn.Module):
    """T5-style可学习相对位置偏置，用于长序列注意力。"""

    def __init__(
        self,
        num_heads: int,
        num_buckets: int = 32,
        max_distance: int = 256,
        bidirectional: bool = True,
    ):
        super().__init__()
        if num_buckets < 2:
            raise ValueError("num_buckets must be >= 2 for relative position bias.")
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """Bucket化相对位置，控制极远距离的分辨率。"""
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        ret = torch.zeros_like(relative_position, dtype=torch.long)

        if self.bidirectional:
            half_buckets = num_buckets // 2
            sign = (relative_position > 0).to(torch.long) * half_buckets
            ret += sign
            relative_position = torch.abs(relative_position)
            num_buckets = half_buckets
        else:
            relative_position = torch.clamp(-relative_position, min=0)

        max_exact = max(1, num_buckets // 2)
        is_small = relative_position < max_exact
        large_pos = relative_position.clone().float()
        large_pos = max_exact + (
            torch.log(large_pos / max_exact + 1e-6)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        )
        large_pos = large_pos.to(torch.long)
        large_pos = torch.clamp(large_pos, max=num_buckets - 1)
        bucket = torch.where(is_small, relative_position, large_pos)
        return ret + bucket

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        context = torch.arange(seq_len, device=device)
        relative_position = context[None, :] - context[:, None]
        bucket = self._relative_position_bucket(relative_position)
        values = self.relative_attention_bias(bucket)
        # (seq, seq, heads) -> (1, heads, seq, seq)
        return values.permute(2, 0, 1).unsqueeze(0)


class RelativeMultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead for multihead attention.")
        self.embed_dim = d_model
        self.num_heads = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        rel_bias: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if rel_bias is not None:
            attn_scores = attn_scores + rel_bias
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        context = (
            torch.matmul(attn_probs, v)
            .transpose(1, 2)
            .contiguous()
            .view(bsz, seq_len, self.embed_dim)
        )
        context = self.out_dropout(context)
        return self.out_proj(context)


class RelativeTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str = "gelu",
    ):
        super().__init__()
        self.self_attn = RelativeMultiheadSelfAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(self, src: torch.Tensor, rel_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output = self.self_attn(src, rel_bias=rel_bias)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        feedforward = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(feedforward)
        return self.norm2(src)


class RelativeTransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        num_layers: int,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                RelativeTransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, src: torch.Tensor, rel_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            src = layer(src, rel_bias=rel_bias)
        return src


class CustomCNN(BaseFeaturesExtractor):
    """
    多尺度CNN特征提取器，用于从时间序列数据中提取不同时间尺度的特征。
    :param observation_space: (gym.Space)
    :param features_dim: (int) 网络输出的特征数量
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # observation_space.shape -> (window_size, num_features)
        # PyTorch Conv1d expects (batch, channels, length)
        # so num_features is channels (in_channels), window_size is length
        num_features = observation_space.shape[1]

        # 多尺度CNN分支
        # 短期模式（3天周期）- 捕捉短期价格波动和技术指标变化
        self.short_conv = nn.Sequential(
            nn.Conv1d(num_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # 中期模式（7天周期）- 捕捉中期趋势和支撑阻力
        self.medium_conv = nn.Sequential(
            nn.Conv1d(num_features, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # 长期模式（15天周期）- 捕捉长期趋势和周期性模式
        self.long_conv = nn.Sequential(
            nn.Conv1d(num_features, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Conv1d(96, 64, kernel_size=1),  # 3*32=96通道融合为64
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Flatten(),
        )

        # 计算Flatten之后的维度
        with torch.no_grad():
            # 创建一个虚拟输入来动态计算CNN输出的大小
            # SB3的 observation 是 (batch_size, window_size, num_features)
            # 需要转换为 (batch_size, num_features, window_size)
            dummy_input = torch.zeros(1, *observation_space.shape).permute(0, 2, 1)
            fused_output = self._forward_multiscale(dummy_input)
            cnn_output_dim = fused_output.shape[1]

        # 使用一个线性层将CNN的输出映射到所需的 features_dim
        self.linear = nn.Sequential(nn.Linear(cnn_output_dim, features_dim), nn.ReLU())

    def _forward_multiscale(self, x):
        """多尺度特征提取的前向传播"""
        # 并行提取不同尺度的特征
        short_feat = self.short_conv(x)  # 短期特征
        medium_feat = self.medium_conv(x)  # 中期特征
        long_feat = self.long_conv(x)  # 长期特征

        # 拼接多尺度特征
        concat_feat = torch.cat([short_feat, medium_feat, long_feat], dim=1)

        # 特征融合
        fused = self.fusion(concat_feat)
        return fused

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # SB3的 observation 是 (batch_size, window_size, num_features)
        # 需要转换为 (batch_size, num_features, window_size)
        observations = observations.permute(0, 2, 1)
        features = self._forward_multiscale(observations)
        return self.linear(features)


class CNNActorCriticPolicy(ActorCriticPolicy):
    """
    一个用于Stable-Baselines3的自定义策略网络，
    它使用自定义的1D-CNN作为特征提取器。
    """

    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        # 明确地将自定义CNN设置为特征提取器
        kwargs["features_extractor_class"] = CustomCNN
        # 可以为特征提取器传递参数
        if "features_extractor_kwargs" not in kwargs:
            kwargs["features_extractor_kwargs"] = dict(
                features_dim=128
            )  # 例如，定义CNN输出的特征维度

        super(CNNActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            # net_arch 定义在特征提取之后 Actor 和 Critic 网络的结构
            # 例如: [dict(pi=[64, 64], vf=[64, 64])]
            # 如果为空，则不添加额外的隐藏层
            net_arch=[],
            **kwargs,
        )


class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    Transformer-based特征提取器，将时间序列视作token序列，通过自注意力聚合多尺度信息。
    默认使用CLS池化，也支持mean池化。
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        d_model: int = 128,
        n_heads: int = 4,
        depth: int = 2,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        pooling: str = "cls",
        positional_encoding: str = "learned",
        use_relative_position_bias: bool = False,
        relative_attention_buckets: int = 32,
        relative_attention_max_distance: int = 256,
    ):
        super().__init__(observation_space, features_dim)

        if pooling not in {"cls", "mean"}:
            raise ValueError(f"Unsupported pooling strategy: {pooling}")
        positional_encoding = positional_encoding.lower()
        if positional_encoding not in {"learned", "sinusoidal", "none"}:
            raise ValueError(f"Unsupported positional encoding: {positional_encoding}")

        seq_len, num_features = observation_space.shape
        self.use_cls = pooling == "cls"
        self.seq_len = seq_len
        self.d_model = d_model
        self.positional_encoding = positional_encoding
        self.use_relative_encoder = use_relative_position_bias

        self.input_proj = nn.Linear(num_features, d_model)
        total_tokens = seq_len + 1 if self.use_cls else seq_len
        self.pos_embedding: Optional[nn.Parameter] = None
        self.sinusoidal_embedding: Optional[torch.Tensor] = None
        if self.positional_encoding == "learned":
            self.pos_embedding = nn.Parameter(torch.zeros(1, total_tokens, d_model))
        elif self.positional_encoding == "sinusoidal":
            sinusoidal = self._build_sinusoidal(total_tokens, d_model)
            self.register_buffer("sinusoidal_embedding", sinusoidal, persistent=False)
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.cls_token = None

        if use_relative_position_bias:
            self.relative_bias = RelativePositionBias(
                num_heads=n_heads,
                num_buckets=relative_attention_buckets,
                max_distance=relative_attention_max_distance,
            )
            self.encoder = RelativeTransformerEncoder(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=ff_dim or d_model * 4,
                dropout=dropout,
                activation="gelu",
                num_layers=depth,
            )
        else:
            self.relative_bias = None
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=ff_dim or d_model * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(d_model, features_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        if self.pos_embedding is not None:
            nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def _build_sinusoidal(self, total_tokens: int, dim: int) -> torch.Tensor:
        position = torch.arange(total_tokens, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(total_tokens, dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _apply_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        if self.positional_encoding == "learned" and self.pos_embedding is not None:
            return x + self.pos_embedding[:, : x.size(1), :]
        if self.positional_encoding == "sinusoidal" and self.sinusoidal_embedding is not None:
            return x + self.sinusoidal_embedding[:, : x.size(1), :].to(x.device)
        return x

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (batch, window_size, num_features)
        x = self.input_proj(observations)  # -> (batch, seq_len, d_model)

        if self.use_cls:
            batch_size = x.size(0)
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self._apply_positional_encoding(x)
        rel_bias = None
        if self.relative_bias is not None:
            rel_bias = self.relative_bias(x.size(1), x.device)

        if self.use_relative_encoder:
            x = self.encoder(x, rel_bias=rel_bias)
        else:
            x = self.encoder(x)

        if self.use_cls:
            x = x[:, 0, :]
        else:
            x = x.mean(dim=1)

        x = self.layer_norm(x)
        x = self.dropout(x)
        return self.projection(x)


class AdaptiveTrustGate(nn.Module):
    """通道级自适应门控，支持可变模态数量。"""

    def __init__(self, num_modalities: int, feature_dim: int):
        super().__init__()
        self.num_modalities = num_modalities
        self.feature_dim = feature_dim
        hidden_dim = max(1, num_modalities * feature_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, modality_tensor: torch.Tensor) -> torch.Tensor:
        # modality_tensor: (batch, num_modalities, feature_dim)
        if modality_tensor.size(1) == 1:
            return modality_tensor[:, 0, :]
        flat = modality_tensor.reshape(modality_tensor.size(0), -1)
        gates = self.net(flat).reshape(
            modality_tensor.size(0), self.num_modalities, self.feature_dim
        )
        weighted = gates * modality_tensor
        denom = gates.sum(dim=1).clamp_min(1e-6)
        return weighted.sum(dim=1) / denom


class CrossAttentionFeatureExtractor(BaseFeaturesExtractor):
    """
    交叉注意力特征提取器：并行提取CNN与Transformer特征，通过交叉注意力与自适应门控融合，
    根据市场状态动态调节对不同指标的信任度。
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        cnn_dim: int = 128,
        transformer_dim: int = 128,
        cross_dim: int = 128,
        cross_heads: int = 4,
        transformer_kwargs: Optional[dict] = None,
        sentiment_dim: int = 0,
        market_dim: Optional[int] = None,
        auxiliary_dim: int = 0,
    ):
        super().__init__(observation_space, features_dim)
        seq_len, total_features = observation_space.shape
        requested_market_dim = int(market_dim) if market_dim else total_features
        self.market_dim = max(1, min(requested_market_dim, total_features))
        self.aux_dim = max(0, min(int(auxiliary_dim or 0), total_features - self.market_dim))
        available_for_sentiment = self.market_dim
        self.sentiment_dim = max(0, min(int(sentiment_dim or 0), available_for_sentiment))
        self.has_sentiment = self.sentiment_dim > 0
        self.base_market_dim = self.market_dim - self.sentiment_dim
        if self.base_market_dim <= 0:
            raise ValueError("基础市场特征数量必须大于0。")

        market_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(seq_len, self.base_market_dim),
            dtype=np.float32,
        )

        default_tf_kwargs = dict(
            d_model=transformer_dim,
            n_heads=cross_heads,
            depth=2,
            ff_dim=None,
            dropout=0.1,
            pooling="cls",
            positional_encoding="learned",
            use_relative_position_bias=True,
            relative_attention_buckets=32,
            relative_attention_max_distance=256,
        )
        if transformer_kwargs:
            default_tf_kwargs.update(transformer_kwargs)

        self.cnn_branch = CustomCNN(market_space, features_dim=cnn_dim)
        self.transformer_branch = TransformerFeatureExtractor(
            market_space,
            features_dim=transformer_dim,
            **default_tf_kwargs,
        )

        self.indicator_norm = nn.LayerNorm(seq_len)
        self.indicator_proj = nn.Linear(seq_len, cross_dim)
        attn_dropout = float(default_tf_kwargs.get("dropout", 0.1))
        self.indicator_token_norm = nn.LayerNorm(cross_dim)
        self.query_norm = nn.LayerNorm(cross_dim)
        self.cross_attn_out_norm = nn.LayerNorm(cross_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=cross_dim,
            num_heads=cross_heads,
            batch_first=True,
            dropout=attn_dropout,
        )

        self.cnn_query = nn.Linear(cnn_dim, cross_dim)
        self.transformer_query = nn.Linear(transformer_dim, cross_dim)
        self.cnn_value = nn.Linear(cnn_dim, cross_dim)
        self.transformer_value = nn.Linear(transformer_dim, cross_dim)

        if self.has_sentiment:
            self.sentiment_norm = nn.LayerNorm(self.sentiment_dim)
            self.sentiment_token_proj = nn.Linear(self.sentiment_dim, cross_dim)
            self.sentiment_token_norm = nn.LayerNorm(cross_dim)
            self.sentiment_query_proj = nn.Linear(cnn_dim + transformer_dim, cross_dim)
            self.sentiment_query_norm = nn.LayerNorm(cross_dim)
            self.sentiment_attn = nn.MultiheadAttention(
                embed_dim=cross_dim,
                num_heads=cross_heads,
                batch_first=True,
                dropout=attn_dropout,
            )
            self.sentiment_attn_out_norm = nn.LayerNorm(cross_dim)
            self.sentiment_decay_proj = nn.Linear(self.sentiment_dim, 1)
            self.sentiment_ema_proj = nn.Linear(self.sentiment_dim, cross_dim)
            self.sentiment_value_attention = nn.Linear(cross_dim, cross_dim)
            self.sentiment_value_ema = nn.Linear(cross_dim, cross_dim)
        else:
            self.sentiment_norm = None
            self.sentiment_token_proj = None
            self.sentiment_token_norm = None
            self.sentiment_query_proj = None
            self.sentiment_attn = None
            self.sentiment_query_norm = None
            self.sentiment_attn_out_norm = None
            self.sentiment_decay_proj = None
            self.sentiment_ema_proj = None
            self.sentiment_value_attention = None
            self.sentiment_value_ema = None

        modality_count = 2 + (2 if self.has_sentiment else 0)
        self.trust_gate = AdaptiveTrustGate(modality_count, cross_dim)
        if self.aux_dim > 0:
            self.aux_norm = nn.LayerNorm(self.aux_dim)
            self.aux_adapter = nn.Linear(self.aux_dim, cross_dim)
        else:
            self.aux_norm = None
            self.aux_adapter = None

        output_components = 2 + (1 if self.aux_adapter is not None else 0)
        self.output_dim = output_components * cross_dim
        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.output_dim),
            nn.Linear(self.output_dim, features_dim),
        )

    def _encode_indicators(self, market_observations: torch.Tensor) -> torch.Tensor:
        indicator_series = market_observations.permute(0, 2, 1)  # (batch, num_features, window)
        tokens = self.indicator_norm(indicator_series)
        tokens = self.indicator_proj(tokens)
        tokens = self.indicator_token_norm(tokens)
        return tokens

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        market_block = observations[:, :, : self.market_dim]
        aux_block = (
            observations[:, :, self.market_dim : self.market_dim + self.aux_dim]
            if self.aux_dim > 0
            else None
        )

        if self.has_sentiment:
            sentiment_seq = market_block[:, :, -self.sentiment_dim :]
            if self.base_market_dim > 0:
                market_obs = market_block[:, :, : self.base_market_dim]
            else:
                raise RuntimeError("市场特征维度配置错误。")
        else:
            sentiment_seq = None
            market_obs = market_block

        cnn_feat = self.cnn_branch(market_obs)
        transformer_feat = self.transformer_branch(market_obs)

        indicator_tokens = self._encode_indicators(market_obs)

        queries = torch.stack(
            [self.cnn_query(cnn_feat), self.transformer_query(transformer_feat)],
            dim=1,
        )
        queries = self.query_norm(queries)
        key_value_tokens = indicator_tokens

        attn_output, _ = self.cross_attention(queries, key_value_tokens, key_value_tokens)
        attn_output = self.cross_attn_out_norm(attn_output + queries)
        attended_indicators = attn_output.mean(dim=1)

        cnn_value = self.cnn_value(cnn_feat)
        transformer_value = self.transformer_value(transformer_feat)
        modalities = [cnn_value, transformer_value]

        if sentiment_seq is not None and self.sentiment_norm is not None:
            normed_sentiment = self.sentiment_norm(sentiment_seq)
            sentiment_tokens = self.sentiment_token_proj(normed_sentiment)
            sentiment_tokens = self.sentiment_token_norm(sentiment_tokens)
            query_source = torch.cat([cnn_feat, transformer_feat], dim=-1)
            sentiment_query = self.sentiment_query_proj(query_source).unsqueeze(1)
            sentiment_query = self.sentiment_query_norm(sentiment_query)
            attn_context, _ = self.sentiment_attn(
                sentiment_query, sentiment_tokens, sentiment_tokens
            )
            attn_context = self.sentiment_attn_out_norm(attn_context + sentiment_query)
            sentiment_context_attn = attn_context.squeeze(1)

            decays = torch.sigmoid(self.sentiment_decay_proj(normed_sentiment)).clamp(0.05, 0.95)
            ema_state = torch.zeros_like(normed_sentiment[:, 0, :])
            for idx in range(normed_sentiment.size(1)):
                token = normed_sentiment[:, idx, :]
                decay = decays[:, idx]
                ema_state = torch.lerp(ema_state, token, decay)
            sentiment_context_ema = self.sentiment_ema_proj(ema_state)

            modalities.append(self.sentiment_value_attention(sentiment_context_attn))
            modalities.append(self.sentiment_value_ema(sentiment_context_ema))

        modality_tensor = torch.stack(modalities, dim=1)
        adaptive_blend = self.trust_gate(modality_tensor)

        components = [adaptive_blend, attended_indicators]
        if self.aux_adapter is not None and aux_block is not None:
            aux_summary = self.aux_adapter(self.aux_norm(aux_block.mean(dim=1)))
            components.append(aux_summary)
        fused = torch.cat(components, dim=-1)
        return self.output_proj(fused)


class TransformerActorCriticPolicy(ActorCriticPolicy):
    """
    SB3策略封装：使用TransformerFeatureExtractor作为特征提取器。
    """

    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        if "features_extractor_kwargs" not in kwargs:
            kwargs["features_extractor_kwargs"] = dict(
                features_dim=128,
                d_model=128,
                n_heads=4,
                depth=2,
                ff_dim=None,
                dropout=0.1,
                pooling="cls",
                positional_encoding="learned",
                use_relative_position_bias=True,
                relative_attention_buckets=32,
                relative_attention_max_distance=256,
            )

        kwargs["features_extractor_class"] = TransformerFeatureExtractor

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[],
            **kwargs,
        )


class CrossAttentionActorCriticPolicy(ActorCriticPolicy):
    """
    使用交叉注意力特征提取器的SB3策略封装。
    """

    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        if "features_extractor_kwargs" not in kwargs:
            kwargs["features_extractor_kwargs"] = dict(
                features_dim=128,
                cnn_dim=128,
                transformer_dim=128,
                cross_dim=128,
                cross_heads=4,
                transformer_kwargs=None,
            )

        kwargs["features_extractor_class"] = CrossAttentionFeatureExtractor

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[],
            **kwargs,
        )
