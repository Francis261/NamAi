import torch
import torch.nn as nn
from .config import ModelConfig
from .hybrid_layer import HybridLayer, RMSNorm


class MambaAttentionLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            HybridLayer(config, i) for i in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.lm_head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=self.config.init_std)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @property
    def device(self):
        return next(self.parameters()).device
