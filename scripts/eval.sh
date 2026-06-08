#!/bin/bash
set -e

CKPT=${1:-checkpoints/final.pt}
CONFIG=${2:-configs/350m.yaml}

python -c "
import torch
from src.model.config import ModelConfig
from src.model.model import MambaAttentionLM
from src.evaluation.perplexity import evaluate_perplexity
from src.evaluation.hellaswag import evaluate_hellaswag
from src.evaluation.lambada import evaluate_lambada

config = ModelConfig.from_yaml('$CONFIG')
model = MambaAttentionLM(config)
state_dict = torch.load('$CKPT', map_location='cpu')
if 'model_state_dict' in state_dict:
    state_dict = state_dict['model_state_dict']
model.load_state_dict(state_dict)
model.cuda()

ppl = evaluate_perplexity(model)
print(f'WikiText PPL: {ppl:.2f}')

acc = evaluate_hellaswag(model)
print(f'HellaSwag: {acc:.4f}')

acc = evaluate_lambada(model)
print(f'LAMBADA: {acc:.4f}')
"
