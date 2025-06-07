from dataclasses import dataclass 

@dataclass
class SLMConfig:
    n_blocks=128
    d_emb=256
    n_layers=4
    vocab_size=50257
    n_heads=4
    is_debug=False
    drop_rate=0.2
    qkv_bias=False
    head_d_emb=d_emb//n_heads

    