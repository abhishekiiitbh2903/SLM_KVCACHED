from model.config import SLMConfig
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
import time

class LayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(config.d_emb))
        self.shift = nn.Parameter(torch.zeros(config.d_emb))

    def forward(self, x):
        mean_value = x.mean(dim=-1, keepdim=True)
        std_value = x.std(dim=-1, keepdim=True)
        normalized_x = (x - mean_value) / (std_value + self.eps)
        return self.scale.to(x.device) * normalized_x + self.shift.to(x.device)
    

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device)) *
            (x + 0.044715 * torch.pow(x, 3))
        )) 
    
class FeedForwardNN(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(in_features=config.d_emb,out_features=4*config.d_emb),
            GELU(),
            nn.Linear(in_features=4*config.d_emb,out_features=config.d_emb)
        )

    def forward(self,x):
        return self.layers(x) 
    
class CausalMultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.W_query = nn.Linear(config.d_emb, config.d_emb, bias=config.qkv_bias)
        self.W_keys = nn.Linear(config.d_emb, config.d_emb, bias=config.qkv_bias)
        self.W_values = nn.Linear(config.d_emb, config.d_emb, bias=config.qkv_bias)
        self.out_proj = nn.Linear(config.d_emb, config.d_emb)
        self.attn_dropout = nn.Dropout(config.drop_rate)
        self.register_buffer('mask', torch.triu(torch.ones(config.n_blocks, config.n_blocks), diagonal=1))

    def forward(self, x, kvcache=None):
        batch, tokens, _ = x.shape
        query = self.W_query(x)
        key = self.W_keys(x)
        value = self.W_values(x)

        if kvcache:
            prev_key, prev_val = kvcache
            key = torch.cat([prev_key, key], dim=1)
            value = torch.cat([prev_val, value], dim=1)

        new_kvcache = [key, value]
        curr_T = key.shape[1]

        query = query.view(batch, tokens, self.config.n_heads, self.config.head_d_emb).transpose(1, 2)
        key = key.view(batch, curr_T, self.config.n_heads, self.config.head_d_emb).transpose(1, 2)
        value = value.view(batch, curr_T, self.config.n_heads, self.config.head_d_emb).transpose(1, 2)

        scores = query @ key.transpose(2, 3)

        if tokens == 1:
            masking_vector = self.mask[curr_T - 1 : curr_T, :curr_T].to(dtype=torch.bool, device=scores.device)
        else:
            masking_vector = self.mask[:tokens, :curr_T].to(dtype=torch.bool, device=scores.device)

        masked_scores = scores.masked_fill(masking_vector, float('-inf'))
        attention_weights = torch.softmax(masked_scores / (key.shape[-1])**0.5, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)

        context_vector = attention_weights @ value
        context_vector = context_vector.transpose(1, 2).contiguous().view(batch, tokens, self.config.d_emb)
        context_vector = self.out_proj(context_vector)

        return context_vector, new_kvcache


class TransformerBlock(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.att=CausalMultiHeadAttention(config)
        self.norm1=LayerNorm(config)
        self.norm2=LayerNorm(config)
        self.ff=FeedForwardNN(config)

    def forward(self,x,kvcache=None):
        shortcut1=x
        x=self.norm1(x)
        x,cache_ele=self.att(x,kvcache)
        x=x+shortcut1 

        shortcut2=x
        x=self.norm2(x)
        x=self.ff(x)
        x=x+shortcut2

        return x,cache_ele
    

class GPT_KV(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.tok_emb=nn.Embedding(num_embeddings=config.vocab_size,embedding_dim=config.d_emb)
        self.pos_emb=nn.Embedding(num_embeddings=config.n_blocks,embedding_dim=config.d_emb)
        self.trf_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm=LayerNorm(config)
        self.out_head=nn.Linear(in_features=config.d_emb,out_features=config.vocab_size,bias=False)
        self.tok_emb.weight=self.out_head.weight

    def forward(self,x,targets=None,kvcache=None):
        device=x.device
        _,tokens=x.shape
        token_embeddings=self.tok_emb(x)
        pos_embeddings=self.pos_emb(torch.arange(0,tokens,device=device))
        x=token_embeddings+pos_embeddings 

        if not kvcache:
            kvcache = [None] * self.config.n_layers
        else:
            x = x[:, [-1], :]

        new_kvcache=[]
        for block, cache in zip(self.trf_blocks,kvcache):
            x, cache_ele = block(x, kvcache=cache)
            new_kvcache.append(cache_ele)
            
            
        x=self.final_norm(x)
        
        if targets is not None:
            logits=self.out_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.out_head(x[:, [-1], :])
            loss=None
            
        return logits,loss,new_kvcache

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        kvcache=None
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.n_blocks else idx[:, -self.config.n_blocks:]
            logits, _,kvcache = self(idx_cond,kvcache=kvcache)
            logits = logits[:, -1, :] / temperature 
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    

if __name__=="__main__":
    config=SLMConfig()
    model=GPT_KV(config)
    total_params=0
    for p in model.parameters(): 
        total_params+=p.numel()
    print(f"Total Parameters in the Model is: {total_params}")
    device="cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    tokenizer=tiktoken.get_encoding("gpt2")
    sentence = "A little girl went to the woods"
    device = next(model.parameters()).device
    context = (torch.tensor(tokenizer.encode_ordinary(sentence)).unsqueeze(dim = 0)).to(device)
    start_time=time.time()
    y = model.generate(context, 100)
    time_taken=time.time()-start_time
    print(tokenizer.decode(y.squeeze().tolist()))
    print(f"Time Taken by the Model :Without Loading Pretrained Model: {time_taken}")
    print("Loading Model Parameters")
    best_model_params_path = "model/best_model_params.pt"
    model.load_state_dict(torch.load(best_model_params_path, map_location=torch.device(device)))
    sentence = "A little girl went to the woods"
    device = next(model.parameters()).device
    context = (torch.tensor(tokenizer.encode_ordinary(sentence)).unsqueeze(dim = 0)).to(device)
    start_time=time.time()
    y = model.generate(context, 100)
    time_taken=time.time()-start_time
    print(tokenizer.decode(y.squeeze().tolist()))
    print(f"Time Taken by the Model :With Loading Pretrained Model: {time_taken}")


