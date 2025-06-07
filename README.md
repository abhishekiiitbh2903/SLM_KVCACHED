# 🧠 Small Language Model (SLM) - 16M Parameters

This repository introduces a lightweight, efficient **Small Language Model (SLM)** built entirely from scratch using the Transformer Decoder architecture. With only **~16 million parameters**, this model stands in sharp contrast to today's large-scale LLMs, making it ideal for experimentation, learning, and resource-constrained applications.

---

## 🚀 Highlights

- ✅ **Custom-built Transformer Decoder**-based architecture  
- ✅ Only **16M parameters** — compact yet functional  
- ✅ **KV Cache** implemented during inference for multi-fold speedup  
- ✅ Training and inference notebooks provided  
- ✅ Fully **Dockerized** and available on [Docker Hub](https://hub.docker.com/repository/docker/laymansoul/slm_kvcache)  

---

## 🏗️ Architecture

The model follows the standard **Transformer Decoder-only** structure, commonly used in autoregressive language models like GPT. Core components include:

- Multi-head self-attention 
- Positional embeddings
- LayerNorm and feed-forward blocks
- Causal masking for autoregression

---

## ⚙️ Inference Optimizations

To improve real-world usage and reduce latency during generation:

- We implemented **Key-Value Caching (KV Cache)** at inference time.
- This drastically reduces repeated computations over previous tokens.


---

## 📒 Contents

- `training.ipynb` – Full training pipeline with dataset preparation  
- `inference.ipynb` – Text generation with and without KV cache  
- `Dockerfile` – Container setup for deployment  
- `docker-compose.yml` – Multi-service orchestration (UI/API)  

---

## 🐳 Dockerized Deployment

To run the application at your system, Follow Following Steps:
- Pull images (slm-ui and slm-api) from the [Docker Hub](https://hub.docker.com/repository/docker/laymansoul/slm_kvcache) through Docker CLI (Note: Make sure you have Docker on your system)
- Run following commands to pull the images
  ```
  docker pull laymansoul/slm_kvcache:slm-api
  docker pull laymansoul/slm_kvcache:slm-ui
  ```
- Run following commands to run container
    ```
    docker network create slmnet
    docker run -d --name slm-api --network slmnet -p 8000:8000 slm-api
    docker run -d --name slm-ui --network slmnet -p 8501:8501 slm-ui
    ```
- ### Access UI

Visit [http://localhost:8501](http://localhost:8501) to interact with the model:

- **Untrained model** response  
- **Trained model** (without KV cache)  
- **Trained model** (with KV cache) -- demonstrates significantly reduced response time thanks to efficient key-value caching

### 🚀 Why KV Cache?

Leveraging **KV (Key-Value) cache** enables the model to avoid redundant computations during inference by reusing previously computed attention keys and values.

This results in:
- ⚡ **Drastically faster response generation**
- 📈 Improved performance for **longer sequences**
- 🔁 Efficiency gains on **repeated or streaming queries**
