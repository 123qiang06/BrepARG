# Pretrained Weights

Pretrained weights are hosted on [Hugging Face](https://huggingface.co/qingtiannihao/BrepARG) and are **not** included in the source repository.

Each checkpoint contains only `model_state_dict` (inference-ready, no optimizer or training metadata).

## Weight List

| File | Dataset | Model | Description | Size | Hugging Face Path |
|------|---------|-------|-------------|------|-------------------|
| [abc_ar.pt](https://huggingface.co/qingtiannihao/BrepARG/blob/main/checkpoint/weights/abc_ar.pt) | ABC | AR | Autoregressive sequence model (vocab=7222) | ~33 MB | `checkpoint/weights/abc_ar.pt` |
| [abc_vqvae.pt](https://huggingface.co/qingtiannihao/BrepARG/blob/main/checkpoint/weights/abc_vqvae.pt) | ABC | SE VQ-VAE | Surface/edge VQ-VAE (codebook=8192, dim=64) | ~220 MB | `checkpoint/weights/abc_vqvae.pt` |
| [deepcad_ar.pt](https://huggingface.co/qingtiannihao/BrepARG/blob/main/checkpoint/weights/deepcad_ar.pt) | DeepCAD | AR | Autoregressive sequence model (vocab=6198) | ~32 MB | `checkpoint/weights/deepcad_ar.pt` |
| [deepcad_vqvae.pt](https://huggingface.co/qingtiannihao/BrepARG/blob/main/checkpoint/weights/deepcad_vqvae.pt) | DeepCAD | SE VQ-VAE | Surface/edge VQ-VAE (codebook=4096, dim=64) | ~219 MB | `checkpoint/weights/deepcad_vqvae.pt` |

## Download

Download all weights into `checkpoint/weights/`:

```bash
pip install -U huggingface_hub

huggingface-cli download qingtiannihao/BrepARG \
  checkpoint/weights/abc_ar.pt \
  checkpoint/weights/abc_vqvae.pt \
  checkpoint/weights/deepcad_ar.pt \
  checkpoint/weights/deepcad_vqvae.pt \
  --local-dir .
```

## Inference Examples

**ABC:**

```bash
python generate_brep.py \
  --ar_model checkpoint/weights/abc_ar.pt \
  --se_vqvae checkpoint/weights/abc_vqvae.pt \
  --dataset_path data/abc_sequences_v3_no_vertex_v10.pkl
```

**DeepCAD:**

```bash
python generate_brep.py \
  --ar_model checkpoint/weights/deepcad_ar.pt \
  --se_vqvae checkpoint/weights/deepcad_vqvae.pt \
  --dataset_path data/deepcad_sequences_v3_no_vertex_v9.9.pkl
```
