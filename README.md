# AutoRegressive Generation with B-rep Holistic Token Sequence Representation（CVPR 2026）
[CVPR 2026] Official PyTorch Implementation of "AutoRegressive Generation with B-rep Holistic Token Sequence Representation".
<img width="1476" height="708" alt="image" src="https://github.com/user-attachments/assets/0c2bec0e-fbd3-43ec-b2bf-f7533cc76d8c" />

# environment

We provide a pre-built Conda environment package on [Hugging Face](https://huggingface.co/datasets/qingtiannihao/BrepARG_conda).

| File | Description | Size |
|------|-------------|------|
| [breparg.tar.gz](https://huggingface.co/datasets/qingtiannihao/BrepARG_conda/resolve/main/breparg.tar.gz) | Pre-built BrepARG conda environment (packed with `conda-pack`) | ~3.0 GB |

## Install from conda package

The environment was packed with `conda-pack`. For the migration workflow, see [this guide](https://blog.csdn.net/weixin_52581013/article/details/146208796).

**1. Download the environment package**

```bash
pip install -U huggingface_hub

huggingface-cli download qingtiannihao/BrepARG_conda breparg.tar.gz --local-dir .
```

**2. Create the target environment folder and extract**

Replace `~/anaconda3` with your own Conda installation path:

```bash
mkdir -p ~/anaconda3/envs/breparg
tar -xzvf breparg.tar.gz -C ~/anaconda3/envs/breparg
```

**3. Run `conda-unpack` (required after `conda-pack` extraction)**

```bash
~/anaconda3/envs/breparg/bin/conda-unpack
```

**4. Activate and verify**

```bash
conda activate breparg
conda info -e
```

# pretrained weights

Pretrained weights are hosted on [Hugging Face](https://huggingface.co/qingtiannihao/BrepARG) 

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

# process data
```python
python process_brep.py
python deduplicate_cad.py
python deduplicate_se_data.py
```

# training
**VQVAE:** --batch_size (Bigger is better) --train_epoch (Adjust according to the data volume)
```python
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_vqvae.py --data_list 'your own data paths' --surface_list 'deduplicated surface source data' --edge_list 'deduplicated edge source data' --batch_size 512 --train_epoch 3000
```

**AR:**
1. Prepare the AR data:
```python
python 2sequence.py
```
2. Train the autoregressive model:
```python
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_ar.py --sequence_file 'your own sequences path' --batch_size 32 --train_epoch 500 --learning_rate 1e-3
```

# generating brep
```python
python generate_brep.py
```

# evaluation

**Valid = success rate * watertight rate**

- **Success rate:** Generated B-reps / Total attempts
- **Watertight Rate:** Watertight models / Generated B-reps

**other Metric:** Follwing BrepGen https://github.com/samxuxiang/BrepGen?tab=readme-ov-file


# Citation
We would like to acknowledge the foundational contributions of the following works:
```bibtex
@article{xu2024brepgen,
  title={BrepGen: A B-rep Generative Diffusion Model with Structured Latent Geometry},
  author={Xu, Xiang and Lambourne, Joseph G and Jayaraman, Pradeep Kumar and Wang, Zhengqing and Willis, Karl DD and Furukawa, Yasutaka},
  journal={arXiv preprint arXiv:2401.15563},
  year={2024}
}
@inproceedings{li2025dtgbrepgen,
  title={Dtgbrepgen: A novel b-rep generative model through decoupling topology and geometry},
  author={Li, Jing and Fu, Yihang and Chen, Falai},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21438--21447},
  year={2025}
}
```
If you find our work or this paper helpful to your research, please consider citing:
```bibtex
@article{li2026autoregressive,
  title={AutoRegressive Generation with B-rep Holistic Token Sequence Representation},
  author={Li, Jiahao and Bai, Yunpeng and Dai, Yongkang and Guo, Hao and Gan, Hongping and Shi, Yilei},
  journal={arXiv preprint arXiv:2601.16771},
  year={2026}
}
```
