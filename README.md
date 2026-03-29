# AutoRegressive Generation with B-rep Holistic Token Sequence Representation（CVPR 2026）
[CVPR 2026] Official PyTorch Implementation of "AutoRegressive Generation with B-rep Holistic Token Sequence Representation".
<img width="1476" height="708" alt="image" src="https://github.com/user-attachments/assets/0c2bec0e-fbd3-43ec-b2bf-f7533cc76d8c" />

# environment
We will provide a Conda environment package later.
```python
conda create --name breparg python=3.10
conda activate breparg

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
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
