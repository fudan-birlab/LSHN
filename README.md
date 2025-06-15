## LSHN
Official pytorch implementation of ["Latent Structured Hopfield Network for Semantic Association and Retrieval"](https://arxiv.org/abs/2506.01303)

### Abstract

<img src=".\assets\method.jpeg" alt="method" style="zoom: 15%;" />

Episodic memory enables humans to recall past experiences by associating semantic elements such as objects, locations, and time into coherent event representations. While large pretrained models have shown remarkable progress in modeling semantic memory, the mechanisms for forming associative structures that support episodic memory remain underexplored. Inspired by hippocampal CA3 dynamics and its role in associative memory, we propose the Latent Structured Hopfield Network (LSHN), a biologically inspired framework that integrates continuous Hopfield attractor dynamics into an autoencoder architecture. LSHN mimics the cortical–hippocampal pathway: a semantic encoder extracts compact latent representations, a latent Hopfield network performs associative refinement through attractor convergence, and a decoder reconstructs perceptual input. Unlike traditional Hopfield networks, our model is trained end-to-end with gradient descent, achieving scalable and robust memory retrieval. Experiments on MNIST, CIFAR-10, and a simulated episodic memory task demonstrate superior performance in recalling corrupted inputs under occlusion and noise, outperforming existing associative memory models. Our work provides a computational perspective on how semantic elements can be dynamically bound into episodic memory traces through biologically grounded attractor mechanisms.


### Installation

Download this repository.

```
git clone https://github.com/fudan-birlab/LSHN.git
cd LSHN
```

Create the environment for LSHN.

```
conda create -n LSHN python=3.10
conda activate LSHN
pip install -r requirements.txt
```

### Getting Start

1. Training

```
python -u train.py --dataset_name CIFAR10 --emb_size 128 --num_pattern 100 --random_idx 0 
```

2. Evaluation

See `evaluation.ipynb`.

### Results

Results in CIFAR10 & MNIST datasets. More results refer to the paper.

<img src=".\assets\result-cifar10.jpeg" alt="eval-results" style="zoom:25%;" />

<img src=".\assets\result-mnist.jpeg" alt="within-Wen" style="zoom:25%;" />

### Citation

If you find our work useful to your research, please consider citing:

```
@misc{li2025latentstructuredhopfieldnetwork,
      title={Latent Structured Hopfield Network for Semantic Association and Retrieval}, 
      author={Chong Li and Xiangyang Xue and Jianfeng Feng and Taiping Zeng},
      year={2025},
      eprint={2506.01303},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.01303}, 
}
```