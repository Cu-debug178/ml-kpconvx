# KPConvX: Modernizing Kernel Point Convolution with Kernel Attention

<div align='center'>
<img src="assets/fig_kpconvx.png" alt="teaser" width="1200" />
</div>

<div align="center">
  <a href="https://paperswithcode.com/sota/semantic-segmentation-on-s3dis-area5?p=kpconvx-modernizing-kernel-point-convolution">
    <img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/kpconvx-modernizing-kernel-point-convolution/semantic-segmentation-on-s3dis-area5" alt="PWC">
  </a>
  <a href="https://paperswithcode.com/sota/semantic-segmentation-on-scannet?p=kpconvx-modernizing-kernel-point-convolution">
    <img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/kpconvx-modernizing-kernel-point-convolution/semantic-segmentation-on-scannet" alt="PWC">
  </a>
  <a href="https://paperswithcode.com/sota/3d-point-cloud-classification-on-scanobjectnn?p=kpconvx-modernizing-kernel-point-convolution">
    <img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/kpconvx-modernizing-kernel-point-convolution/3d-point-cloud-classification-on-scanobjectnn" alt="PWC">
  </a>
</div>

<div align="center">
  <b>
    <a href="https://machinelearning.apple.com/research/kpconvx">[Blog]</a>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://arxiv.org/pdf/2405.13194">   [ArXiv]</a>
  </b>
</div>

This repo is the official project repository of the paper **KPConvX: Modernizing Kernel Point Convolution with Kernel Attention**.


## Highlights

- Nov 14, 2024: We release the trained models for ScanObjectNN and S3DIS datasets
- Nov 14, 2024: KPConvX repository is officially realeased. We provide two implementation for KPConvX, an standalone training pipeline, and wrappers to use it in the Pointcept library.
- Feb 28, 2024: KPConvX is accepted by CVPR 2024! 🎉 


## Setup

The setup instructions are different if you plan to use the Standalone or the Pointcept version:
- [Setup the standalone version](./Standalone/).
- [Setup the Pointcept version](./Pointcept-wrapper/).


## Experiment reports

Lightweight, versioned training and evaluation summaries are indexed in
[experiment_reports](./experiment_reports/). Checkpoints, datasets, raw logs,
and other large runtime artifacts are intentionally kept outside Git.


## Citation
If you found this code useful, please cite the following paper:
```
@inproceedings{thomas2024kpconvx,
  title={KPConvX: Modernizing Kernel Point Convolution with Kernel Attention},
  author={Thomas, Hugues and Tsai, Yao-Hung Hubert and Barfoot, Timothy D and Zhang, Jian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5525--5535},
  year={2024}
}
```

## Acknowledgements
Our codebase is built using multiple opensource contributions, please see [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS) for more details. 
