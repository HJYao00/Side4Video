<div align="center">

<h1> Side4Video: Spatial-Temporal Side Network for Memory-Efficient Image-to-Video Transfer Learning </h1>

<h5 align="center"> 
  
<!--[![arXiv](https://img.shields.io/badge/Arxiv-2311.15769-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2311.15769)


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/side4video-spatial-temporal-side-network-for/action-recognition-in-videos-on-something-1)](https://paperswithcode.com/sota/action-recognition-in-videos-on-something-1?p=side4video-spatial-temporal-side-network-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/side4video-spatial-temporal-side-network-for/action-classification-on-kinetics-400)](https://paperswithcode.com/sota/action-classification-on-kinetics-400?p=side4video-spatial-temporal-side-network-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/side4video-spatial-temporal-side-network-for/video-retrieval-on-vatex)](https://paperswithcode.com/sota/video-retrieval-on-vatex?p=side4video-spatial-temporal-side-network-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/side4video-spatial-temporal-side-network-for/action-recognition-in-videos-on-something)](https://paperswithcode.com/sota/action-recognition-in-videos-on-something?p=side4video-spatial-temporal-side-network-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/side4video-spatial-temporal-side-network-for/video-retrieval-on-msr-vtt-1ka)](https://paperswithcode.com/sota/video-retrieval-on-msr-vtt-1ka?p=side4video-spatial-temporal-side-network-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/side4video-spatial-temporal-side-network-for/video-retrieval-on-msvd)](https://paperswithcode.com/sota/video-retrieval-on-msvd?p=side4video-spatial-temporal-side-network-for) -->

</h5>
</div>

This repository is the official implementation of Side4Video, which significantly reduces the training memory cost for action recognition and text-video retrieval tasks.
<div align=center>
<img width="500" alt="image" src="imgs/mem.png">
</div>

<!--[![Paper](http://img.shields.io/badge/Paper-arxiv.2307.08908-b31b1b.svg)](https://arxiv.org/abs/2307.08908)-->

## 🗺️ Overview
<!--[The motivation of Side4Video is to reduce the training cost, enabling us to train a larger model with limited resources.-->

<div align=center>
<img width="795" alt="image" src="imgs/Side4Video.png">
</div>

<!-- ![Side4Video](imgs/Side4Video.png) -->

## 🚀 Training and Testing
For training and testing our model, please refer to the Recognition and Retrieval folders.

## 📊 Results
<div align=center>
<img width="800" alt="image" src="imgs/memory.png">
</div>
Our best model can achieve an accuracy of 67.3% & 74.6 on Something-Something V1 & V2, 88.6% on Kinetics-400 and a Recall@1 of 52.3% on MSR-VTT, 56.1% on MSVD, 68.8% on VATEX.

