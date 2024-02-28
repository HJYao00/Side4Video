# Side4Video for Action Recognition
Official implementation of Side4Video for Video Action Recognition
## Requirement
- PyTorch >= 1.8
- RandAugment
- pprint
- dotmap
- yaml

## Data Preparation
**(Recommend)** To train all of our models, we extract videos into frames for fast reading. Please refer to [MVFNet](https://github.com/whwu95/MVFNet/blob/main/data_process/DATASETS.md) repo for the detaied guide of data processing.  
The annotation file is a text file with multiple lines, and each line indicates the directory to frames of a video, total frames of the video and the label of a video, which are split with a whitespace. Here is the format: 
```sh
abseiling/-7kbO0v4hag_000107_000117 300 0
abseiling/-bwYZwnwb8E_000013_000023 300 0
```

**(Optional)** We can also decode the videos in an online fashion using [decord](https://github.com/dmlc/decord). This manner should work but are not tested. All of the models offered have been trained using offline frames. Example of annotation:
```sh
abseiling/-7kbO0v4hag_000107_000117.mp4 0
abseiling/-bwYZwnwb8E_000013_000023.mp4 0
```


## Train
- After Data Preparation, you will need to download the CLIP weights from [OpenAI](https://github.com/openai/CLIP?tab=readme-ov-file) or [EVA](https://github.com/baaivision/EVA/tree/master/EVA-CLIP), and place them in the `clip_pretrain` folder. Finally, modify the path here
```sh
# clip_pretrain /
# |-- ViT-B-32.pt /
# |-- ViT-B-16.pt 
# For example, fine-tuning on Something-Something V1 using the following command:
sh scripts/run_train_vision.sh configs/sthv1/sthv1_train_rgb_vitb-16-side4video.yaml
```

## 📱 Model Zoo

Here we provide some off-the-shelf pre-trained checkpoints of our models in the following tables. More checkpoints will be provided soon.

*#Frame = #input_frame x #spatial crops x #temporal clips*
#### Kinetics-400

| Backbone |#Frame |  Top-1 Acc.(%) | checkpoint |
|:------------:|:-------------------:|:------------------:|:-----------------:|
| ViT-B/16 | 8x3x4 | 83.6 | [OneDrive](https://github.com/whwu95/Text4Vis/releases/download/v1/k400-vitb-32-f8.pt) |

#### Something-Something V1

| Backbone |#Frame |  Top-1 Acc.(%) | checkpoint |
|:------------:|:-------------------:|:------------------:|:-----------------:|
| ViT-B/16 | 8x3x4 | 59.4 | [OneDrive](https://github.com/whwu95/Text4Vis/releases/download/v1/k400-vitb-32-f8.pt) |

#### Something-Something V2

| Backbone |#Frame |  Top-1 Acc.(%) | checkpoint |
|:------------:|:-------------------:|:------------------:|:-----------------:|
| ViT-B/16 | 8x3x4 | 70.6 | [OneDrive](https://github.com/whwu95/Text4Vis/releases/download/v1/k400-vitb-32-f8.pt) |



## Test
- Replace the path in `--init_model` and run the command to test the model.
```sh
sh scripts/run_test_vision.sh configs/sthv1/sthv1_train_rgb_vitb-16-side4video.yaml exp_onehot/ssv1/model_best.pt --test_crops 3 --test_clips 2
```


## 🖇️ Citation
If you find this repository is useful, please star🌟 this repo and cite🖇️ our paper.
```bibtex
@article{yao2023side4video,
  title={Side4Video: Spatial-Temporal Side Network for Memory-Efficient Image-to-Video Transfer Learning},
  author={Yao, Huanjin and Wu, Wenhao and Li, Zhiheng},
  journal={arXiv preprint arXiv:2311.15769},
  year={2023}
}
```
