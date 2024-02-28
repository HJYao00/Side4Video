# Side4Video for Text-Video Retrieval
Official implementation of Side4Video for Text-Video Retrieval

## Requirement
```
conda install --yes -c pytorch pytorch=1.8.1 torchvision cudatoolkit=11.1
pip install ftfy regex tqdm
pip install opencv-python boto3 requests pandas
```

## Data Preparation
All video datasets can be downloaded from respective official links. To enhance efficiency, we directly utilize frames provided by [Cap4Video](https://github.com/whwu95/Cap4Video) for training.
| Dataset | Official Link| Ours|
|:------------:|:-------------------:|:------------------:|
| MSRVTT | [Video](http://ms-multimedia-challenge.com/2017/dataset)| [Frames](https://unisyd-my.sharepoint.com/:u:/g/personal/wenhao_wu_sydney_edu_au/EQEYklCTUClGu01komekxcgBQ5lxeInfRm-fhlikMyb8hA?e=11DEyO) | 
| MSVD | [Video](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/)| [Frames](https://unisyd-my.sharepoint.com/:u:/g/personal/wenhao_wu_sydney_edu_au/EUdl9tM7TRlFsWqLC4V3ffUBAvqIrcUBXHbLEE4p3SiQVQ?e=W3iYZi)| 
| VATEX | [Video](https://eric-xw.github.io/vatex-website/download.html)| [Frames](https://unisyd-my.sharepoint.com/:u:/g/personal/wenhao_wu_sydney_edu_au/EQd5BwA_bcFBn7SRl0D69XABO4xveLZtu6PUK_DQKEyxfg?e=D7kGBQ) | 


## Train
- After Data Preparation, you will need to download the CLIP weights from [OpenAI](https://github.com/openai/CLIP?tab=readme-ov-file) or [EVA](https://github.com/baaivision/EVA/tree/master/EVA-CLIP), and place them in the `clip_pretrain` folder.
```sh
# clip_pretrain /
# |-- ViT-B-32.pt /
# |-- ViT-B-16.pt 
# For example, fine-tuning on MSR-VTT using the following command:
sh script_train/run_train_msrvtt.sh
```

## Test
- Replace the path in `--init_model` and run the command to test the model.
```sh
sh script_test/run_test_msrvtt.sh
```
