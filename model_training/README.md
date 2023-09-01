# ShuffleFaceNet Pytorch

A PyTorch Implementation of ShuffleFaceNet using CosFace Loss and Complexity 0.5x. The code can be trained on CASIA-Webface and tested on LFW.

[ShuffleFaceNet: A Lightweight Face Architecture for Efficientand Highly-Accurate Face Recognition](http://openaccess.thecvf.com/content_ICCVW_2019/papers/LSR/Martindez-Diaz_ShuffleFaceNet_A_Lightweight_Face_Architecture_for_Efficient_and_Highly-Accurate_Face_ICCVW_2019_paper.pdf)

## Prerequisites

Download the two training datasets with this command:

```sh
python get_datasets.py
```

It will download the CASIA WebFace and Labelled Faces in the Wild (LFW) datasets into *DATASETS* folder.

## Training

To run trainng you can use the `train.py` script. In the `config.py` file you can adjust the settings to your needs.

## Inference

Inside the inference folder you have a python script which runs Face Detection, based on Blaze Face, and after FaceID based on a Shuffle Face Net trained with this repository.

To run the inference on sample images:

```sh
cd inference
python inference_blazeface.py
```

## Convert to ONNX

The script `checkpoint2ONNX.py` converts the trained pytorch model into ONNX format to be then imported onto NNTool and generate Gap9 code. 


## References
[ShuffleNet](https://github.com/kuangliu/pytorch-cifar/blob/master/models/shufflenet.py)

[CosFace](https://github.com/YirongMao/softmax_variants/blob/master/model_utils.py)
