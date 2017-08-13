# neural-vqa-attention

Torch implementation of an attention-based visual question answering model ([Stacked Attention Networks for Image Question Answering, Yang et al., CVPR16][1]).

![Imgur](http://i.imgur.com/VbqIRZz.png)

1. [Train your own network](#train-your-own-network)
    1. [Extract image features](#extract-image-features)
    2. [Preprocess VQA dataset](#preprocess-vqa-dataset)
    3. [Training](#training)
2. [Use a pretrained model](#use-a-pretrained-model)
    1. [Pretrained models and data files](#pretrained-models-and-data-files)
    2. [Running evaluation](#running-evaluation)
3. [Results](#results)

Intuitively, the model looks at an image, reads a question, and comes up with an answer to the question and a heatmap of where it looked in the image to answer it.

The model/code also supports referring back to the image multiple times ([Stacked Attention][1]) before producing the answer. This is supported via a `num_attention_layers` parameter in the code (default = 1).

**NOTE**: This is NOT a state-of-the-art model. Refer to [MCB][7], [MLB][8] or [HieCoAtt][9] for that.
This is a simple, somewhat interpretable model that gets decent accuracies and produces [nice-looking results](#results).
The code was written about ~1 year ago as part of [VQA-HAT][12], and I'd meant to release it earlier, but couldn't get around to cleaning things up.

If you just want to run the model on your own images, download links to pretrained models are given below.

## Train your own network

### Preprocess VQA dataset

Pass `split` as `1` to train on `train` and evaluate on `val`, and `2` to train on `train`+`val` and evaluate on `test`.

```
cd data/
python vqa_preprocessing.py --download True --split 1
cd ..
```
```
python prepro.py --input_train_json data/vqa_raw_train.json --input_test_json data/vqa_raw_test.json --num_ans 1000
```

### Extract image features

Since we don't finetune the CNN, training is significantly faster if image features are pre-extracted. We use image features from VGG-19. The model can be downloaded and features extracted using:

```
sh scripts/download_vgg19.sh
th prepro_img.lua -image_root /path/to/coco/images/ -gpuid 0
```

### Training

```
th train.lua
```

## Use a pretrained model

### Pretrained models and data files

All files available for download [here][10].

- `san1_2.t7`: model pretrained on `train`+`val` with 1 attention layer (SAN-1)
- `san2_2.t7`: model pretrained on `train`+`val` with 2 attention layers (SAN-2)
- `params_1.json`: vocabulary file for training on `train`, evaluating on `val`
- `params_2.json`: vocabulary file for training on `train`+`val`, evaluating on `test`
- `qa_1.h5`: QA features for training on `train`, evaluating on `val`
- `qa_2.h5`: QA features for training on `train`+`val`, evaluating on `test`
- `img_train_1.h5` & `img_test_1.h5`: image features for training on `train`, evaluating on `val`
- `img_train_2.h5` & `img_test_2.h5`: image features for training on `train`+`val`, evaluating on `test`

### Running evaluation

```
model_path=checkpoints/model.t7 qa_h5=data/qa.h5 params_json=data/params.json img_test_h5=data/img_test.h5 th eval.lua
```

This will generate a JSON file containing question ids and predicted answers. To compute accuracy on `val`, use [VQA Evaluation Tools][13]. For `test`, submit to [VQA evaluation server on EvalAI][14].

## Results

**Format**: sets of 3 columns, col 1 shows original image, 2 shows 'attention' heatmap of where the model looks, 3 shows image overlaid with attention. Input question and answer predicted by model are shown below examples.
![](http://i.imgur.com/Q0byOyp.jpg)

More results available [here][3].

### Quantitative Results

Trained on `train` for `val` accuracies, and trained on `train`+`val` for `test` accuracies.

#### VQA v2.0

| Method                | val     | test    |
| ------                | ---     | ----    |
| SAN-1                 | 53.15   | 55.28   |
| SAN-2                 | 52.82   | -       |
| [d-LSTM + n-I][4]     | 51.62   | 54.22   |
| [HieCoAtt][9]         | 54.57   | -       |
| [MCB][7]              | 59.14   | -       |

#### VQA v1.0

| Method                | test-std    |
| ------                | --------    |
| SAN-1                 | 59.87       |
| SAN-2                 | 59.59       |
| [d-LSTM + n-I][4]     | 58.16       |
| [HieCoAtt][9]         | 62.10       |
| [MCB][7]              | 65.40       |

## References

- [Stacked Attention Networks for Image Question Answering][1], Yang et al., CVPR16
- [Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering][11], Goyal and Khot et al., CVPR17
- [VQA: Visual Question Answering][6], Antol et al., ICCV15


## Acknowledgements

- Data preprocessing script borrowed from [VT-vision-lab/VQA_LSTM_CNN][4]

## License

[MIT][2]


[1]: https://arxiv.org/abs/1511.02274
[2]: https://abhshkdz.mit-license.org/
[3]: https://computing.ece.vt.edu/~abhshkdz/neural-vqa-attention/figures/
[4]: https://github.com/VT-vision-lab/VQA_LSTM_CNN
[5]: http://visualqa.org/download.html
[6]: http://arxiv.org/abs/1505.00468
[7]: https://github.com/akirafukui/vqa-mcb
[8]: https://github.com/jnhwkim/MulLowBiVQA
[9]: https://github.com/jiasenlu/HieCoAttenVQA
[10]: https://computing.ece.vt.edu/~abhshkdz/neural-vqa-attention/pretrained/
[11]: https://arxiv.org/abs/1612.00837
[12]: https://computing.ece.vt.edu/~abhshkdz/vqa-hat/
[13]: https://github.com/VT-vision-lab/VQA
[14]: https://evalai.cloudcv.org/featured-challenges/1/overview
