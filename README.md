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

### Extract image features

Since we don't finetune the CNN, training is significantly faster if image features are pre-extracted. We use image features from VGG-19. The model can be downloaded and features extracted using:

```
sh scripts/download_vgg19.sh
th prepro_img.lua -image_root /path/to/coco/images/ -gpuid 0
```

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

### Training

```
th train.lua
```

## Use a pretrained model

### Pretrained models and data files

All files available for download [here][10].
- `san1_120000.t7`: pretrained model with 1 attention layer
- `san2_150000.t7`: pretrained model with 2 attention layers
- `params.json`: vocabulary file
- `img_train.h5` & `img_test.h5` (optional): extracted COCO image features using VGG-19
- `qa.h5` (optional): extracted VQA v1.9 features

### Running evaluation

TODO

## Results

**Format**: sets of 3 columns, col 1 shows original image, 2 shows 'attention' heatmap of where the model looks, 3 shows image overlaid with attention. Input question and answer predicted by model are shown below examples.
![](http://i.imgur.com/Q0byOyp.jpg)

More results available [here][3].

### Quantitative Results

Trained on [VQA v1.9][5] `train`, a single attention layer model (SAN-1) gets **53.01** VQA accuracy on `val`.

Other available VQA models are evaluated on VQA v1.0, which is a significantly easier dataset.
For a point of reference, the [Deeper LSTM + Norm I][4] model from [Antol et al., ICCV15][6] gets 51.52 on VQA v1.9 `val`.

#### VQA v1.0

Trained on [VQA v1.0][6] `train`+`val`, SAN-2 gets **59.59** on `test-std`. This is slightly better than the best reported result in the [paper][1] (58.9). For comparison, [MCB][7] gets ~65.4 (66.7 for ensemble), [MLB][8] gets 65.07, [HieCoAtt][9] gets 62.1 and [Deeper LSTM + Norm I][4] gets 58.16.

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
