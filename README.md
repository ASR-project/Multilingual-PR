# Multilingual-PR

Implementation of the project ```Multi-lingual Phoneme Recognition using self-supervised methods on foreign languages```

> Authors: [Apavou Clément](https://github.com/clementapa) & [Belkada Younes](https://github.com/younesbelkada) & [Leo Tronchon](https://github.com/leot13) & [Arthur Zucker](https://github.com/ArthurZucker)

![Python](https://img.shields.io/badge/Python-green.svg?style=plastic)
![PyTorch](https://img.shields.io/badge/PyTorch-orange.svg?style=plastic)
![PyTorch Lightning](https://img.shields.io/badge/PyTorch-Lightning-blueviolet.svg?style=plastic)

<p align="center">
  <img width="300" height="300" src="assets/img_readme/parrot.png">
</p>

This repository is powered by HuggingFace :hugs:,  Pytorch-Lightning and Weight & Biases.

## :bird: Introduction 

TODO (wait to do correctly in the report)

## :sparkles: Main features

+ Modularity between SOTA models in self-supervision for speech
+ Freedom to select any languages available on CommonVoice hosted at [HuggingFace](https://huggingface.co/datasets/common_voice). 
+ Nice visualization tool through wandb.

## :pencil2: Schema  

<p align="center">
  <img width="400" height="500" src="assets/img_readme/Network.drawio.png">
</p>
<p align="center">
  <em> Diagram of the models used for the experiments. N=22 and h=1024 for HuBERT, and N=11 and h=768 for Wav2vec2 and WavLM. Made by us. </em>
</p>

## :books: Language that we can done with phonemes dictionaries available
Dutch (du), Spanish (es), French (fr), Italian (it), Kyrgyz (ky), Russian (ru), Sweedish
(sv), Turkish (tr), Tatar (tt) and Mandarin (zh). From https://github.com/facebookresearch/CPC_audio.

## :sound: Dataset

The project is based on [Mozilla CommonVoice dataset](https://commonvoice.mozilla.org/fr) available on [HuggingFace](https://huggingface.co/datasets/common_voice). 
When the script is launched, the program will automatically download the correct dataset and try to transform ground truth sentences to phonemes using [phonemizer](https://github.com/bootphon/phonemizer). You are free to chose any dataset available on HuggingFace with phonemes dictionaries previously cited to run your models. For our experiments we use:
```
it, nl, tr, ru, sv
```
Feel free to try any other languages and submit a Pull Request :electric_plug:.

## :paperclip: Pre-trained model studied

<p align="center">
  <img src="assets/img_readme/wav2vec2.png" width="400" height="200"/>
  <img src="assets/img_readme/hubert.jpeg" width="300" height="300"/>
  <img src="assets/img_readme/wavlm.png" width="400" height="400"/>
</p>
<p align="center">
<em> Schema of Wav2vec2, HuBERT and WavLM. </em>
</p>

For our experiments, we used models hosted on Hugging Face library, that are pre-trained on 960 hours of **English** audio data from Librispeech dataset on 16kHz sampled speech audio. The following pre-trained models were used:
- Wav2vec2:  [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h)
- WavLM: [microsoft/wavlm-base](https://huggingface.co/microsoft/wavlm-base)
- HuBERT: [facebook/hubert-large-ls960-ft](https://huggingface.co/facebook/hubert-large-ls960-ft)
## :scroll: Data processing part

- [X] Explore the dataset on Mozilla common voices (https://commonvoice.mozilla.org/fr) available on HuggingFace? (https://huggingface.co/datasets/common_voice & https://huggingface.co/mozilla-foundation)
- [X] Script to transform sentence to phoneme (phonemizer : https://github.com/bootphon/phonemizer) : sentence_to_phoneme.py (language available : https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md)
- [X] Tokenize labels to apply CTC
- [X] metric per
- [x] Wandb display using Hugging Face ? :
    - [x] Phoneme Error Rate (train and validation)
    - [x] Loss values (train and validation)
    - [x] some validation audio files with phoneme in labels and predictions ?
- [x] Get the features from a pre-trained model (Wav2Vec, HuBert and WavLM) on HuggingFace on the retrieved dataset
    - [x] Wav2Vec : https://huggingface.co/facebook/wav2vec2-base https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/wav2vec2#overview
    - [x] HuBert : https://huggingface.co/docs/transformers/model_doc/hubert https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/hubert#overview
    - [x] WavLM : https://huggingface.co/microsoft/wavlm-base https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/wavlm#overview
- [X] Split the dataset into a trainval / test set. Make sure that the speakers do not occur both on the train set and test set -> **Already done in HF**

## :pencil: Modeling part

- [X] Implement CTC algorithm using PyTorch
- [X] Metric : implement the Phoneme Error Rate
- [x] Train the model on the built dataset and using pretrained features of different SSL method
- [ ] Train on 10 minutes, 1 hour and 10 hours of data
- [x] Benchmark for different languages

## :family: Language Family

<center>

| Language | Family |
|---|---|
| Italian :it: | *Romance* |
| Russian :ru: | *East Slavic* |
| Dutch 🇳🇱 | *West Germanic* |
| Swedish 🇸🇪 | *North Germanic* |
| Turkish :tr: | *Turkic* |

</center>

**English** is a part of the *West Germanic* family.\
Source: https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md

## :chart_with_upwards_trend: Main results

dataset: Common Voice Corpus 6.1 : https://commonvoice.mozilla.org/fr/datasets 

Pretrained English models to other languages

### 🚀 Fine-tuning

| Language | Training data (in hours) | Language Family | Model    | PER validation | PER test | Runs                                                                                                                                                              |
|----------|--------------------------|-----------------|----------|----------------|----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Italian  | 62.34                    | Romance         | Wav2Vec2 | 19.05          | 17.95    | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/1y0wqakj?workspace=user-clementapa)|
|          |                          |                 | Hubert   | 14.05          | 12.67    | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/22429a1f?workspace=user-clementapa) |
|          |                          |                 | WavLM    | 19.83          | 25.60    | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/1tozo0p7?workspace=user-clementapa)  |
| Russian  | 15.55                    | East Slavic     | Wav2Vec2 | 32.16          | 31.66    | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/3n11rfhy?workspace=user-clementapa)|
|          |                          |                 | Hubert   | 25.10          | 24.09    | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/109itv4h?workspace=user-clementapa) |
|          |                          |                 | WavLM    | 20.25          | 18.88    | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/1i0j5f2r?workspace=user-clementapa)  |
| Dutch    | 12.78                    | West Germanic   | Wav2Vec2 | 16.18          | 20.83    | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/22slhrhk?workspace=user-clementapa) |
|          |                          |                 | Hubert   | 12.77          | 16.49    | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/1d4o8yug?workspace=user-clementapa)  |
|          |                          |                 | WavLM    | 15.96          | 19.91    | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/x9orkmct?workspace=user-clementapa)  |
| Swedish  | 3.22                     | North Germanic  | Wav2Vec2 | 26.50          | 24.16    | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/1cm9q4ud?workspace=user-clementapa) |
|          |                          |                 | Hubert   | 21.77          | 19.38    | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/1ztn3i01?workspace=user-clementapa)   |
|          |                          |                 | WavLM    | 26.86          | 24.61    | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/2ofpwcgv?workspace=user-clementapa)   |
| Turkish  | 2.52                     | Turkic          | Wav2Vec2 | 19.62          | 19.03    | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/3ebdnaq9?workspace=user-clementapa) |
|          |                          |                 | Hubert   | 15.51          | 14.19    | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/3jp8et3b?workspace=user-clementapa)   |
|          |                          |                 | WavLM    | 19.85          | 18.95    | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/2ldnghcw?workspace=user-clementapa)   |
| Average  | -                        | All             | Wav2Vec2 | 22.70          | 22.73    |                                                                                                                                                                  |
|          |                          |                 | Hubert   | 17.84          | 17.36    |                                                                                                                                                                   |
|          |                          |                 | WavLM    | 20.55          | 21.59    |                                                                                                                                                                   |


### 🧊 Frozen Features

| Language | Training data (in hours) | Language Family | Model    | PER validation | PER test | Runs |
|----------|--------------------------|-----------------|----------|----------------|----------|------|
| Italian :it:| 62.34                    | Romance         | Wav2Vec2 | 38.94          | 36.84    | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/1wkydddw?workspace=user-clementapa)      |
|          |                          |                 | Hubert   | 23.85          |  21.15   | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/2esx3e99?workspace=user-clementapa)      |
|          |                          |                 | WavLM    | 27.29          | 25.98    | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/2771pb3w?workspace=user-clementapa)      |
| Russian :ru:| 15.55                    | East Slavic     | Wav2Vec2 | 50.11          | 48.69    | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/tqxh9iho?workspace=user-clementapa)      |
|          |                          |                 | Hubert   | 38.36          | 36.18    | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/1vcphgy6?workspace=user-clementapa)      |
|          |                          |                 | WavLM    | 40.66          | 38.76    | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/3em0h48e?workspace=user-clementapa)      |
| Dutch 🇳🇱 | 12.78                    | West Germanic   | Wav2Vec2 | 40.15          | 39.23    | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/3qiea6qt?workspace=user-clementapa)      |
|          |                          |                 | Hubert   | 27.62          | 26.68    | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/13nprcb5?workspace=user-clementapa)      |
|          |                          |                 | WavLM    | 34.94          | 35.67    | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/204zyxrk?workspace=user-clementapa)      |
| Swedish 🇸🇪| 3.22                     | North Germanic  | Wav2Vec2 | 50.30          | 45.23    | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/2v7why5t?workspace=user-clementapa)      |
|          |                          |                 | Hubert   | 37.34          | 32.68    | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/mgt5ofzn?workspace=user-clementapa)      |
|          |                          |                 | WavLM    | 43.65          | 40.55    | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/qt9ymhm1?workspace=user-clementapa)      |
| Turkish :tr:| 2.52                     | Turkic          | Wav2Vec2 | 53.92          | 52.08    | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/1kyc217g?workspace=user-clementapa)      |
|          |                          |                 | Hubert   | 39.55          | 37.08    | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/9flufqqm?workspace=user-clementapa)      |
|          |                          |                 | WavLM    | 47.18          | 45.53    | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/x9gqzh3q?workspace=user-clementapa) 
| Average  |                          | All             | Wav2Vec2 | 46.684         | 44.414   | -    |
|          |                          |                 | Hubert   | 33.344         | 30.754   | -    | 
|          |                          |                 | WavLM    | 38.744         | 37.298   | -    | 

### ⌚ Training data  

| Training set | Training data | Model    | PER validation | PER test | Runs |
|--------------|---------------|----------|----------------|----------|------|
| 5%           | ~ 10 min      | Wav2Vec2 | 55.35          | 50.91    |[![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/3ti8zmeo?workspace=user-clementapa)       |
|              |               | Hubert   | 44.96          | 39.38    |[![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/259nzpmx?workspace=user-clementapa)       |
|              |               | WavLM    | 56.22          | 51.25    |[![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/103hww9n?workspace=user-clementapa)       |
| 10%          | ~ 20 min      | Wav2Vec2 | 52.97          | 49.01    |[![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/1e0h39p9?workspace=user-clementapa)       |
|              |               | Hubert   | 42.61          | 37.50    |[![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/2a975lju?workspace=user-clementapa)       |
|              |               | WavLM    | 46.54          | 43.64    |[![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/oh1g6apq?workspace=user-clementapa)       |
| 50%          | ~ 2 h         | Wav2Vec2 | 51.23          | 46.24    |[![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/tq9rqz7g?workspace=user-clementapa)       |
|              |               | Hubert   | 39.91          | 35.27    |[![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/32cckeib?workspace=user-clementapa)       |
|              |               | WavLM    | 44.57          | 42.33    |[![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/103hww9n?workspace=user-clementapa)       |
| 100%         | ~ 3 h         | Wav2Vec2 | 50.30          | 45.23    |[![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/2v7why5t?workspace=user-clementapa)       |
|              |               | Hubert   | 37.34          | 32.68    |[![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/mgt5ofzn?workspace=user-clementapa)       |
|              |               | WavLM    | 43.65          | 40.55    |[![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr/runs/x9gqzh3q?workspace=user-clementapa)       |


## :pushpin: Project structure

```
├── agents
|   ├── BaseTrainer.py       
|   
├── assets                      # database and vocab phonemes are put here
|
├── config
|   ├── hparams.py              # configuration file
|
├── Datasets
|   |
|   ├── datamodule.py           # datamodules PyTorch lightning for CommonVoice dataset
|          
├── models
|   ├── BaseModule.py           #  lightning module 
|   ├── models.py               # Wav2vec2 WavLM and Hubert using Hugging Face library
| 
├── utils                       # utils functions
|   ├── agent_utils.py
|   ├── callbacks.py
|   ├── dataset_utils.py
|   ├── logger.py
|   ├── metrics.py              
|   ├── per.py                  # torch metrics implementation of the phoneme error rate
|
├── hparams.py                   # configuration file
|
├── main.py                      # main script to launch for training of inference 
|
└── README.md
```

## ⚡ Powered by

 <p align="center">
    <a href="https://huggingface.co/">
    <img src="https://raw.githubusercontent.com/huggingface/awesome-huggingface/main/logo.svg" width="10%" height="10%" style="margin-right: 50px;" alt="logo hugging face"/>
    <a href="https://wandb.ai/site">
    <img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-dots-logo.svg"  style="margin-right: 50px;" width="10%" height="10%" alt="logo wandb"/>
    <a href="https://pytorch-lightning.readthedocs.io/en/latest/">
    <img src="https://github.com/PyTorchLightning/pytorch-lightning/blob/master/docs/source/_static/images/logo.png" width="25%" height="25%" alt="logo pytorch lightning"/>

</p>
