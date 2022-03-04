# Multilingual-PR

Implementation of the project ```Multi-lingual Phoneme Recognition using self-supervised methods on foreign languages```

> Authors: [Apavou Clément](https://github.com/clementapa) & [Belkada Younes](https://github.com/younesbelkada) & [Leo Tronchon](https://github.com/leot13) & [Arthur Zucker](https://github.com/ArthurZucker)

![Python](https://img.shields.io/badge/Python-green.svg?style=plastic)
![PyTorch](https://img.shields.io/badge/PyTorch-orange.svg?style=plastic)
![PyTorch Lightning](https://img.shields.io/badge/PyTorch-Lightning-blueviolet.svg?style=plastic)

## Project structure

```
├── assets                      # Put database here
├── datamodules
|   |
|   ├── commonvoice_datamodule.py     # datamodules PyTorch lightning for CommonVoice dataset
|         
├── datasets
|   ├── commonvoice.py                # CommonVoice dataset HF
|          
├── lightningmodules
|   ├── classification.py        # lightning module for image classification (multi-label)
| 
├── utils                        # utils functions
|   ├── callbacks.py
|   ├── utils_functions.py
|
├── weights                     # put models weights here
|
├── analyse_score_latent_space.ipynb  # notebook to analyse scores predicted
|
├── hparams.py                   # configuration file
|
├── main.py                      # main script to launch for training of inference 
|
└── README.md
```

### Useful Links
https://huggingface.co/datasets/common_voice
https://huggingface.co/mozilla-foundation

https://pytorch.org/audio/stable/datasets.html#commonvoice

https://huggingface.co/datasets/superb
SUPERB: https://arxiv.org/abs/2110.13900
https://superbbenchmark.org/

WavLM: https://arxiv.org/abs/2105.01051
github WavLM: https://github.com/microsoft/unilm

### Data processing part

**Goal:** Create a script that takes as input
- the language
- the model type
- the split 
 
And push it to wandb as an artifact


- [X] Explore the dataset on Mozilla common voices (https://commonvoice.mozilla.org/fr) available on HuggingFace? (https://huggingface.co/datasets/common_voice & https://huggingface.co/mozilla-foundation)
- [X] Script to transform sentence to phoneme (phonemizer : https://github.com/bootphon/phonemizer) : sentence_to_phoneme.py (language available : https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md)
- [X] Tokenize labels to apply CTC
- [ ] use load_metric HF for metrics : load_metric("wer") load_metric("per")
- [ ] Wandb display using Hugging Face ? :
    - [ ] Phoneme Error Rate (train and validation)
    - [ ] Loss values (train and validation)
    - [ ] some validation audio files with phoneme in labels and predictions ?
- [ ] Get the features from a pre-trained model (Wav2Vec, HuBert and WavLM) on HuggingFace on the retrieved dataset
    - [ ] Wav2Vec : https://huggingface.co/facebook/wav2vec2-base https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/wav2vec2#overview
    - [ ] HuBert : https://huggingface.co/docs/transformers/model_doc/hubert https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/hubert#overview
    - [ ] WavLM : https://huggingface.co/microsoft/wavlm-base https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/wavlm#overview
- [X] Split the dataset into a trainval / test set. Make sure that the speakers do not occur both on the train set and test set -> **Already done in HF**

### Modeling part

- [ ] Implement CTC algorithm using PyTorch
- [ ] Metric : implement the Phoneme Error Rate
- [ ] Train the model on the built dataset and using pretrained features of different SSL method
- [ ] Train on 10 minutes, 1 hour and 10 hours of data
- [ ] Benchmark for different languages

### Evaluation

- [ ] Evaluate the model on custom test set built at the step 1

# Running unit tests

- [ ] Add an automatic push to the main branch if the tests are successful. Otherwise don't. This will allow us to merge with main as soon as it is up and running.
- [ ] Add dataset and model tests, which take care of asserts on the datasizes etc. 
Refer to the directory ```tests``` and you can write your own testing function. The function name has to start with ```test_```
