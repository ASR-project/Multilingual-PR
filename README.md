# Multilingual-PR

Implementation of the project ```Multi-lingual Phoneme Recognition using self-supervised methods on foreign languages```

> Authors: [Apavou Clément](https://github.com/clementapa) & [Belkada Younes](https://github.com/younesbelkada) & [Leo Tronchon](https://github.com/leot13) & [Arthur Zucker](https://github.com/ArthurZucker)

![Python](https://img.shields.io/badge/Python-green.svg?style=plastic)
![PyTorch](https://img.shields.io/badge/PyTorch-orange.svg?style=plastic)
![PyTorch Lightning](https://img.shields.io/badge/PyTorch-Lightning-blueviolet.svg?style=plastic)

## Project structure

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
- [X] metric per
- [x] Wandb display using Hugging Face ? :
    - [x] Phoneme Error Rate (train and validation)
    - [x] Loss values (train and validation)
    - [x] some validation audio files with phoneme in labels and predictions ?
- [ ] Get the features from a pre-trained model (Wav2Vec, HuBert and WavLM) on HuggingFace on the retrieved dataset
    - [x] Wav2Vec : https://huggingface.co/facebook/wav2vec2-base https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/wav2vec2#overview
    - [ ] HuBert : https://huggingface.co/docs/transformers/model_doc/hubert https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/hubert#overview
    - [ ] WavLM : https://huggingface.co/microsoft/wavlm-base https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/wavlm#overview
- [X] Split the dataset into a trainval / test set. Make sure that the speakers do not occur both on the train set and test set -> **Already done in HF**

### Modeling part

- [X] Implement CTC algorithm using PyTorch
- [X] Metric : implement the Phoneme Error Rate
- [ ] Train the model on the built dataset and using pretrained features of different SSL method
- [ ] Train on 10 minutes, 1 hour and 10 hours of data
- [ ] Benchmark for different languages

### language that we can done with annotation available
Dutch (du), Spanish (es), French (fr), Italian (it), Kyrgyz (ky), Russian (ru), Sweedish
(sv), Turkish (tr), Tatar (tt) and Mandarin (zh).

### Benchmark

dataset: Common Voice Corpus 6.1 : https://commonvoice.mozilla.org/fr/datasets 

| Language | Model | PER validation | PER test | Training time of data | Run |
|---|---|---|---|---|---|
| Sweedish | Wav2Vec2 | X | X | X | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr?workspace=user-clementapa) |
| Sweedish | WavLM | X | X | X |[![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr?workspace=user-clementapa) |
| Sweedish | Hubert | X | X | X |[![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr?workspace=user-clementapa) |
| Italian | Wav2Vec2 | X | X | X | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr?workspace=user-clementapa) |
| Italian | WavLM | X | X | X | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr?workspace=user-clementapa) |
| Italian | Hubert | X | X | X | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr?workspace=user-clementapa) |
| ... | Wav2Vec2 | X | X | X | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr?workspace=user-clementapa) |
| ... | WavLM | X | X | X | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr?workspace=user-clementapa) |
| ... | Hubert | X | X | X | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/asr-project/test-asr?workspace=user-clementapa) |

### Running unit tests

- [ ] Add an automatic push to the main branch if the tests are successful. Otherwise don't. This will allow us to merge with main as soon as it is up and running.
- [ ] Add dataset and model tests, which take care of asserts on the datasizes etc. 
Refer to the directory ```tests``` and you can write your own testing function. The function name has to start with ```test_```


## ToDo docker 
- [ ] Add a workflow to automatically build and push the docker. 
- [ ] Add a workflow to run a training on AZURE using their gpus 

