import os
import os.path as osp
import pandas as pd
import json
import argparse
import wandb
from tqdm import tqdm

from phonemizer.phonemize import phonemize
# https://github.com/bootphon/phonemizer

from datasets import load_dataset

def main(language_name):
    dataset_name = "common_voice"       # https://huggingface.co/mozilla-foundation or https://huggingface.co/datasets/common_voice
    use_auth_token = False              # True if use mozilla-foundation datasets
    #language_name = "vi"                       # chosen language 
    download_mode = "reuse_dataset_if_exists"
    cache_dir = osp.join(os.getcwd(), "assets")

    dataset = load_dataset(dataset_name, language_name,
                            use_auth_token=use_auth_token, 
                            download_mode=download_mode, 
                            cache_dir=cache_dir)

    dict_res = {k: {} for k in dataset.keys()}

    for subset_name in tqdm(dataset.keys(), position = 0):
        subset = dataset[subset_name]
        for sample in tqdm(subset, position=1):
            sentence = sample['sentence']
            phoneme_sentence = phonemize(sentence, language=language_name)
            dict_res[subset_name][sample['path']] = phoneme_sentence 

    phonemes_path = osp.join(os.getcwd(), "assets", "phoneme_labels")

    if not os.path.exists(phonemes_path):
        os.makedirs(phonemes_path)

    file_dict = os.path.join(phonemes_path, f"{dataset_name}-{language_name}.json")
    with open(file_dict, 'w') as fp:
        json.dump(dict_res, fp)

    # Push artifact
    wandb.init(
        entity="asr-project",
        project="asr"
    )

    artifact = wandb.Artifact(
        name=os.path.basename(file_dict),
        type="dataset",
        metadata={
            "dataset": dataset_name,
            "language": language_name
        },
        description=f"Labels of the sentences on the dataset {dataset_name}-{language_name}"
    )

    artifact.add_file(file_dict)
    wandb.log_artifact(artifact, aliases=["latest"])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Push an artifact to wandb")
    parser.add_argument("--language_name", required=True, type = str, help = "name of the language that you want to convert")
    args = parser.parse_args()
    main(args.language_name)


# FIXME
# from utils.agent_utils import get_progress_bar
# with get_progress_bar() as progress1:
#     i = 0
#     task1 = progress1.add_task(f"[cyan]Split ", total=len(dataset.keys()), info="-")
#     for subset_name in dataset.keys():
#         subset = dataset[subset_name]
#         with get_progress_bar() as progress2:
#             j = 0
#             task2 = progress2.add_task(f"[cyan]Samples ", total=len(subset), info="-")
#             for sample in subset:
#                 sentence = sample['sentence']
#                 phoneme_sentence = phonemize(sentence, language=language_name)
#                 dict_res[subset_name][sample['path']] = phoneme_sentence 

#                 j+=1
#                 progress2.update(task1, advance=1, info=f"{j}/{len(subset)}")
#                 break
#         i+=1
#         progress1.update(task1, advance=1, info=f"{i}/{len(dataset.keys())}")