import argparse
import os

import wandb


def push_artifact(name):
    path_artifact = name

    wandb.init(
        entity="asr-project",
        project="asr"
    )

    name = name.replace('_', '-')
    artifact = wandb.Artifact(
        name=os.path.basename(name),
        type="dataset",
        metadata={
            "emb_dim":768
        },
        description=f"Embeddings obtained from the top 10 keywords for each abstract using {name.replace('.npy', '')}"
    )

    artifact.add_file(path_artifact)
    wandb.log_artifact(artifact, aliases=["latest"])
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Push an artifact to wandb")
    parser.add_argument("--file_name",required=True,type = str, help = "name of the file which should be located at /input/file_name")
    args = parser.parse_args()
    push_artifact(args.file_name)
