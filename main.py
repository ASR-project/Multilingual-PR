import faulthandler

faulthandler.enable()

from pytorch_lightning.loggers import WandbLogger

# Standard libraries
import wandb
from agents.BaseTrainer import BaseTrainer
from config.hparams import Parameters
from utils.agent_utils import parse_params


def main():
    parameters = Parameters.parse()

    # initialize wandb instance
    wdb_config = parse_params(parameters)
    
    if parameters.hparams.train:
        tags = [
                    parameters.data_param.dataset_name,
                    parameters.data_param.subset,
                    parameters.optim_param.optimizer,
                    parameters.network_param.network_name,
                    f"{'not'*(not parameters.network_param.freeze)} freezed",
                    parameters.network_param.pretrained_name
                ]

        if parameters.hparams.limit_train_batches!=1.0: tags += [f'{parameters.hparams.limit_train_batches}_train']
        
        wandb.init(
                name = f"{parameters.network_param.network_name}_{parameters.data_param.language}{'_fine_tune'*(not parameters.network_param.freeze)}{f'_{parameters.hparams.limit_train_batches}_train'*(parameters.hparams.limit_train_batches!=1.0)}",
                config = wdb_config,
                project = parameters.hparams.wandb_project,
                entity = parameters.hparams.wandb_entity,
                allow_val_change = True,
                job_type = "train",
                tags = tags
            )
        
        wandb_run = WandbLogger(
            config=wdb_config,
            project=parameters.hparams.wandb_project,
            entity=parameters.hparams.wandb_entity,
            allow_val_change=True,
        )
        
        agent = BaseTrainer(parameters, wandb_run)
        agent.run()
    else: 
        wandb_run = wandb.init(
                name = f"{parameters.network_param.network_name}_{parameters.data_param.language}_test",
                config = wdb_config,
                project = parameters.hparams.wandb_project,
                entity = parameters.hparams.wandb_entity,
                allow_val_change = True,
                job_type = "test",
                tags = [
                    parameters.data_param.language,
                    parameters.data_param.dataset_name,
                    parameters.data_param.subset,
                    parameters.optim_param.optimizer,
                    parameters.network_param.network_name,
                    "test"
                    ]
            )

        wandb_logger = WandbLogger(
            config=wdb_config,
            project=parameters.hparams.wandb_project,
            entity=parameters.hparams.wandb_entity,
            allow_val_change=True,
        )

        agent = BaseTrainer(parameters, wandb_logger)
        agent.predict()
        
if __name__ == '__main__':
    main()
