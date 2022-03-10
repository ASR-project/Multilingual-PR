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
        wandb.init(
                # vars(parameters),  # FIXME use the full parameters
                name = f"{parameters.network_param.network_name}_{parameters.data_param.language}",
                config = wdb_config,
                project = parameters.hparams.wandb_project,
                entity = parameters.hparams.wandb_entity,
                allow_val_change = True,
                job_type = "train",
                tags = [
                    parameters.data_param.dataset_name,
                    parameters.data_param.subset,
                    parameters.optim_param.optimizer,
                    parameters.network_param.network_name,
                    parameters.network_param.network_name,
                    ]
            )
        
        wandb_run = WandbLogger(
            config=wdb_config,# vars(parameters),  # FIXME use the full parameters
            project=parameters.hparams.wandb_project,
            entity=parameters.hparams.wandb_entity,
            allow_val_change=True,
            #save_dir=parameters.hparams.save_dir,
        )
        
        agent = BaseTrainer(parameters, wandb_run)
        agent.run()
    else: 
        wandb_run = wandb.init(
                # vars(parameters),  # FIXME use the full parameters
                name = f"{parameters.network_param.network_name}_{parameters.data_param.language}_predict",
                config = wdb_config,
                project = parameters.hparams.wandb_project,
                entity = parameters.hparams.wandb_entity,
                allow_val_change = True,
                job_type = "predict",
                tags = [
                    parameters.data_param.language,
                    parameters.data_param.dataset_name,
                    parameters.data_param.subset,
                    parameters.optim_param.optimizer,
                    parameters.network_param.network_name,
                    f"{parameters.data_param.dataset_name}-predict"
                    ]
            )

        wandb_logger = WandbLogger(
            config=wdb_config,# vars(parameters),  # FIXME use the full parameters
            project=parameters.hparams.wandb_project,
            entity=parameters.hparams.wandb_entity,
            allow_val_change=True,
            #save_dir=parameters.hparams.save_dir,
        )

        agent = BaseTrainer(parameters, wandb_logger)
        agent.predict()
        
if __name__ == '__main__':
    main()
