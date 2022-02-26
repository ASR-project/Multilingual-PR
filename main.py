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
                config = wdb_config,
                project = parameters.hparams.wandb_project,
                entity = parameters.hparams.wandb_entity,
                allow_val_change=True,
                job_type="train"
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
        wandb.init(
                # vars(parameters),  # FIXME use the full parameters
                config = wdb_config,
                project = parameters.hparams.wandb_project,
                entity = parameters.hparams.wandb_entity,
                allow_val_change=True,
                job_type="test"
        )
        agent = BaseTrainer(parameters)
        agent.predict()
        
if __name__ == '__main__':
    main()
