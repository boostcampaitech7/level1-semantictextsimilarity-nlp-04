import os
import torch
import numpy as np
import base.base_data_loader as module_data
import module.loss as module_loss
import module.metric as module_metric
import module.model as module_arch
from module.trainer import Trainer
from omegaconf import DictConfig, OmegaConf
from utils import *
import hydra
import wandb
from typing import Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg):

    def sweep_train(cfg: Any = cfg) -> None:
        cfg.pwd = os.getcwd()
        config = OmegaConf.to_container(cfg, resolve=True)

        # 0. DictConfig to dict
        wandb.init(project=config['wandb']['project_name'])
        sweep_config = wandb.config

        # 1. set data_module(=pl.DataModule class)
        data_module = init_obj(
            config["data_module"]["type"],
            config["data_module"]["args"],
            module_data,
        )

        # 2. set model(=nn.Module class)
        model = init_obj(config["arch"]["type"],
                         config["arch"]["args"],
                         module_arch,
        )

        # 3. set device(cpu or gpu)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        model = model.to(device)

        # 4. set loss function & metrics
        criterion = getattr(module_loss, config["loss"])
        metrics = [getattr(module_metric, met) for met in config["metrics"]]

        # 5. set optimizer & learning scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        # Optimizer 생성
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=sweep_config.lr,  # learning rate
            weight_decay=sweep_config.weight_decay,  # weight decay
            amsgrad=config["optimizer"]["args"]["amsgrad"]  # amsgrad
        )

        # Learning rate scheduler 생성
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config["lr_scheduler"]["args"]["step_size"],
            gamma=config["lr_scheduler"]["args"]["gamma"]
        )

        # 6. 위에서 설정한 내용들을 trainer에 넣는다.
        trainer = Trainer(
            model,
            criterion,
            metrics,
            optimizer,
            config=config,
            device=device,
            data_module=data_module,
            lr_scheduler=lr_scheduler,
        )

        # 6. train
        trainer.train()

    # 7. sweep 실행
    #sweep_id = wandb.sweep(cfg.sweep_config, project=cfg.wandb['project_name'])
    sweep_config_dict = OmegaConf.to_container(cfg.sweep_config, resolve=True)
    sweep_id = wandb.sweep(sweep=sweep_config_dict, project=cfg.wandb['project_name'])
    wandb.agent(sweep_id, function=sweep_train, count=cfg.wandb['sweep_count'])

if __name__ == "__main__":
    main()
