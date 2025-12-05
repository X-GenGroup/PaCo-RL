import logging
from argparse import Namespace
from accelerate import Accelerator
from accelerate.logging import get_logger
import datetime

logger = get_logger(__name__)


def set_online_log(accelerator : Accelerator, config : Namespace):
    """
        Initialize logging with platform
    """
    if config.logging_platform == 'wandb':
        run, logging_platform = setup_wandb_log(accelerator, config)
    elif config.logging_platform == 'swanlab':
        run, logging_platform = setup_swanlab_log(accelerator, config)
    else:
        raise ValueError(f"Unsupported logging platform: {config.logging_platform}")
    
    return run, logging_platform

def setup_wandb_log(accelerator : Accelerator, config : Namespace):
    """
        Initialize wandb training log
    """
    import wandb
    if config.resume_from_id is not None:
        project_name = config.project_name
        run_id = config.resume_from_id
        # Get history
        api_run = wandb.Api().run(f"{project_name}/{run_id}")
        history = api_run.history()
        if not history.empty:
            if config.resume_from_step is None:
                config.resume_from_step = int(history['_step'].iloc[-1])
            if config.resume_from_epoch is None:
                config.resume_from_epoch = config.resume_from_step // 2
            logger.info(f"Auto-resuming from step {config.resume_from_step}, epoch {config.resume_from_epoch}")
        else:
            logger.info("No previous history found, starting from beginning")
            config.resume_from_step = 0
            config.resume_from_epoch = 0

        if accelerator.is_main_process:
            run = wandb.init(
                project=config.project_name,
                config=config.to_dict(),
                id=run_id,
                resume='must'
            )
        else:
            run = None
    else:
        unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        if not config.run_name:
            config.run_name = unique_id
        else:
            config.run_name += "_" + unique_id
        if accelerator.is_main_process:
            run = wandb.init(
                project=config.project_name,
                name=config.run_name,
                config=config.to_dict()
            )
        else:
            run = None

    return run, wandb

def setup_swanlab_log(accelerator : Accelerator, config : Namespace):
    """
        Initialize swanlab training log
    """
    import swanlab
    if config.resume_from_id:
        project_name = config.project_name
        run_id = config.resume_from_id
        # Get history
        api = swanlab.OpenApi()
        run_summary = api.get_summary(project=project_name, exp_id=run_id)
        if config.resume_from_step is None:
            config.resume_from_step = run_summary.data['epoch']['max']['step']
        if config.resume_from_epoch is None:
            config.resume_from_epoch = run_summary.data['epoch']['max']['value']
        logger.info(f"Auto-resuming from step {config.resume_from_step}, epoch {config.resume_from_epoch}")

        if accelerator.is_main_process:
            run = swanlab.init(
                project=project_name,
                config=config.to_dict(),
                resume=True,
                id=run_id
            )
        else:
            run = None
    else:
        unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        if not config.run_name:
            config.run_name = unique_id
        else:
            config.run_name += "_" + unique_id

        if accelerator.is_main_process:
            run = swanlab.init(
                project=config.project_name,
                name=config.run_name,
                config=config.to_dict()
            )
        else:
            run = None

    return run, swanlab