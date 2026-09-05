import os
from datetime import datetime

from tianshou.highlevel.logger import LoggerFactoryDefault


def log_tensorboard(task, persistence_base_dir, seed, resume_id=None):
    now = datetime.now().strftime("%y%m%d-%H%M%S")
    algo_name = "ppo"
    log_name = os.path.join(task, algo_name, str(seed), now)
    log_path = os.path.join(persistence_base_dir, log_name)
    logger_factory = LoggerFactoryDefault()
    logger_factory.logger_type = "tensorboard"
    return (
        logger_factory.create_logger(
            log_dir=log_path,
            experiment_name=log_name,
            run_id=resume_id,
        ),
        log_path,
    )
