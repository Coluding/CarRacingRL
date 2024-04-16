import torch
from typing import Callable, Tuple
import logging

def convert_tuple_output(func: Callable, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
    return tuple([torch.tensor(x) for x in func(*args, **kwargs)])


def prep_logger() -> logging.Logger:
    # Setup basic configuration for logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a file handler that logs even debug messages
    log_file = 'training.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)  # Set this to DEBUG if you want to log debug messages too
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Get the logger and attach the file handler
    logger = logging.getLogger('RLTrainer')
    logger.addHandler(file_handler)

    return logger
