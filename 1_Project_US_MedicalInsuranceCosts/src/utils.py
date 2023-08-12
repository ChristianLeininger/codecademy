import pandas as pd
from logging.handlers import RotatingFileHandler
import plotly.graph_objects as go
from datetime import datetime
import wandb
import logging
import os
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def get_logger(cfg: DictConfig) -> logging.Logger:
    """ Get logger

    Args:
        cfg (DictConfig): Configuration
    Returns:
        logging.Logger: Logger
    """
    # some assertions
    assert isinstance(cfg, DictConfig), "cfg should be a DictConfig"

    # Create a custom logger
    logger = logging.getLogger(cfg.job_logging.name)
    # Set to your desired logging level.
    logger.setLevel(cfg.job_logging.handlers.file.level)

    # Create a file handler
    current_data_time = datetime.now().strftime("%A,_%d_%B_%Y_%H:%M:%S")
    file_handler = RotatingFileHandler(
        os.path.join(
            cfg.pth,
            "logs",
            f"{cfg.job_logging.name}_{current_data_time}.log"),
        maxBytes=2000,
        backupCount=5)
    # Set to your desired logging level. Here, DEBUG is chosen.
    file_handler.setLevel(cfg.job_logging.handlers.file.level)

    # Create a console handler
    console_handler = logging.StreamHandler()
    # Set to your desired logging level. Here, ERROR is chosen.
    console_handler.setLevel(cfg.job_logging.handlers.file.level)

    # Create a formatter and set it for both handlers

    formatter = logging.Formatter(cfg.job_logging.formatters.basic.format)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add both file and console handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger



def get_categorical_columns(df: pd.DataFrame) -> list:
    """ Get list of categorical columns from a Pandas DataFrame

    Args:
        df (pd.DataFrame): Pandas DataFrame
    Returns:
        list: List of categorical columns
    """
    # Select columns of type 'object' and 'category'
    categorical_columns = df.select_dtypes(
        include=['object', 'category']).columns
    return list(categorical_columns)

