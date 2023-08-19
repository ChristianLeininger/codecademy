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
    # console_handler.setLevel(cfg.job_logging.handlers.file.level)
    console_handler.setLevel(logging.ERROR)
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


def save_correlation_matrix(
        df: pd.DataFrame,
        save_path: str = "default",
        size: tuple = (
            20,
            10),
        max_name_length: int = 30) -> None:
    """ Plot statistics for a column
    Args:
        df (pd.DataFrame): Pandas DataFrame
        save_path (str): Path to save plot
        size (tuple): Size of plot
        max_name_length (int): Maximum length of column name
    Returns:
        None
    """
    # some assertions
    assert isinstance(df, pd.DataFrame), "df should be a Pandas DataFrame"
    assert isinstance(size, tuple), "size should be a tuple"
    assert isinstance(save_path, str), "save_path should be a string"
    assert isinstance(
        max_name_length, int), "max_name_length should be an integer"
    if save_path is "default":
        save_path = os.path.join(os.getcwd(), 'images')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        logging.info("Set save_path its default value")
        logging.info(f"save_path set to {save_path}")
    save_path = os.path.join(save_path, 'correlation_matrix.png')
    # rename to long column names
    df.rename(columns=lambda x: x if len(x) <=
              max_name_length else x[:max_name_length], inplace=True)
    plt.figure(figsize=size)
    sns.heatmap(df.corr(), annot=True, cmap='Dark2_r', linewidths=2)
    plt.savefig(save_path, bbox_inches='tight')


def save_statistics(
    df: pd.DataFrame,
    column: str,
    save_path: str,
    logger: logging.Logger,
    size: tuple = (
        20,
        10)) -> None:
    """ Plot statistics for a column
    Args:
        df (pd.DataFrame): Pandas DataFrame
        column (str): Column to plot
        save_path (str): Path to save plot
        size (tuple): Size of plot
    Returns:
        None
    """
    # some assertions
    assert isinstance(df, pd.DataFrame), "df should be a Pandas DataFrame"
    assert isinstance(column, str), "column should be a string"
    assert column in df.columns, "column should be in df"
    assert isinstance(size, tuple), "size should be a tuple"
    save_path = os.path.join(save_path, f'{column}.png')
    plt.figure(figsize=size)
    if len(df[column].unique()) > 10:
        # Bin the data into 10 intervals if there are more than 10 unique values
        range_val = df[column].max() - df[column].min()
        
        # Calculate interval size (rounded up to nearest integer)
        interval_size = int(np.ceil(range_val / 10))
        
        # Define integer boundaries for bins
        bins = list(range(int(df[column].min()), int(df[column].min() + interval_size * 11), interval_size))
        
        binned_data = pd.cut(df[column], bins=bins, right=False, include_lowest=True)
        # binned_data = pd.cut(df[column], bins=10)
        value_counts = binned_data.value_counts(sort=False)
        value_counts.sort_index(inplace=True)
    else:
        value_counts = df[column].value_counts()
    value_counts.plot(kind='bar')
    plt.title(f"Statistics for {column}")
    plt.ylabel('Frequency')
    plt.xlabel(f"{column}")
    plt.savefig(save_path, bbox_inches='tight')
    logger.info(f"Saved statistics plot of {column} to {save_path}")


def save_df_head(df: pd.DataFrame, save_path: str, fig_size: tuple) -> None:
    """ Save head of Pandas DataFrame to image file
    Args:
        df (pd.DataFrame): Pandas DataFrame
        save_path (str): Path to save plot
        figsize (tuple): Size of plot
    """

    logging.info(f"Saving head of Pandas DataFrame to {save_path} ....")

    fig, ax = plt.subplots()
    # Adjust bbox values as needed to center the table
    # Create the table and set cell padding
    data = df.head().T.round(2)
    table = plt.table(cellText=data.values,
                      colLabels=df.index,
                      rowLabels=df.columns,
                      cellLoc='center',
                      loc='center',
                      )

    table.auto_set_column_width(col=list(range(len(df.columns))))
    ax.axis('off')
    plt.savefig(os.path.join(save_path, 'head.png'), bbox_inches='tight')
    plt.close()
    plt.clf()
    logging.info(f"Saved head of Pandas DataFrame to {save_path}")


def save_describtion(df: pd.DataFrame, save_path: str) -> None:
    """ Save descriptive statistics to image file
    Args:
        df (pd.DataFrame): Pandas DataFrame
        save_path (str): Path to save plot
        file_name (str): Name of file
    """
    # some assertions
    assert isinstance(df, pd.DataFrame), "df should be a Pandas DataFrame"
    assert isinstance(save_path, str), "save_path should be a string"

    save_path = os.path.join(save_path, 'descriptive_statistics.png')
    # if already image already exists return
    if os.path.exists(save_path):
        logging.info(f"Image already exists at {save_path}")
        return

    desc_stats = df.describe().round(2)

    fig, ax = plt.subplots()
    # Adjust bbox values as needed to center the table
    # Create the table and set cell padding
    table = plt.table(cellText=desc_stats.values.T,
                      colLabels=desc_stats.index,
                      rowLabels=desc_stats.columns,
                      cellLoc='center',
                      loc='center',
                      )

    table.auto_set_column_width(col=list(range(len(desc_stats.columns))))
    ax.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    logging.info(f"Saved plot to {save_path}")
 



def save_scatterplot(data: pd.DataFrame, save_path: str, logger: logging.Logger, fig_size: tuple) -> None:
    """
    Save scatter plots of age vs. charges, bmi vs. charges, and children vs. charges
    Args:
        data (pd.DataFrame): Pandas DataFrame
        save_path (str): Path to save plot
    """
    # some assertions
    assert isinstance(data, pd.DataFrame), "data should be a Pandas DataFrame"
    assert isinstance(save_path, str), "save_path should be a string"
    # Set the style and color palette for seaborn plots
    sns.set_style("whitegrid")
    sns.set_palette("Set2")

    # Create a figure with 3 subplots
    fig, ax = plt.subplots(3, 1, figsize=fig_size)

    # Plot Age vs. Charges
    sns.scatterplot(x='age', y='charges', hue='smoker', data=data, ax=ax[0])
    ax[0].set_title('Age vs. Charges')
    ax[0].set_xlabel('Age')
    ax[0].set_ylabel('Charges')

    # Plot BMI vs. Charges
    sns.scatterplot(x='bmi', y='charges', hue='smoker', data=data, ax=ax[1])
    ax[1].set_title('BMI vs. Charges')
    ax[1].set_xlabel('BMI')
    ax[1].set_ylabel('Charges')

    # Plot Age vs. BMI
    sns.scatterplot(x='age', y='bmi', hue='smoker', data=data, ax=ax[2])
    ax[2].set_title('Age vs. BMI')
    ax[2].set_xlabel('Age')
    ax[2].set_ylabel('BMI')

    # Adjust the layout of the plots
    # save plot
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'scatter_plots.png'), bbox_inches='tight')
    logger.info(f"Saved plot to {os.path.join(save_path, 'scatter_plots.png')}")




def compute_metric(pred: np.ndarray, target: np.ndarray,  logger: logging.Logger, model_name: str, path: str) -> float:
    """ Compute metric
    
    Args:
        pred (list): List of predictions
        target (list): List of targets
        logger (logging.Logger): Logger
        model_name (str): Name of model
        path (str): Path to save plot
    """ 
    # some assertions
    assert isinstance(pred, np.ndarray), "pred should be a numpy array"
    assert isinstance(target, np.ndarray), "target should be a numpy array"
    assert isinstance(logger, logging.Logger), "logger should be a logging.Logger"
    # assert same shape
    assert pred.shape == target.shape, "pred and target should have the same shape"

    # Count overestimations and underestimations
    overestimations = sum([1 for pred, true in zip(pred, target) if pred > true])
    underestimations = sum([1 for pred, true in zip(pred, target) if pred < true])
    # Compute ratio of overestimations to underestimations
    # If underestimations is 0, then the ratio would be infinity. We handle that case as well.
    ratio = overestimations / underestimations if underestimations != 0 else float('inf')
    logger.info(f"Model {model_name} Overestimations: {overestimations} Underestimations: {underestimations} Ratio: {ratio}")
    
    # Compute the absolute differences
    absolute_differences = np.abs(pred - target)

    
    
    max_diff = max(absolute_differences)

    # Create human-readable, rounded bins
    bin_size = np.ceil(max_diff / 20)
    rounded_bin_size = int(10 * np.ceil(bin_size/10))  # Round upwards to the nearest 10
    bins = np.arange(0, rounded_bin_size*21, rounded_bin_size)

    # Create a histogram
    hist, bin_edges = np.histogram(absolute_differences, bins=bins)
    total_observations = len(absolute_differences)

    # Create a bar plot with uniform bar widths
    bar_width = 0.8
    positions = np.arange(len(hist))  # uniform bar positions

    bars = plt.bar(positions, hist, width=bar_width, edgecolor="black")

    # Set the x-axis labels as ranges
    labels = [f"{int(edge)}-{int(bin_edges[i+1])}" for i, edge in enumerate(bin_edges[:-1])]
    plt.xticks(positions, labels, rotation=45, ha='right')  # place ticks at the center of the bars and rotate for better visualization

    # Annotate each bar with its count and percentage
    for bar in bars:
        yval = bar.get_height()
        percentage = yval / total_observations * 100
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{round(yval, 2)}\n({percentage:.1f}%)", ha='center', va='bottom')

    plt.ylabel("Count")
    plt.title(f"Histogram of Absolute Differences from {model_name} Predictions")
    plt.xlabel("Absolute Difference Range")
    
    plt.tight_layout()  # to ensure that all labels fit well
    # plt.show()
    # clear plot
    plt.clf()
    # Scatter plots for actual and predicted values
    # Scatter plots for actual and predicted values
    plt.scatter(target, target, color='blue', label='Actual', alpha=0.5, s=50)  # actual values in blue
    plt.scatter(target, pred, color='red', label='Predicted', alpha=0.5, s=50)  # predicted values in red

    # Lines connecting actual and predicted points
    for i, txt in enumerate(target):
        plt.plot([target[i], target[i]], [target[i], pred[i]], color='grey', linestyle='--', lw=0.5)

    # 45-degree line
    plt.plot([min(target), max(target)], [min(target), max(target)], color='green') 

    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Actual vs. Predicted for {model_name}")
    plt.legend(loc='upper left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"{model_name}_actual_vs_predicted.png"), bbox_inches='tight')
    plt.clf()
    residuals = pred - target
    fig, ax = plt.subplots(figsize=(10, 6))

    n, bins, patches = ax.hist(residuals, bins=20, edgecolor='black', rwidth=0.8)

    total = sum(n)  # Total number of residuals
    threshold_percentage = 5  # Threshold for what we consider a "large" bar

    # Determine the font size.
    base_font_size = 10
    font_size = base_font_size * (8 / fig.get_size_inches()[0]) 

    # For each bar: Place a label
    for bin, patch, value in zip(bins[:-1], patches, n):  # Skipping the last bin value
        # Get the x-coordinate (center of the bar)
        x2 = patch.get_x() + patch.get_width() / 2 - 0.3 * patch.get_width()
        x = patch.get_x() + patch.get_width() / 2 
        percentage = (value / total) * 100
        
        # If bar's percentage is more than the threshold
        if percentage > threshold_percentage:
            # Display percentage inside the bar
            ax.text(x, value / 2, f'{percentage:.1f}%', ha='center', va='center', color='white', fontsize=font_size)
        
        # Place the bin value just below the bar (only for non-zero bars)
        if value > 0:
            ax.text(x2, -0.05 * max(n), f'{bin:.2f}', ha='center', va='center', rotation=45, color='black', fontsize=font_size)
        
        # Place the count value on top of the bar
        ax.text(x, value + 0.02 * max(n), f'{int(value)}', ha='center', va='center', color='black', fontsize=font_size)

    ax.set_xlabel('Residual')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Residuals')
    ax.grid(True)
    ax.set_xticks([])  # Remove x-axis tick labels
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"{model_name}_residuals.png"), bbox_inches='tight')
