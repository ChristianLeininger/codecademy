import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import numpy as np
import hydra
import wandb
import omegaconf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from itertools import product
import time

from src.utils import get_logger, get_categorical_columns
from src.utils import save_correlation_matrix, save_statistics
from src.utils import save_df_head, save_describtion
from src.utils import save_scatterplot, compute_metric





class Insurance:
    def __init__(self, cfg):
        """ Initialize Insurance object 
        Args:
            cfg (DictConfig): Configuration
        """
        # some assertions
        assert os.path.exists(cfg['pth']), "pth should be a valid path change in hydra config file"
        self.cfg = cfg
        self.cv = cfg.models['cv']
        self.pth = cfg.pth
        self.data_path = os.path.join(self.pth,  cfg.data['data_path'])
        self.seed = cfg['seed']
        self.track = cfg['track']
        self.fig_size = (cfg.plot['figsize_x'], cfg.plot['figsize_y'])
        self.image_path = os.path.join(self.pth, 'images')
        self.params = cfg.models["params"]
        self.check_dir()
        if self.track:
            experiment_name = f"{cfg.wandb['experiment_name']}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_seed_{cfg['seed']}"
            self.init_wandb(project_name=cfg.wandb['project_name'], experiment_name=experiment_name)
    

    def check_dir(self):
        """ Check if directories exist and create them if not
        """
        if not os.path.exists(self.pth):
            os.makedirs(self.pth)
        
        self.log_path = os.path.join(self.pth, 'logs')
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.logger = get_logger(self.cfg)
        self.logger.info(f"Logging to {self.log_path}")
        if not os.path.exists(self.image_path):
            os.makedirs(self.image_path)
            self.logger.info(f"Created directory {self.image_path}")
        else:
            self.logger.info(f"Directory {self.image_path} already exists")
        

    def init_wandb(self, project_name: str, experiment_name: str) -> None:
        """ Initialize wandb
        Args:
            project_name (str): Name of project
            experiment_name (str): Name of experiment
        """
        self.logger.info("Initializing wandb...")
        assert isinstance(project_name, str), "project_name should be a string"
        assert isinstance(experiment_name, str), "experiment_name should be a string"
        # initialize wandb
        self.run = wandb.init(project=project_name,
                name=experiment_name,
                )
        self.logger.info("Successfully initialized wandb.")


    def load_data(self):
        """ Load data from csv file

        """
        self.data = pd.read_csv(self.data_path, sep=',')
        assert isinstance(self.data, pd.DataFrame), "data should be a pandas DataFrame"
        self.logger.info(f"Data loaded from {self.data_path} with shape {self.data.shape}")
        self.logger.info(f"Data columns {self.data.columns}")

    def data_cleaning(self):
        """ check for null values and process them
            with the given strategy in the config file

        """  
        #check for null values
        self.logger.info(f"Data cleaning with shape {self.data.shape}")
        self.logger.info(f"Data columns {self.data.columns}")
        self.logger.info(f"Data info {self.data.info()}")
        # check for null values
        self.logger.info(f"Data null values {self.data.isnull().sum()}")
        # choose strategy
        if self.cfg.data['strategy'] == 'drop':
            self.data.dropna(inplace=True)
            return
        # first get categorical columns to encode them
        self.preprocess_data()
        if self.cfg.data['strategy'] == 'mean':
            self.data.fillna(self.data.mean(), inplace=True)
        elif self.cfg.data['strategy'] == 'median':
            self.data.fillna(self.data.median(), inplace=True)
        elif self.cfg.data['strategy'] == 'mode':
            self.data.fillna(self.data.mode(), inplace=True)
        else:
            self.logger.info(f"Strategy {self.cfg.data['strategy']} not implemented")
        
    def preprocess_data(self):
        """ le and one hot encode categorical columns
        """
        # get categorical columns
        cat_cols = get_categorical_columns(self.data)
        # Create label encoder
        le = LabelEncoder()
        for col in cat_cols:
            self.data[col] = le.fit_transform(self.data[col])
        
        
        self.logger.info(f"New data head {self.data.head()}")
        self.logger.info(f"Data preprocessed with shape {self.data.shape}")
    
    def perform_eta(self, perform_eta: bool = True):
        """ Perform eta analysis
        Args:
            perform_eta (bool, optional): Perform eta analysis. Defaults to True.
        """
        if not perform_eta:
            self.logger.info("Skipping eta analysis")
            return
        # get categorical columns
        save_correlation_matrix( self.data, self.image_path)
        save_df_head(self.data, self.image_path, fig_size=self.fig_size)
        save_describtion(self.data, self.image_path)
        save_scatterplot(self.data, self.image_path, logger=self.logger, fig_size=self.fig_size)
        for col in self.data.columns:
            save_statistics(df = self.data, column=col, save_path=self.image_path, logger=self.logger)
    
    def perform_data_preprozesing(self):
        """ """
        # Split the data into train and a temporary set using stratified sampling
        self.X_train, temp_df = train_test_split(self.data, test_size=0.4, stratify=self.data['smoker'], random_state=self.seed)

        # Split the temporary set into validation and test sets
        self.X_valid, self.X_test = train_test_split(temp_df, test_size=0.5, stratify=temp_df['smoker'], random_state=self.seed)

        # Separate the target variable from the train, validation, and test sets
        self.y_train = self.X_train.pop('charges')
        self.y_valid = self.X_valid.pop('charges')
        self.y_test = self.X_test.pop('charges')
        # check shapes
        self.logger.info(f"X_train shape {self.X_train.shape}")
        self.logger.info(f"X_valid shape {self.X_valid.shape}")
        self.logger.info(f"X_test shape {self.X_test.shape}")
    
    def train_baseline(self):
        """ """

        model = LinearRegression()

        # Train the model
        model.fit(self.X_train, self.y_train)
        
        # Check model performance on train and test sets
        mse = model.score(self.X_train, self.y_train)
        self.logger.info(f"Train MSE: {mse}")
        # Predict using the test set
        y_pred = model.predict(self.X_test)
        

        # Evaluate the model using Mean Squared Error (MSE)
        mse = mean_squared_error(self.y_test, y_pred)
        
        compute_metric(self.y_test.values, y_pred, self.logger, model_name='baseline')
        
        self.logger.info(f"Mean Squared Error (MSE): {mse}")
    

    def train_lasso_model(self):
        """ """
        # train model with different alpha values
        # then compare the results
        # and choose the best alpha value
        alpha_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 15, 20, 25, 30, 50, 100, 1000]
        res = {}
        for alpha in alpha_values:
            model = Lasso(alpha=alpha, random_state=self.seed)
            model.fit(self.X_train, self.y_train)
            # Check model performance on train and test sets
            mse_train = model.score(self.X_train, self.y_train)
            self.logger.info(f"Train MSE: {mse_train}")
            # Predict using the test set
            y_pred = model.predict(self.X_test)
            # Evaluate the model using Mean Squared Error (MSE)
            mse = mean_squared_error(self.y_test, y_pred)
            self.logger.info(f"Mean Squared Error (MSE): {mse}")
            res.update({alpha: {'mse_train': mse_train, 'mse': mse}})
        # sort res dict by mse value lowest first
        res = dict(sorted(res.items(), key=lambda item: item[1]['mse']))
        # log results starting with the best alpha value
        for key, value in res.items():
            self.logger.info(f"Alpha {key}  mse_train {value['mse_train']:.3f} mse {value['mse']:.3f}")
    

    def train_decision_tree(self):
        """ """
        regressor = DecisionTreeRegressor(max_depth=10)
        regressor.fit(self.X_train, self.y_train)
        # Check model performance on train and test sets
        mse_train = regressor.score(self.X_train, self.y_train)
        self.logger.info(f"Train MSE: {mse_train}")
        # Predict using the test set
        y_pred = regressor.predict(self.X_test)
        # Evaluate the model using Mean Squared Error (MSE)
        mse = mean_squared_error(self.y_test, y_pred)
        self.logger.info(f"Mean Squared Error (MSE): {mse}")
        compute_metric(self.y_test.values, y_pred, self.logger, model_name='decision_tree')



    def grid_search_regression_tree(self):
        """   """
        estimator = DecisionTreeRegressor()
        total_combinations = 1
        for key, values in self.cfg.models.params.items():
            total_combinations *= len(values)
        keys, values = zip(*self.cfg.models.params.items())
        # table = wandb.Table(columns=["Hyperparameters", "Mean CV Score", "Test Score"])
        hyperparam_columns = list(self.cfg.models.params.keys())
        table_columns = hyperparam_columns + ["Mean CV Score", "Test Score"]
        
        table = wandb.Table(columns=table_columns)
        # start measturing time
        start_time = time.time()
        for idx, combination in enumerate(product(*values)):  # Iterate over cartesian product

            params = dict(zip(keys, combination))
            
            estimator.set_params(**params)
            # does not fit the model
            cv_scores = cross_val_score(estimator, self.X_train, self.y_train, cv=self.cv)
            mean_cv_score = np.mean(cv_scores)
            # import pdb; pdb.set_trace()
            estimator.fit(self.X_train, self.y_train)
            test_score = estimator.score(self.X_test, self.y_test)
            estimator.predict(self.X_test)
            params_str = ', '.join([f"{k}={v}" for k, v in params.items()])
            # self.logger.info(f" Run {idx} of {total_combinations}  params: {params_str}, mean_cv_score: {mean_cv_score:.2f}, test_score: {test_score:.2f}")

            row_data = [ str(params[key]) for key in hyperparam_columns] + [mean_cv_score, test_score]
           

            table.add_data(*row_data)
            if idx % 500 == 0:
                elapsed_time = time.time() - start_time
                minutes, seconds = divmod(elapsed_time, 60)
                self.logger.info(f" Run {idx} of {total_combinations}  mean_cv_score: {mean_cv_score:.2f}, test_score: {test_score:.2f} Elapsed time: {int(minutes)} minutes and {seconds:.2f} seconds")
                
        
    
        if self.track:
            self.run.log({"grid_search": table})



    def remove_outliers(self, columns: list):
        """ 
        Remove outliers from the data
        Args:
            columns (list): List of columns to remove outliers from
        """
        assert isinstance(columns, omegaconf.listconfig.ListConfig), "columns must be a list"
        assert len(columns) > 0, "columns must not be empty"
        assert all(isinstance(x, str) for x in columns), "columns must be a list of strings"

        # log columns to remove outliers before 
        self.logger.info(f"Columns to remove outliers from {columns}")
        # log stats before
        self.logger.info(f"data size before removing outliers {self.data.shape}")
        self.logger.info(f"Stats before removing outliers {self.data[columns].describe()}")
        # remove outliers
        for col in columns:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            self.data = self.data[~((self.data[col] < (Q1 - 1.5 * IQR)) |(self.data[col] > (Q3 + 1.5 * IQR)))]
        
        # log stats after
        self.logger.info(f"data size after removing outliers {self.data.shape}")
        self.logger.info(f"Stats after removing outliers {self.data[columns].describe()}")

        


@hydra.main(config_path="config", config_name="stable_train")
def main(cfg):
    """ Main function    
    Args:
        cfg (DictConfig): Configuration
    """
    # create Insurance object
    insurance = Insurance(cfg)
    # load data
    insurance.load_data()
    # data cleaning
    insurance.data_cleaning()
    # perform eta analysis
    insurance.perform_eta(perform_eta=cfg.eta)
    # perform data preprocessing
    insurance.perform_data_preprozesing()

    # train baseline model
    #insurance.train_baseline()
    #insurance.train_decision_tree()
    # remove outliers
    # import pdb; pdb.set_trace()
    #insurance.remove_outliers(columns=cfg.outliers)
    #insurance.train_lasso_model()
    #insurance.perform_data_preprozesing()
    # insurance.train_baseline()
    insurance.grid_search_regression_tree()


if __name__ == '__main__':
    main()