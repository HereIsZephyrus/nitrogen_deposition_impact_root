import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor

logger = logging.getLogger(__name__)

class LightGBMAnalyzer:
    """LightGBM Model Analyzer"""

    def __init__(self, data_path=None, params=None):
        """
        Initialize LightGBM analyzer

        Args:
            data_path: Data file path
            params: LightGBM parameters dictionary
        """
        self.data_path = data_path
        self.params = params or {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0,
            'n_estimators': 100
        }
        self.model = None
        self.raw_data = None
        self.not_zero_data = None
        self.zero_data = None

    def load_data(self, data_path=None):
        """Load data"""
        path = data_path or self.data_path
        if path is None:
            raise ValueError("Please provide data file path")

        self.raw_data = pd.read_csv(path)
        # Separate non-zero and zero samples
        self.not_zero_data = self.raw_data.loc[self.raw_data['GTDindex'] != 0]
        self.zero_data = self.raw_data.loc[self.raw_data['GTDindex'] == 0]

        logger.info(f"Data loading completed: non-zero samples {len(self.not_zero_data)}, zero samples {len(self.zero_data)}")

    def get_ratio_data(self, zero_ratio=1):
        """
        Get data by ratio

        Args:
            zero_ratio: Ratio of zero samples to non-zero samples

        Returns:
            Combined data
        """
        if self.not_zero_data is None or self.zero_data is None:
            raise ValueError("Please load data first")

        not_zero_row_num = self.not_zero_data.shape[0]
        sample_num = not_zero_row_num * int(zero_ratio)

        # Sample zero data
        sampled_zero_data = self.zero_data.sample(n=sample_num, replace=True)

        # Combine data
        full_data = pd.concat([self.not_zero_data, sampled_zero_data], axis=0)
        return full_data

    def log_transform_data(self, data):
        """
        Apply log transformation to data

        Args:
            data: Input data

        Returns:
            Transformed data
        """
        # Remove negative values
        data = data.drop(data[data['GTDindex'] < 0].index)
        # Apply log2 to GTDindex
        data['GTDindex'] = data['GTDindex'].apply(lambda x: np.log2(x) if x > 1 else x)
        return data

    def prepare_features_target(self, data):
        """
        Prepare features and target variables

        Args:
            data: Input data

        Returns:
            features, target
        """
        # Remove non-feature columns
        features = data.drop(labels=['gid', 'xcoord', 'ycoord', 'col', 'row', 'GTDindex'], axis=1)
        target = data["GTDindex"]
        return features, target

    def train_model(self, data, test_size=0.2, random_state=7):
        """
        Train LightGBM model

        Args:
            data: Training data
            test_size: Test set proportion
            random_state: Random seed

        Returns:
            Trained model and performance metrics
        """
        features, target = self.prepare_features_target(data)

        # Split training and test sets
        train_x, test_x, train_y, test_y = train_test_split(
            features, target, test_size=test_size, random_state=random_state
        )

        # Train model
        self.model = LGBMRegressor(**self.params)
        self.model.fit(train_x, train_y)

        # Predict
        test_predict = self.model.predict(test_x)
        train_predict = self.model.predict(train_x)

        # Calculate performance metrics
        mse_test = mean_squared_error(test_y, test_predict)
        mse_train = mean_squared_error(train_y, train_predict)
        r2_test = self.model.score(test_x, test_y)
        corr_coef = np.corrcoef(test_predict, test_y)[0, 1]

        results = {
            'mse_test': mse_test,
            'mse_train': mse_train,
            'r2_test': r2_test,
            'correlation': corr_coef,
            'test_x': test_x,
            'test_y': test_y,
            'test_predict': test_predict,
            'train_predict': train_predict
        }

        return results

    def evaluate_model(self, test_data):
        """
        Evaluate model on test data

        Args:
            test_data: Test data

        Returns:
            Evaluation results
        """
        if self.model is None:
            raise ValueError("Please train the model first")

        gid = test_data['gid']
        target = test_data["GTDindex"]
        features = test_data.drop(labels=['gid', 'xcoord', 'ycoord', 'col', 'row', 'GTDindex'], axis=1)

        predicted = self.model.predict(features)

        mse = mean_squared_error(predicted, target)
        r2 = r2_score(predicted, target)
        corr_coef = np.corrcoef(predicted, target)[0, 1]

        results = {
            'mse': mse,
            'r2': r2,
            'correlation': corr_coef,
            'predicted': predicted,
            'target': target,
            'gid': gid
        }

        return results

    def train_with_ratio(self, zero_ratio=1, use_log=True, verbose=True):
        """
        Train model with specified ratio

        Args:
            zero_ratio: Zero sample ratio
            use_log: Whether to use logarithmic transformation
            verbose: Whether to output detailed information

        Returns:
            Training results
        """
        data = self.get_ratio_data(zero_ratio)

        if use_log:
            data = self.log_transform_data(data)

        if verbose:
            logger.info(f"Ratio is 1:{zero_ratio}")
            print("------------------")

        results = self.train_model(data)

        if verbose:
            logger.info(f"Test set RMSE: {results['mse_test']}")
            logger.info(f"Training set RMSE: {results['mse_train']}")
            logger.info(f"R^2: {results['r2_test']}")
            logger.info(f"Correlation coefficient: {results['correlation']}")
            logger.info("------------------")

        return results

    def export_predictions(self, predictions, target, gid, file_path):
        """
        Export prediction results to CSV file

        Args:
            predictions: Predicted values
            target: True values
            gid: Grid ID
            file_path: Output file path
        """
        # Create DataFrame
        predict_df = pd.DataFrame({
            'gid': gid.reset_index(drop=True),
            'GTDindex': target.reset_index(drop=True),
            'predGTDindex': predictions
        })

        # Save to CSV
        predict_df.to_csv(file_path, index=False)
        logger.info(f"Prediction results exported to: {file_path}")

    def print_evaluation(self, results, dataset_name="数据集"):
        """
        Print evaluation results

        Args:
            results: Evaluation results dictionary
            dataset_name: Dataset name
        """
        logger.info(f"--------On {dataset_name}----------")
        logger.info(f"RMSE: {results['mse']}")
        logger.info(f"R^2: {results['r2']}")
        logger.info(f"Correlation coefficient: {results['correlation']}")
        logger.info("--------------------------------")

    def get_feature_importance(self, plot=False):
        """
        Get feature importance

        Args:
            plot: Whether to plot feature importance

        Returns:
            Feature importance dictionary
        """
        if self.model is None:
            raise ValueError("Please train the model first")

        # Get feature importance
        importance = self.model.feature_importances_
        feature_names = self.model.feature_names_in_ if hasattr(self.model, 'feature_names_in_') else None

        if feature_names is not None:
            importance_dict = dict(zip(feature_names, importance))
            # Sort by importance
            importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        else:
            importance_dict = {f'feature_{i}': imp for i, imp in enumerate(importance)}

        if plot:
            try:
                import matplotlib.pyplot as plt

                features = list(importance_dict.keys())[:20]  # Show top 20 important features
                values = list(importance_dict.values())[:20]

                plt.figure(figsize=(10, 6))
                plt.barh(features, values)
                plt.xlabel('Feature importance')
                plt.title('LightGBM Feature importance')
                plt.tight_layout()
                plt.show()
            except ImportError:
                logger.info("Please install matplotlib to plot feature importance")

        return importance_dict


def load_2010_data(data_path):
    """
    Load 2010 data

    Args:
        data_path: Data file path

    Returns:
        Processed 2010 data
    """
    df2010 = pd.read_csv(data_path)

    # Remove ocean grids
    df2010.dropna(axis=0, subset=["landarea"], inplace=True)

    # Remove useless attributes
    columns_to_drop = [
        'diamsec_y', 'diamprim_y', 'drug_y', 'diamsec_s', 'goldplac_1',
        'goldvein_y', 'goldsurf_1', 'petroleu_1', 'diamsec_s', 'diamprim_s',
        'gem_s', 'goldplacer', 'goldvein_s', 'goldsurfac', 'petroleum_', 'gem_y'
    ]
    existing_columns = [col for col in columns_to_drop if col in df2010.columns]
    df2010.drop(existing_columns, axis=1, inplace=True)

    # Apply log transformation
    df2010['GTDindex'] = df2010['GTDindex'].apply(lambda x: np.log2(x) if x > 1 else x)

    return df2010
