"""
Kernel SVM Regressor for biomass relationship analysis
"""
import logging
import numpy as np
from typing import Optional, Union, Tuple, Dict, List
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.compose import ColumnTransformer

logger = logging.getLogger(__name__)


class KernelSVMRegressor:
    """
    Kernel Support Vector Machine Regressor

    Support Vector Machine Regressor with kernel functions, providing automatic feature preprocessing,
    hyperparameter tuning and model evaluation.
    """

    def __init__(self,
                 kernel: str = 'rbf',
                 C: float = 1.0,
                 epsilon: float = 0.1,
                 gamma: Union[str, float] = 'scale',
                 auto_tune: bool = False):
        """
        Initialize Kernel SVM Regressor

        Args:
            kernel: Kernel function type ('linear', 'poly', 'rbf', 'sigmoid')
            C: Regularization parameter
            epsilon: Epsilon-tube error not included in loss
            gamma: Kernel function coefficient, 'scale' or 'auto' or specific value
            auto_tune: Whether to automatically tune hyperparameters
        """
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.auto_tune = auto_tune

        self.model = None
        self.preprocessor = None
        self.is_fitted = False

        self.continuous_features = None
        self.categorical_features = None
        self.feature_names = None

        self.best_params = None
        self.cv_results = None

        logger.info("Initialized KernelSVMRegressor: kernel=%s, C=%.2f, epsilon=%.2f, gamma=%s, auto_tune=%s", kernel, C, epsilon, gamma, auto_tune)

    def _create_preprocessor(self,
                            X_continuous: Optional[np.ndarray] = None,
                            X_categorical: Optional[np.ndarray] = None) -> ColumnTransformer:
        """
        Create feature preprocessor

        Args:
            X_continuous: Continuous features matrix
            X_categorical: Categorical features matrix

        Returns:
            ColumnTransformer preprocessor
        """
        transformers = []

        # Continuous features: standardization
        if X_continuous is not None and X_continuous.shape[1] > 0:
            n_continuous = X_continuous.shape[1]
            continuous_indices = list(range(n_continuous))
            transformers.append(
                ('continuous', StandardScaler(), continuous_indices)
            )
            self.continuous_features = n_continuous
            logger.info("Added %d continuous features", n_continuous)

        # Categorical features: one-hot encoding
        if X_categorical is not None and X_categorical.shape[1] > 0:
            n_categorical = X_categorical.shape[1]
            offset = self.continuous_features if self.continuous_features else 0
            categorical_indices = list(range(offset, offset + n_categorical))
            transformers.append(
                ('categorical',
                 OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
                 categorical_indices)
            )
            self.categorical_features = n_categorical
            logger.info("Added %d categorical features", n_categorical)

        if not transformers:
            raise ValueError("At least one type of feature (continuous or categorical) must be provided")

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'
        )

        return preprocessor

    def _combine_features(self,
                         X_continuous: Optional[np.ndarray] = None,
                         X_categorical: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Combine continuous and categorical features

        Args:
            X_continuous: Continuous features matrix
            X_categorical: Categorical features matrix

        Returns:
            Combined features matrix
        """
        features = []

        if X_continuous is not None:
            if len(X_continuous.shape) == 1:
                X_continuous = X_continuous.reshape(-1, 1)
            features.append(X_continuous)

        if X_categorical is not None:
            if len(X_categorical.shape) == 1:
                X_categorical = X_categorical.reshape(-1, 1)
            features.append(X_categorical)

        if not features:
            raise ValueError("At least one type of feature (continuous or categorical) must be provided")

        X_combined = np.hstack(features)
        return X_combined

    def fit(self,
            y: np.ndarray,
            X_continuous: Optional[np.ndarray] = None,
            X_categorical: Optional[np.ndarray] = None,
            cv_folds: int = 5) -> Dict[str, any]:
        """
        Train SVM regression model

        Args:
            y: Dependent variable vector
            X_continuous: Continuous features matrix
            X_categorical: Categorical features matrix
            cv_folds: Cross-validation folds

        Returns:
            Training results dictionary, including best parameters and cross-validation scores
        """
        # Check input
        if X_continuous is None and X_categorical is None:
            raise ValueError("At least one type of feature (continuous or categorical) must be provided")

        y = np.asarray(y).ravel()

        # Combine features
        X_combined = self._combine_features(X_continuous, X_categorical)

        n_samples = X_combined.shape[0]
        n_features = X_combined.shape[1]

        if len(y) != n_samples:
            raise ValueError(
                f"Length of y ({len(y)}) does not match number of feature samples ({n_samples})"
            )

        logger.info("Starting training SVM: %d samples, %d features", n_samples, n_features)

        # Create preprocessor
        self.preprocessor = self._create_preprocessor(X_continuous, X_categorical)

        # Fit preprocessor and transform data
        X_transformed = self.preprocessor.fit_transform(X_combined)
        logger.info("Feature preprocessing completed, transformed feature dimension: %d", X_transformed.shape[1])

        # Hyperparameter tuning
        if self.auto_tune:
            logger.info("Starting hyperparameter tuning...")
            self.model = self._tune_hyperparameters(X_transformed, y, cv_folds)
        else:
            # Train with specified parameters
            self.model = SVR(
                kernel=self.kernel,
                C=self.C,
                epsilon=self.epsilon,
                gamma=self.gamma
            )
            self.model.fit(X_transformed, y)

            # Cross-validation evaluation
            cv_scores = cross_val_score(
                self.model, X_transformed, y,
                cv=cv_folds,
                scoring='r2'
            )
            self.cv_results = {
                'mean_cv_score': cv_scores.mean(),
                'std_cv_score': cv_scores.std(),
                'cv_scores': cv_scores
            }
            logger.info("Cross-validation R² score: %.4f (+/- %.4f)", cv_scores.mean(), cv_scores.std())

        self.is_fitted = True

        # Calculate performance on training set
        y_pred = self.model.predict(X_transformed)
        train_r2 = r2_score(y, y_pred)
        train_rmse = np.sqrt(mean_squared_error(y, y_pred))
        train_mae = mean_absolute_error(y, y_pred)

        results = {
            'train_r2': train_r2,
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'n_samples': n_samples,
            'n_features': n_features,
            'n_transformed_features': X_transformed.shape[1],
            'cv_results': self.cv_results,
            'best_params': self.best_params
        }

        logger.info("Training completed - R²: %.4f, RMSE: %.4f, MAE: %.4f", train_r2, train_rmse, train_mae)

        return results

    def _tune_hyperparameters(self,
                             X: np.ndarray,
                             y: np.ndarray,
                             cv_folds: int = 5) -> SVR:
        """
        Use grid search for hyperparameter tuning

        Args:
            X: Feature matrix
            y: Target variable
            cv_folds: Cross-validation folds

        Returns:
            Best SVR model
        """
        # Define parameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 0.2],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
        }

        if self.kernel == 'rbf':
            # RBF kernel parameters
            pass
        elif self.kernel == 'linear':
            # Linear kernel does not need gamma
            param_grid.pop('gamma', None)

        svr = SVR(kernel=self.kernel)

        grid_search = GridSearchCV(
            svr,
            param_grid,
            cv=cv_folds,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X, y)

        self.best_params = grid_search.best_params_
        self.cv_results = {
            'mean_cv_score': grid_search.best_score_,
            'best_params': self.best_params,
            'all_results': grid_search.cv_results_
        }

        logger.info("Best parameters: %s", self.best_params)
        logger.info("Best cross-validation R² score: %.4f", grid_search.best_score_)

        return grid_search.best_estimator_

    def predict(self,
                X_continuous: Optional[np.ndarray] = None,
                X_categorical: Optional[np.ndarray] = None,
                return_weighted: bool = True) -> np.ndarray:
        """
        Predict weighted results for new samples

        Args:
            X_continuous: Continuous features matrix
            X_categorical: Categorical features matrix
            return_weighted: Whether to return weighted results (default True)

        Returns:
            Weighted results for predicted samples
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call fit() method first.")

        # Combine features
        X_combined = self._combine_features(X_continuous, X_categorical)

        # Preprocess
        X_transformed = self.preprocessor.transform(X_combined)

        # Predict
        y_pred = self.model.predict(X_transformed)

        logger.info("Completed prediction for %d samples", len(y_pred))

        # Weighted results (here using decision function values as weights
        if return_weighted:
            # For SVR, directly return predicted values as weighted results
            # Also using decision_function to get support vector distances
            if hasattr(self.model, 'decision_function'):
                decision_values = self.model.decision_function(X_transformed)
                # Normalize decision values as weights
                weights = np.abs(decision_values) / np.max(np.abs(decision_values))
                weighted_pred = y_pred * weights
                logger.info("Returning weighted prediction results")
                return weighted_pred

        return y_pred

    def evaluate(self,
                 y_true: np.ndarray,
                 X_continuous: Optional[np.ndarray] = None,
                 X_categorical: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate model performance on test set

        Args:
            y_true: True target values
            X_continuous: Continuous features matrix
            X_categorical: Categorical features matrix

        Returns:
            Evaluation metrics dictionary
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call fit() method first.")

        y_pred = self.predict(X_continuous, X_categorical, return_weighted=False)

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        # Calculate relative error
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

        metrics = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }

        logger.info("Evaluation results - R²: %.4f, RMSE: %.4f, MAE: %.4f, MAPE: %.2f%%", r2, rmse, mae, mape)

        return metrics

    def get_support_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get support vector information

        Returns:
            (support_vectors, support_vector_indices) tuple
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call fit() method first.")

        return self.model.support_vectors_, self.model.support_

    def get_model_info(self) -> Dict[str, any]:
        """
        Get model detailed information

        Returns:
            Model information dictionary
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call fit() method first.")

        info = {
            'kernel': self.kernel,
            'n_support_vectors': len(self.model.support_vectors_),
            'intercept': self.model.intercept_[0] if len(self.model.intercept_) > 0 else None,
            'C': self.model.C,
            'epsilon': self.model.epsilon,
            'gamma': self.model.gamma,
            'continuous_features': self.continuous_features,
            'categorical_features': self.categorical_features,
            'best_params': self.best_params
        }

        return info

    def __repr__(self) -> str:
        """String representation"""
        if self.is_fitted:
            n_sv = len(self.model.support_vectors_)
            return (f"KernelSVMRegressor(kernel={self.kernel}, "
                   f"fitted=True, n_support_vectors={n_sv})")
        else:
            return f"KernelSVMRegressor(kernel={self.kernel}, fitted=False)"


def train_multiple_kernels(
    y: np.ndarray,
    X_continuous: Optional[np.ndarray] = None,
    X_categorical: Optional[np.ndarray] = None,
    kernels: List[str] = ['linear', 'rbf', 'poly'],
    auto_tune: bool = True,
    cv_folds: int = 5
) -> Dict[str, KernelSVMRegressor]:
    """
    Train multiple SVM models with different kernels and compare

    Args:
        y: Dependent variable vector
        X_continuous: Continuous features matrix
        X_categorical: Categorical features matrix
        kernels: List of kernels to try
        auto_tune: Whether to perform hyperparameter tuning
        cv_folds: Cross-validation folds

    Returns:
        Trained model dictionary
    """
    results = {}

    logger.info("Starting training %d SVM models with different kernels", len(kernels))
    logger.info("=" * 60)

    for kernel in kernels:
        logger.info("\nTraining %s kernel...", kernel)

        model = KernelSVMRegressor(kernel=kernel, auto_tune=auto_tune)
        train_results = model.fit(
            y=y,
            X_continuous=X_continuous,
            X_categorical=X_categorical,
            cv_folds=cv_folds
        )

        results[kernel] = {
            'model': model,
            'train_results': train_results
        }

        logger.info(
            "%s kernel: R² = %.4f, RMSE = %.4f", kernel, train_results['train_r2'], train_results['train_rmse']
        )

    # Find best model
    best_kernel = max(results.keys(), key=lambda k: results[k]['train_results']['train_r2'])
    logger.info("\n" + "=" * 60)
    logger.info("Best kernel: %s", best_kernel)
    logger.info("R² = %.4f", results[best_kernel]['train_results']['train_r2'])

    return results
