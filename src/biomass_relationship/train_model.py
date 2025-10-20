"""
模型训练器 - 使用留一法(LOOCV)交叉验证

集成NLS、SVM、决策树模型的训练和验证流程
"""
import os
import logging
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from .model import BiomassModel

from .svm import KernelSVMRegressor
from .shap import SHAPAnalyzer
from .nls.nls_model import (
    LinearModel,
    AdditiveModel,
    MichaelisMentenModel,
    ExponentialModel,
)

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Nitrogen Deposition and Vegetation Root Biomass Relationship Model Trainer

    Use Leave-One-Out Cross-Validation (LOOCV) to evaluate multiple statistical models:
    - Non-linear least squares (NLS) model
    - Support vector machine (SVM) regression
    - Decision tree model (XGBoost, LightGBM)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray,group: int,output_dir: str,random_state: int = 42):
        self.X = X
        self.y = y
        self.group = group
        self.output_dir = output_dir
        self.random_state = random_state
        os.makedirs(output_dir, exist_ok=True)
        self.group_dir = os.path.join(output_dir, f'group_{group}')
        os.makedirs(self.group_dir, exist_ok=True)
        self.nls_models = {}
        self.svm_model = None
        self.xgb_model = None
        self.lgb_model = None
        self.nls_cv_results = {}
        self.svm_cv_results = {}
        self.xgb_cv_results = {}
        self.lgb_cv_results = {}
        logger.info(f"Initialized ModelTrainer - Group: {group}, Number of samples: {len(y)}, Number of features: {X.shape[1]}")

    def _loocv_evaluate(self,
                        model_fit_func,
                        model_predict_func,
                        model_name: str = "Model") -> Dict[str, Any]:
        """
        Use Leave-One-Out Cross-Validation (LOOCV) to evaluate the model

        Args:
            model_fit_func: Model fitting function, accepts (X_train, y_train)
            model_predict_func: Model prediction function, accepts (X_test), returns prediction values
            model_name: Model name (for logging)

        Returns:
            Dictionary containing LOOCV evaluation results
        """
        logger.info(f"Starting LOOCV for {model_name}...")
        loo = LeaveOneOut()
        y_true_list = []
        y_pred_list = []
        fold_errors = []
        for fold_idx, (train_index, test_index) in enumerate(loo.split(self.X)):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            try:
                model_fit_func(X_train, y_train)
                y_pred = model_predict_func(X_test)
                y_true_list.append(y_test[0])
                y_pred_list.append(y_pred[0] if isinstance(y_pred, np.ndarray) else y_pred)
                fold_error = np.abs(y_test[0] - y_pred_list[-1])
                fold_errors.append(fold_error)

            except Exception as e:
                logger.warning(f"Fold {fold_idx + 1}/{len(self.y)} failed: {str(e)}")
                continue

        y_true_array = np.array(y_true_list)
        y_pred_array = np.array(y_pred_list)

        r2 = r2_score(y_true_array, y_pred_array)
        rmse = np.sqrt(mean_squared_error(y_true_array, y_pred_array))
        mae = mean_absolute_error(y_true_array, y_pred_array)
        mape = np.mean(np.abs((y_true_array - y_pred_array) / (y_true_array + 1e-10))) * 100

        results = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'y_true': y_true_array,
            'y_pred': y_pred_array,
            'fold_errors': fold_errors,
            'n_successful_folds': len(y_true_list)
        }

        logger.info(f"  LOOCV completed - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        return results

    def _create_biomass_model(
        self, 
        results: Dict[str, Any], 
        model_type: str,
        training_config: Dict[str, Any] = None
    ) -> 'BiomassModel':
        """
        Select best model from results and wrap it in BiomassModel
        
        Args:
            results: Dictionary of model results
            model_type: Type of model ('NLS', 'SVM', 'DecisionTree')
            training_config: Additional training configuration
            
        Returns:
            BiomassModel with best performing model
        """
        
        # Select best model based on LOOCV R²
        best_model_name = max(results.keys(), key=lambda k: results[k]['cv_results']['r2'])
        best_result = results[best_model_name]
        best_cv_results = best_result['cv_results']
        
        logger.info(f"\nBest {model_type} model selected: {best_model_name}")
        logger.info(f"  LOOCV R² = {best_cv_results['r2']:.4f}")
        logger.info(f"  LOOCV RMSE = {best_cv_results['rmse']:.4f}")
        
        # Prepare feature names
        feature_names = ['N_addition'] + [f'PC{i+1}' for i in range(self.X.shape[1])]
        
        # Prepare performance metrics
        performance_metrics = {
            'r2': best_cv_results['r2'],
            'rmse': best_cv_results['rmse'],
            'mae': best_cv_results['mae'],
            'mape': best_cv_results.get('mape', 0.0),
            'n_successful_folds': best_cv_results.get('n_successful_folds', len(self.y))
        }
        
        # Merge training config
        config = {
            'n_samples': len(self.y),
            'n_features': self.X.shape[1],
            'model_type': model_type,
            'model_variant': best_model_name
        }
        if training_config:
            config.update(training_config)
        
        # Create BiomassModel wrapper
        best_model_wrapper = BiomassModel(
            model_name=f'{model_type}_{best_model_name}',
            model_type=model_type,
            trained_model=best_result['model'],
            climate_group=self.group,
            performance_metrics=performance_metrics,
            pca_analyzer=None,  # Will be set by processor
            n_impact_calculator=None,  # Will be set by processor
            feature_names=feature_names,
            training_config=config,
            model_variant=best_model_name
        )
        
        return best_model_wrapper

    def train_nls(self, 
                  n_addtion: np.ndarray, 
                  alpha_l1: float = 0.0,
                  alpha_l2: float = 0.0,
                  auto_regularization: bool = True,
                  save_to_file: bool = True, 
                  plot: bool = False) -> BiomassModel:
        """
        Train Non-linear least squares (NLS) model, using LOOCV validation

        Args:
            n_addtion: Nitrogen addition rate
            alpha_l1: L1 regularization parameter (Lasso), default 0.0
                      Promotes parameter sparsity for feature selection
            alpha_l2: L2 regularization parameter (Ridge), default 0.0
                      Promotes small parameter values for stability
            auto_regularization: If True and no regularization provided, automatically
                                suggest regularization based on overfitting risk
            save_to_file: Whether to save results to file
            plot: Whether to generate visualization plots

        Returns:
            NLS model training and validation results
        """
        logger.info("=" * 70)
        logger.info(f"Starting training NLS model - Group {self.group}")
        logger.info("=" * 70)
        
        # Check overfitting risk and suggest regularization if needed
        if auto_regularization and alpha_l1 == 0.0 and alpha_l2 == 0.0:
            n_samples = len(self.y)
            n_features = self.X.shape[1] + 1  # +1 for nitrogen_add
            # Estimate max parameters (for AdditiveModel: 1 + n_features + n_features)
            max_params = 1 + 2 * n_features
            param_ratio = max_params / n_samples
            
            logger.info(f"\nOverfitting Risk Assessment:")
            logger.info(f"  Samples: {n_samples}, Features: {n_features}, Max params: {max_params}")
            logger.info(f"  Parameter/Sample ratio: {param_ratio:.3f}")
            
            if param_ratio > 0.5:
                alpha_l1 = 0.1
                alpha_l2 = 0.05
                logger.warning(f"  HIGH overfitting risk detected!")
                logger.warning(f"  Auto-enabling regularization: L1={alpha_l1}, L2={alpha_l2}")
                logger.warning(f"  To disable: set auto_regularization=False")
            elif param_ratio > 0.2:
                alpha_l1 = 0.05
                alpha_l2 = 0.01
                logger.warning(f"  MODERATE overfitting risk detected!")
                logger.warning(f"  Auto-enabling regularization: L1={alpha_l1}, L2={alpha_l2}")
                logger.warning(f"  To disable: set auto_regularization=False")
            elif param_ratio > 0.1:
                alpha_l1 = 0.01
                alpha_l2 = 0.001
                logger.info(f"  ℹ️  LOW overfitting risk, applying light regularization")
                logger.info(f"  Using: L1={alpha_l1}, L2={alpha_l2}")
            else:
                logger.info(f"  ✓ MINIMAL overfitting risk - no regularization needed")
        
        if alpha_l1 > 0 or alpha_l2 > 0:
            logger.info(f"\nRegularization enabled:")
            logger.info(f"  L1 (Lasso) α = {alpha_l1} - promotes sparsity")
            logger.info(f"  L2 (Ridge) α = {alpha_l2} - promotes stability")

        models = [
            LinearModel(),
            AdditiveModel(),
            MichaelisMentenModel(),
            ExponentialModel('v2')
        ]

        all_results = {}

        for model in models:
            model_name = model.name
            logger.info(f"\nTraining NLS model: {model_name}")

            # Define model fitting and prediction functions
            def fit_func(X_train, y_train):
                X_input = np.column_stack([n_addtion[:len(y_train)], X_train])
                model.fit(
                    X_input, 
                    y_train,
                    alpha_l1=alpha_l1,
                    alpha_l2=alpha_l2,
                    check_overfitting=False  # Already checked at trainer level
                )

            def predict_func(X_test):
                X_input = np.column_stack([n_addtion[:len(X_test)], X_test])
                return model.predict(X_input)

            # LOOCV evaluation
            cv_results = self._loocv_evaluate(fit_func, predict_func, model_name)

            # Train final model on full data
            X_full = np.column_stack([n_addtion, self.X])
            final_result = model.fit(
                X_full, 
                self.y,
                alpha_l1=alpha_l1,
                alpha_l2=alpha_l2,
                check_overfitting=False  # Already checked at trainer level
            )

            all_results[model_name] = {
                'model': model,
                'cv_results': cv_results,
                'final_fit': final_result,
                'summary': model.summary()
            }

            self.nls_models[model_name] = model
            self.nls_cv_results[model_name] = cv_results

        # Compare all NLS models
        comparison_df = self._compare_nls_results(all_results)

        # Save results
        if save_to_file:
            self._save_nls_results(all_results, comparison_df, alpha_l1, alpha_l2)

        # Visualize results
        if plot:
            self._plot_nls_results(all_results)

        logger.info(f"\nNLS model training completed - Group {self.group}")

        # Create BiomassModel with best NLS model
        training_config = {
            'alpha_l1': alpha_l1,
            'alpha_l2': alpha_l2
        }
        return self._create_biomass_model(all_results, 'NLS', training_config)

    def train_svm(self,
                  n_addtion: np.ndarray,
                  kernels: List[str] = None,
                  auto_tune: bool = True,
                  intensive_search: bool = True,
                  save_to_file: bool = True, 
                  plot: bool = False) -> BiomassModel:
        """
        Train SVM regression model with nitrogen addition as a feature, using LOOCV validation

        Args:
            n_addtion: Nitrogen addition rate (cumulative N deposition)
            kernels: List of kernels to test
            auto_tune: Whether to perform hyperparameter tuning (recommended)
            intensive_search: If True, use more extensive parameter search
            save_to_file: Whether to save results to file
            plot: Whether to generate visualization plots

        Returns:
            SVM model training and validation results
        """
        # Validate n_addition
        if n_addtion is None or len(n_addtion) != len(self.y):
            raise ValueError(f"n_addtion must be provided and match y length: {len(self.y)}")
        
        # Set default kernel list
        if kernels is None:
            kernels = ['rbf', 'linear', 'poly']
            
        logger.info("=" * 70)
        logger.info(f"Starting SVM training - Group {self.group}")
        logger.info("=" * 70)
        logger.info(f"Samples: {len(self.y)}, Features: {self.X.shape[1]} + 1 (N addition)")
        logger.info(f"Hyperparameter tuning: {'ENABLED' if auto_tune else 'DISABLED'}")
        logger.info(f"Search mode: {'INTENSIVE' if intensive_search else 'STANDARD'}")
        logger.info(f"Testing kernels: {', '.join(kernels)}")
        
        # Warn about prediction uniformity issue
        n_samples = len(self.y)
        if n_samples < 10:
            logger.warning("Small sample size may cause uniform predictions")
            logger.warning("Consider using auto_tune=True and intensive_search=True")

        all_results = {}

        for kernel in kernels:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training SVM with kernel: {kernel}")
            logger.info(f"{'='*50}")

            # Create SVM model with appropriate settings
            if auto_tune:
                # Use auto-tuning to find best parameters
                logger.info("Grid search enabled - searching for optimal hyperparameters...")
                svm_model = KernelSVMRegressor(
                    kernel=kernel,
                    auto_tune=True,
                    C=1.0,  # Initial value (will be optimized)
                    epsilon=0.1,  # Initial value (will be optimized)
                    gamma='scale'  # Initial value (will be optimized)
                )
            else:
                # Use more aggressive parameters for better fitting
                if intensive_search:
                    # Higher C and lower epsilon for stronger fitting
                    C_val = 100.0 if kernel in ['rbf', 'poly'] else 10.0
                    epsilon_val = 0.01
                else:
                    C_val = 1.0
                    epsilon_val = 0.1
                    
                svm_model = KernelSVMRegressor(
                    kernel=kernel,
                    auto_tune=False,
                    C=C_val,
                    epsilon=epsilon_val,
                    gamma='scale'
                )
                logger.info(f"Manual parameters: C={C_val}, epsilon={epsilon_val}")

            # Define fitting and prediction functions with N addition as feature
            def fit_func(X_train, y_train):
                # Combine N addition with PCA features
                X_input = np.column_stack([n_addtion[:len(X_train)], X_train])
                svm_model.fit(y=y_train, X_continuous=X_input)

            def predict_func(X_test):
                # Combine N addition with PCA features for prediction
                X_input = np.column_stack([n_addtion[:len(X_test)], X_test])
                return svm_model.predict(X_continuous=X_input, return_weighted=False)

            # LOOCV evaluation
            logger.info("Starting LOOCV evaluation...")
            cv_results = self._loocv_evaluate(fit_func, predict_func, f"SVM-{kernel}")

            # Train final model on full data with N addition
            logger.info("Training final model on full dataset...")
            X_full = np.column_stack([n_addtion, self.X])
            final_results = svm_model.fit(y=self.y, X_continuous=X_full)

            # Store results with additional info
            result_dict = {
                'model': svm_model,
                'cv_results': cv_results,
                'train_results': final_results
            }
            
            # Add best parameters if auto-tuning was used
            if auto_tune and hasattr(svm_model, 'best_params') and svm_model.best_params:
                result_dict['best_params'] = svm_model.best_params
                logger.info(f"  Best parameters found: {svm_model.best_params}")
            
            all_results[kernel] = result_dict
            self.svm_cv_results[kernel] = cv_results

        # Select best kernel based on LOOCV performance
        best_kernel = max(all_results.keys(), 
                         key=lambda k: all_results[k]['cv_results']['r2'])
        self.svm_model = all_results[best_kernel]['model']

        logger.info(f"\n{'='*50}")
        logger.info(f"Best SVM kernel: {best_kernel}")
        logger.info(f"  LOOCV R²: {all_results[best_kernel]['cv_results']['r2']:.4f}")
        logger.info(f"  LOOCV RMSE: {all_results[best_kernel]['cv_results']['rmse']:.4f}")
        if 'best_params' in all_results[best_kernel]:
            logger.info(f"  Optimized parameters: {all_results[best_kernel]['best_params']}")
        logger.info(f"{'='*50}")

        # Save results
        if save_to_file:
            self._save_svm_results(all_results, best_kernel)

        # Visualize results
        if plot:
            self._plot_svm_results(all_results)

        logger.info(f"\nSVM model training completed - Group {self.group}")

        # Create BiomassModel with best SVM model
        training_config = {
            'kernels': kernels,
            'auto_tune': auto_tune,
            'intensive_search': intensive_search,
            'best_kernel': best_kernel
        }
        # Add best parameters if available
        if 'best_params' in all_results[best_kernel]:
            training_config['best_params'] = all_results[best_kernel]['best_params']
        
        return self._create_biomass_model(all_results, 'SVM', training_config)

    def train_decision_tree(self, 
                            n_addtion: np.ndarray,
                            max_depth: int = 5,
                            min_samples_split: int = 2,
                            min_samples_leaf: int = 1,
                            learning_rate: float = 0.1,
                            n_estimators: int = 100,
                            save_to_file: bool = True, 
                            plot: bool = False) -> BiomassModel:
        """
        Train decision tree models (XGBoost and LightGBM) with N addition as feature,
        using LOOCV validation and SHAP analysis

        Args:
            n_addtion: Nitrogen addition rate (cumulative N deposition)
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum number of samples required to split an internal node
            min_samples_leaf: Minimum number of samples required to be at a leaf node
            learning_rate: Learning rate for gradient boosting
            n_estimators: Number of boosting iterations
            save_to_file: Whether to save results to file
            plot: Whether to generate visualization plots

        Returns:
            Decision tree model training and validation results with SHAP analysis
        """
        # Validate n_addition
        if n_addtion is None or len(n_addtion) != len(self.y):
            raise ValueError(f"n_addtion must be provided and match y length: {len(self.y)}")
            
        logger.info("=" * 70)
        logger.info(f"Starting decision tree training - Group {self.group}")
        logger.info("=" * 70)
        logger.info(f"Samples: {len(self.y)}, Features: {self.X.shape[1]} + 1 (N addition)")
        logger.info(f"Hyperparameters: max_depth={max_depth}, learning_rate={learning_rate}, n_estimators={n_estimators}")

        all_results = {}

        # Prepare feature names for SHAP analysis
        feature_names = ['N_addition'] + [f'PC{i+1}' for i in range(self.X.shape[1])]
        logger.info(f"Feature names: {feature_names}")

        # Train XGBoost
        logger.info("\n" + "="*50)
        logger.info("Training XGBoost model...")
        logger.info("="*50)
        
        xgb_params = {
            'max_depth': max_depth,
            'min_child_weight': min_samples_leaf,  # XGBoost equivalent of min_samples_leaf
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'random_state': self.random_state,
            'verbosity': 0
        }
        logger.info(f"XGBoost parameters: {xgb_params}")

        xgb_model = XGBRegressor(**xgb_params)

        def xgb_fit(X_train, y_train):
            # Combine N addition with PCA features
            X_input = np.column_stack([n_addtion[:len(X_train)], X_train])
            xgb_model.fit(X_input, y_train)

        def xgb_predict(X_test):
            # Combine N addition with PCA features for prediction
            X_input = np.column_stack([n_addtion[:len(X_test)], X_test])
            return xgb_model.predict(X_input)

        logger.info("Starting LOOCV evaluation for XGBoost...")
        xgb_cv_results = self._loocv_evaluate(xgb_fit, xgb_predict, "XGBoost")

        # Train on full dataset with N addition
        logger.info("Training XGBoost on full dataset...")
        X_full = np.column_stack([n_addtion, self.X])
        xgb_model.fit(X_full, self.y)
        y_pred_train = xgb_model.predict(X_full)
        xgb_train_r2 = r2_score(self.y, y_pred_train)
        xgb_train_rmse = np.sqrt(mean_squared_error(self.y, y_pred_train))
        
        logger.info(f"XGBoost training completed - Train R²: {xgb_train_r2:.4f}, RMSE: {xgb_train_rmse:.4f}")

        all_results['XGBoost'] = {
            'model': xgb_model,
            'cv_results': xgb_cv_results,
            'train_r2': xgb_train_r2,
            'train_rmse': xgb_train_rmse,
            'feature_names': feature_names
        }

        self.xgb_model = xgb_model
        self.xgb_cv_results = xgb_cv_results

        # Train LightGBM
        logger.info("\n" + "="*50)
        logger.info("Training LightGBM model...")
        logger.info("="*50)
        
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'min_child_samples': min_samples_leaf,
            'random_state': self.random_state,
            'verbose': -1
        }
        logger.info(f"LightGBM parameters: {lgb_params}")

        lgb_model = LGBMRegressor(**lgb_params)

        def lgb_fit(X_train, y_train):
            # Combine N addition with PCA features
            X_input = np.column_stack([n_addtion[:len(X_train)], X_train])
            lgb_model.fit(X_input, y_train)

        def lgb_predict(X_test):
            # Combine N addition with PCA features for prediction
            X_input = np.column_stack([n_addtion[:len(X_test)], X_test])
            return lgb_model.predict(X_input)

        logger.info("Starting LOOCV evaluation for LightGBM...")
        lgb_cv_results = self._loocv_evaluate(lgb_fit, lgb_predict, "LightGBM")

        # Train on full dataset with N addition
        logger.info("Training LightGBM on full dataset...")
        lgb_model.fit(X_full, self.y)
        y_pred_train = lgb_model.predict(X_full)
        lgb_train_r2 = r2_score(self.y, y_pred_train)
        lgb_train_rmse = np.sqrt(mean_squared_error(self.y, y_pred_train))
        
        logger.info(f"LightGBM training completed - Train R²: {lgb_train_r2:.4f}, RMSE: {lgb_train_rmse:.4f}")

        all_results['LightGBM'] = {
            'model': lgb_model,
            'cv_results': lgb_cv_results,
            'train_r2': lgb_train_r2,
            'train_rmse': lgb_train_rmse,
            'feature_names': feature_names
        }

        self.lgb_model = lgb_model
        self.lgb_cv_results = lgb_cv_results

        # Compare models
        logger.info("\n" + "="*70)
        logger.info("Decision Tree Model Comparison:")
        logger.info("="*70)
        logger.info(f"XGBoost  - LOOCV R²: {xgb_cv_results['r2']:.4f}, RMSE: {xgb_cv_results['rmse']:.4f}")
        logger.info(f"LightGBM - LOOCV R²: {lgb_cv_results['r2']:.4f}, RMSE: {lgb_cv_results['rmse']:.4f}")

        # SHAP analysis - Always perform for decision trees
        logger.info("\n" + "="*70)
        logger.info("Performing SHAP Feature Importance Analysis...")
        logger.info("="*70)
        
        # Prepare data for SHAP analysis
        X_df = pd.DataFrame(X_full, columns=feature_names)
        
        # SHAP analysis for XGBoost
        logger.info("\nAnalyzing XGBoost feature contributions with SHAP...")
        xgb_shap_results = self._shap_analysis(xgb_model, "XGBoost", X_df, save_plots=save_to_file or plot)
        all_results['XGBoost']['shap_results'] = xgb_shap_results
        
        # SHAP analysis for LightGBM
        logger.info("\nAnalyzing LightGBM feature contributions with SHAP...")
        lgb_shap_results = self._shap_analysis(lgb_model, "LightGBM", X_df, save_plots=save_to_file or plot)
        all_results['LightGBM']['shap_results'] = lgb_shap_results

        if save_to_file:
            self._save_decision_tree_results(all_results)

        if plot:
            self._plot_decision_tree_results(all_results)

        logger.info(f"\nDecision tree model training completed - Group {self.group}")

        # Create BiomassModel with best Decision Tree model
        training_config = {
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators
        }
        return self._create_biomass_model(all_results, 'DecisionTree', training_config)

    def _compare_nls_results(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Compare NLS model results"""
        comparison_data = []

        for model_name, result in results.items():
            cv_res = result['cv_results']
            final_res = result['final_fit']

            comparison_data.append({
                'Model': model_name,
                'LOOCV_R²': cv_res['r2'],
                'LOOCV_RMSE': cv_res['rmse'],
                'LOOCV_MAE': cv_res['mae'],
                'Train_R²': final_res.r2,
                'Train_RMSE': final_res.rmse,
                'AIC': final_res.aic,
                'BIC': final_res.bic,
                'Num_Params': len(final_res.params)
            })

        df = pd.DataFrame(comparison_data)
        df = df.sort_values('LOOCV_R²', ascending=False)

        logger.info("\nNLS model comparison (sorted by LOOCV R²):")
        logger.info("\n" + df.to_string(index=False))

        return df

    def _save_nls_results(self, results: Dict[str, Any], comparison_df: pd.DataFrame, 
                          alpha_l1: float = 0.0, alpha_l2: float = 0.0):
        """Save NLS results"""
        # Save comparison results
        comparison_path = os.path.join(self.group_dir, 'nls_model_comparison.csv')
        comparison_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
        logger.info(f"NLS model comparison results saved: {comparison_path}")
        
        # Save regularization settings
        if alpha_l1 > 0 or alpha_l2 > 0:
            reg_info_path = os.path.join(self.group_dir, 'nls_regularization_info.txt')
            with open(reg_info_path, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write("NLS Model Regularization Settings\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"L1 Regularization (Lasso): α = {alpha_l1}\n")
                f.write(f"  - Promotes parameter sparsity\n")
                f.write(f"  - Automatic feature selection\n\n")
                f.write(f"L2 Regularization (Ridge): α = {alpha_l2}\n")
                f.write(f"  - Promotes small parameter values\n")
                f.write(f"  - Improves model stability\n\n")
                if alpha_l1 > 0 and alpha_l2 > 0:
                    f.write("Using Elastic Net regularization (L1 + L2)\n")
                f.write("\n" + "=" * 70 + "\n")
            logger.info(f"Regularization settings saved: {reg_info_path}")

        # Save detailed results for each model
        for model_name, result in results.items():
            model_dir = os.path.join(self.group_dir, 'nls', model_name.replace(' ', '_'))
            os.makedirs(model_dir, exist_ok=True)

            # Save model summary
            summary_path = os.path.join(model_dir, 'summary.txt')
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(result['summary'])

            # Save LOOCV prediction results
            cv_res = result['cv_results']
            cv_df = pd.DataFrame({
                'y_true': cv_res['y_true'],
                'y_pred': cv_res['y_pred'],
                'error': cv_res['y_true'] - cv_res['y_pred']
            })
            cv_path = os.path.join(model_dir, 'loocv_predictions.csv')
            cv_df.to_csv(cv_path, index=False)

    def _save_svm_results(self, results: Dict[str, Any], best_kernel: str):
        """Save SVM results"""
        svm_dir = os.path.join(self.group_dir, 'svm')
        os.makedirs(svm_dir, exist_ok=True)

        # Save comparison results
        comparison_data = []
        for kernel, result in results.items():
            cv_res = result['cv_results']
            row_data = {
                'Kernel': kernel,
                'LOOCV_R²': cv_res['r2'],
                'LOOCV_RMSE': cv_res['rmse'],
                'LOOCV_MAE': cv_res['mae'],
                'Train_R²': result['train_results']['train_r2'],
                'Train_RMSE': result['train_results']['train_rmse'],
                'Best': 'Yes' if kernel == best_kernel else 'No'
            }
            
            # Add best parameters if available
            if 'best_params' in result:
                for param_name, param_value in result['best_params'].items():
                    row_data[f'Best_{param_name}'] = param_value
                    
            comparison_data.append(row_data)

        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = os.path.join(svm_dir, 'svm_kernel_comparison.csv')
        comparison_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
        logger.info(f"SVM model comparison results saved: {comparison_path}")
        
        # Save detailed best parameters info
        best_params_path = os.path.join(svm_dir, 'best_parameters_info.txt')
        with open(best_params_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("SVM Best Model Parameters\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Best Kernel: {best_kernel}\n\n")
            
            best_result = results[best_kernel]
            if 'best_params' in best_result:
                f.write("Optimized Parameters (from Grid Search):\n")
                for param_name, param_value in best_result['best_params'].items():
                    f.write(f"  {param_name}: {param_value}\n")
                f.write("\nParameter Meanings:\n")
                f.write("  C: Regularization parameter (higher = stronger fitting)\n")
                f.write("  epsilon: Width of epsilon-tube (lower = tighter predictions)\n")
                f.write("  gamma: Kernel coefficient (for RBF/poly kernels)\n")
            else:
                f.write("Manual Parameters Used:\n")
                f.write(f"  (No auto-tuning performed)\n")
                
            f.write("\nPerformance:\n")
            f.write(f"  LOOCV R²: {best_result['cv_results']['r2']:.4f}\n")
            f.write(f"  LOOCV RMSE: {best_result['cv_results']['rmse']:.4f}\n")
            f.write(f"  Train R²: {best_result['train_results']['train_r2']:.4f}\n")
            f.write(f"  Train RMSE: {best_result['train_results']['train_rmse']:.4f}\n")
            f.write("\n" + "=" * 70 + "\n")
        logger.info(f"Best parameters info saved: {best_params_path}")

    def _save_decision_tree_results(self, results: Dict[str, Any]):
        """
        Save decision tree results including SHAP analysis
        """
        dt_dir = os.path.join(self.group_dir, 'decision_tree')
        os.makedirs(dt_dir, exist_ok=True)

        logger.info("Saving decision tree results...")

        # Save comparison results
        comparison_data = []
        for model_name, result in results.items():
            cv_res = result['cv_results']
            row_data = {
                'Model': model_name,
                'LOOCV_R²': cv_res['r2'],
                'LOOCV_RMSE': cv_res['rmse'],
                'LOOCV_MAE': cv_res['mae'],
                'Train_R²': result['train_r2'],
                'Train_RMSE': result['train_rmse']
            }
            
            # Add SHAP importance for top feature if available
            if 'shap_results' in result and 'importance_df' in result['shap_results']:
                shap_df = result['shap_results']['importance_df']
                if not shap_df.empty:
                    top_feature = shap_df.iloc[0]['Feature']
                    top_importance = shap_df.iloc[0]['SHAP_Importance']
                    row_data['Top_Feature'] = top_feature
                    row_data['Top_SHAP_Value'] = top_importance
            
            comparison_data.append(row_data)

        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = os.path.join(dt_dir, 'decision_tree_comparison.csv')
        comparison_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
        logger.info(f"Decision tree comparison saved: {comparison_path}")

        # Save built-in feature importance and SHAP results
        for model_name, result in results.items():
            model = result['model']
            feature_names = result.get('feature_names', [f'PC{i+1}' for i in range(self.X.shape[1])])
            
            # Save built-in feature importance
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)

                importance_path = os.path.join(dt_dir, f'{model_name}_builtin_importance.csv')
                importance_df.to_csv(importance_path, index=False, encoding='utf-8-sig')
                logger.info(f"{model_name} built-in importance saved: {importance_path}")
            
            # Save SHAP results summary
            if 'shap_results' in result and 'importance_df' in result['shap_results']:
                shap_df = result['shap_results']['importance_df']
                if not shap_df.empty:
                    shap_summary_path = os.path.join(dt_dir, f'{model_name}_shap_importance.csv')
                    shap_df.to_csv(shap_summary_path, index=False, encoding='utf-8-sig')
                    logger.info(f"{model_name} SHAP importance saved: {shap_summary_path}")
        
        logger.info("Decision tree results saved successfully")

    def _shap_analysis(self, model, model_name: str, X_df: pd.DataFrame = None, save_plots: bool = True):
        """
        Perform SHAP feature importance analysis for tree-based models
        
        Args:
            model: Trained tree-based model (XGBoost or LightGBM)
            model_name: Name of the model for saving results
            X_df: Feature DataFrame with proper column names
            save_plots: Whether to save SHAP plots
            
        Returns:
            Dictionary with SHAP analysis results
        """
        try:
            logger.info(f"Initializing SHAP analyzer for {model_name}...")
            shap_analyzer = SHAPAnalyzer(model)
            
            # Use provided X_df or create default
            if X_df is None:
                X_df = pd.DataFrame(self.X, columns=[f'PC{i+1}' for i in range(self.X.shape[1])])

            # Create explainer for tree-based models
            logger.info("Creating SHAP TreeExplainer...")
            shap_analyzer.create_explainer(X_df, explainer_type='tree')
            
            # Calculate SHAP values
            logger.info("Calculating SHAP values...")
            shap_analyzer.calculate_shap_values(X_df)

            # Get feature importance
            importance = shap_analyzer.get_feature_importance()
            importance_df = pd.DataFrame(list(importance.items()), 
                                        columns=['Feature', 'SHAP_Importance'])
            importance_df = importance_df.sort_values('SHAP_Importance', ascending=False)
            
            logger.info(f"\nSHAP Feature Importance for {model_name}:")
            logger.info("\n" + importance_df.to_string(index=False))

            # Save results if requested
            if save_plots:
                shap_dir = os.path.join(self.group_dir, 'shap', model_name)
                os.makedirs(shap_dir, exist_ok=True)

                # Summary plot
                logger.info("Generating SHAP summary plot...")
                summary_path = os.path.join(shap_dir, 'shap_summary.png')
                shap_analyzer.summary_plot(save_path=summary_path)

                # Save feature importance CSV
                importance_path = os.path.join(shap_dir, 'shap_feature_importance.csv')
                importance_df.to_csv(importance_path, index=False, encoding='utf-8-sig')

                logger.info(f"SHAP analysis results saved to: {shap_dir}")
            
            return {
                'feature_importance': importance,
                'importance_df': importance_df,
                'shap_values': shap_analyzer.shap_values if hasattr(shap_analyzer, 'shap_values') else None
            }
            
        except Exception as e:
            logger.error(f"SHAP analysis failed for {model_name}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'feature_importance': {},
                'importance_df': pd.DataFrame(),
                'shap_values': None,
                'error': str(e)
            }

    def _plot_nls_results(self, results: Dict[str, Any]):
        """Visualize NLS results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle(f'第{self.group}组气候类型样本的多元回归模型比较', fontweight='bold', fontsize=18)
        axes = axes.flatten()

        for idx, (model_name, result) in enumerate(results.items()):
            if idx >= 4:
                break

            cv_res = result['cv_results']
            ax = axes[idx]

            ax.scatter(cv_res['y_true'], cv_res['y_pred'], alpha=0.6)

            min_val = min(cv_res['y_true'].min(), cv_res['y_pred'].min())
            max_val = max(cv_res['y_true'].max(), cv_res['y_pred'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

            ax.set_xlabel('实际值')
            ax.set_ylabel('预测值 (LOOCV)')
            ax.set_title(f'{model_name}\nR²={cv_res["r2"]:.3f}')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.group_dir, 'nls_loocv_predictions.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"NLS prediction plot saved: {plot_path}")

    def _plot_svm_results(self, results: Dict[str, Any]):
        """Visualize SVM results"""
        fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 4))
        fig.suptitle(f'第{self.group}组气候类型样本的支持向量机模型比较', fontweight='bold', fontsize=18)
        if len(results) == 1:
            axes = [axes]

        for idx, (kernel, result) in enumerate(results.items()):
            cv_res = result['cv_results']
            ax = axes[idx]

            ax.scatter(cv_res['y_true'], cv_res['y_pred'], alpha=0.6)

            min_val = min(cv_res['y_true'].min(), cv_res['y_pred'].min())
            max_val = max(cv_res['y_true'].max(), cv_res['y_pred'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

            ax.set_xlabel('实际值')
            ax.set_ylabel('预测值 (LOOCV)')
            ax.set_title(f'SVM-{kernel}\nR²={cv_res["r2"]:.3f}')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.group_dir, 'svm_loocv_predictions.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"SVM prediction plot saved: {plot_path}")

    def _plot_decision_tree_results(self, results: Dict[str, Any]):
        """Visualize decision tree results"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'第{self.group}组气候类型样本的决策树模型比较', fontweight='bold', fontsize=18)

        for idx, (model_name, result) in enumerate(results.items()):
            cv_res = result['cv_results']
            ax = axes[idx]

            ax.scatter(cv_res['y_true'], cv_res['y_pred'], alpha=0.6)

            min_val = min(cv_res['y_true'].min(), cv_res['y_pred'].min())
            max_val = max(cv_res['y_true'].max(), cv_res['y_pred'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

            ax.set_xlabel('实际值')
            ax.set_ylabel('预测值 (LOOCV)')
            ax.set_title(f'{model_name}\nR²={cv_res["r2"]:.3f}')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.group_dir, 'decision_tree_loocv_predictions.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Decision tree prediction plot saved: {plot_path}")

    def get_all_results(self) -> Dict[str, Any]:
        """
        Get summary results for all models

        Returns:
            Dictionary containing LOOCV results for all models
        """
        summary = {
            'group': self.group,
            'n_samples': len(self.y),
            'n_features': self.X.shape[1],
            'nls_results': self.nls_cv_results,
            'svm_results': self.svm_cv_results,
            'xgboost_results': self.xgb_cv_results,
            'lightgbm_results': self.lgb_cv_results
        }

        return summary
