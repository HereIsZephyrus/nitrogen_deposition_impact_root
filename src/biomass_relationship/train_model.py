"""
模型训练器 - 使用留一法(LOOCV)交叉验证

集成NLS、SVM、决策树模型的训练和验证流程
"""
import os
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from .svm import KernelSVMRegressor
from .shap import SHAPAnalyzer
from .nls.nls_model import (
    LinearModel,
    AdditiveModel, 
    MultiplicativeModel, 
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

    def train_nls(self, 
                  n_addtion: np.ndarray, 
                  alpha_l1: float = 0.0,
                  alpha_l2: float = 0.0,
                  auto_regularization: bool = True,
                  save_to_file: bool = True, 
                  plot: bool = False) -> Dict[str, Any]:
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

        return all_results

    def train_svm(self,
                  n_addtion: np.ndarray = None,  # For API compatibility, not used in SVM
                  kernels: List[str] = None,
                  auto_tune: bool = True,
                  intensive_search: bool = True,
                  save_to_file: bool = True, 
                  plot: bool = False) -> Dict[str, Any]:
        """
        Train SVM regression model, using LOOCV validation

        Args:
            n_addtion: Nitrogen addition rate
            kernels: List of kernels to test
            auto_tune: Whether to perform hyperparameter tuning (recommended)
            intensive_search: If True, use more extensive parameter search
            save_to_file: Whether to save results to file
            plot: Whether to generate visualization plots

        Returns:
            SVM model training and validation results
        """
        # Set default kernel list
        if kernels is None:
            kernels = ['rbf', 'linear', 'poly']
            
        logger.info("=" * 70)
        logger.info(f"Starting training SVM model - Group {self.group}")
        logger.info("=" * 70)
        logger.info(f"Hyperparameter tuning: {'ENABLED' if auto_tune else 'DISABLED'}")
        logger.info(f"Search mode: {'INTENSIVE' if intensive_search else 'STANDARD'}")
        logger.info(f"Testing kernels: {', '.join(kernels)}")
        
        # Warn about prediction uniformity issue
        n_samples = len(self.y)
        if n_samples < 10:
            logger.warning("   Small sample size may cause uniform predictions")
            logger.warning("   Consider using auto_tune=True and intensive_search=True")

        all_results = {}

        for kernel in kernels:
            logger.info(f"\nTraining SVM model - Kernel: {kernel}")

            # Create SVM model with appropriate settings
            if auto_tune:
                # Use auto-tuning to find best parameters
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
                logger.info(f"  Using manual parameters: C={C_val}, epsilon={epsilon_val}")

            # Define fitting and prediction functions
            def fit_func(X_train, y_train):
                svm_model.fit(y=y_train, X_continuous=X_train)

            def predict_func(X_test):
                return svm_model.predict(X_continuous=X_test, return_weighted=False)

            # LOOCV evaluation
            cv_results = self._loocv_evaluate(fit_func, predict_func, f"SVM-{kernel}")

            # Train final model on full data
            final_results = svm_model.fit(y=self.y, X_continuous=self.X)

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

        return all_results

    def train_decision_tree(self, 
                           save_to_file: bool = True, 
                           plot: bool = False) -> Dict[str, Any]:
        """
        Train decision tree models (XGBoost and LightGBM) using LOOCV validation

        Args:
            save_to_file: Whether to save results to file
            plot: Whether to generate visualization plots

        Returns:
            Decision tree model training and validation results
        """
        logger.info("=" * 70)
        logger.info(f"Starting decision tree model training - Group {self.group}")
        logger.info("=" * 70)

        all_results = {}

        # Train XGBoost
        logger.info("\nTraining XGBoost model...")
        xgb_params = {
            'max_depth': 5,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': self.random_state,
            'verbosity': 0
        }

        xgb_model = XGBRegressor(**xgb_params)

        def xgb_fit(X_train, y_train):
            xgb_model.fit(X_train, y_train)

        def xgb_predict(X_test):
            return xgb_model.predict(X_test)

        xgb_cv_results = self._loocv_evaluate(xgb_fit, xgb_predict, "XGBoost")

        # 在全数据上训练
        xgb_model.fit(self.X, self.y)
        y_pred_train = xgb_model.predict(self.X)
        xgb_train_r2 = r2_score(self.y, y_pred_train)
        xgb_train_rmse = np.sqrt(mean_squared_error(self.y, y_pred_train))

        all_results['XGBoost'] = {
            'model': xgb_model,
            'cv_results': xgb_cv_results,
            'train_r2': xgb_train_r2,
            'train_rmse': xgb_train_rmse
        }

        self.xgb_model = xgb_model
        self.xgb_cv_results = xgb_cv_results

        # Train LightGBM
        logger.info("\nTraining LightGBM model...")
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': self.random_state,
            'verbose': -1
        }

        lgb_model = LGBMRegressor(**lgb_params)

        def lgb_fit(X_train, y_train):
            lgb_model.fit(X_train, y_train)

        def lgb_predict(X_test):
            return lgb_model.predict(X_test)

        lgb_cv_results = self._loocv_evaluate(lgb_fit, lgb_predict, "LightGBM")

        # 在全数据上训练
        lgb_model.fit(self.X, self.y)
        y_pred_train = lgb_model.predict(self.X)
        lgb_train_r2 = r2_score(self.y, y_pred_train)
        lgb_train_rmse = np.sqrt(mean_squared_error(self.y, y_pred_train))

        all_results['LightGBM'] = {
            'model': lgb_model,
            'cv_results': lgb_cv_results,
            'train_r2': lgb_train_r2,
            'train_rmse': lgb_train_rmse
        }

        self.lgb_model = lgb_model
        self.lgb_cv_results = lgb_cv_results

        # Compare models
        logger.info("\nDecision tree model comparison:")
        logger.info(f"  XGBoost  - LOOCV R²: {xgb_cv_results['r2']:.4f}, RMSE: {xgb_cv_results['rmse']:.4f}")
        logger.info(f"  LightGBM - LOOCV R²: {lgb_cv_results['r2']:.4f}, RMSE: {lgb_cv_results['rmse']:.4f}")

        # SHAP analysis (optional)
        if plot:
            logger.info("\nPerforming SHAP feature importance analysis...")
            self._shap_analysis(xgb_model, "XGBoost")
            self._shap_analysis(lgb_model, "LightGBM")

        # 保存结果
        if save_to_file:
            self._save_decision_tree_results(all_results)

        # 可视化
        if plot:
            self._plot_decision_tree_results(all_results)

        logger.info(f"\nDecision tree model training completed - Group {self.group}")

        return all_results

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
        """Save decision tree results"""
        dt_dir = os.path.join(self.group_dir, 'decision_tree')
        os.makedirs(dt_dir, exist_ok=True)

        # Save comparison results
        comparison_data = []
        for model_name, result in results.items():
            cv_res = result['cv_results']
            comparison_data.append({
                'Model': model_name,
                'LOOCV_R²': cv_res['r2'],
                'LOOCV_RMSE': cv_res['rmse'],
                'LOOCV_MAE': cv_res['mae'],
                'Train_R²': result['train_r2'],
                'Train_RMSE': result['train_rmse']
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = os.path.join(dt_dir, 'decision_tree_comparison.csv')
        comparison_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
        logger.info(f"Decision tree model comparison results saved: {comparison_path}")

        # Save feature importance
        for model_name, result in results.items():
            model = result['model']
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': [f'PC{i+1}' for i in range(len(model.feature_importances_))],
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)

                importance_path = os.path.join(dt_dir, f'{model_name}_feature_importance.csv')
                importance_df.to_csv(importance_path, index=False)

    def _shap_analysis(self, model, model_name: str):
        """SHAP feature importance analysis"""
        try:
            shap_analyzer = SHAPAnalyzer(model)
            X_df = pd.DataFrame(self.X, columns=[f'PC{i+1}' for i in range(self.X.shape[1])])

            shap_analyzer.create_explainer(X_df, explainer_type='tree')
            shap_analyzer.calculate_shap_values(X_df)

            # 保存SHAP图表
            shap_dir = os.path.join(self.group_dir, 'shap', model_name)
            os.makedirs(shap_dir, exist_ok=True)

            # Summary plot
            summary_path = os.path.join(shap_dir, 'shap_summary.png')
            shap_analyzer.summary_plot(save_path=summary_path)

            # Feature importance
            importance = shap_analyzer.get_feature_importance()
            importance_df = pd.DataFrame(list(importance.items()), 
                                        columns=['Feature', 'SHAP_Importance'])
            importance_path = os.path.join(shap_dir, 'shap_feature_importance.csv')
            importance_df.to_csv(importance_path, index=False)

            logger.info(f"SHAP analysis results for {model_name} saved to: {shap_dir}")
        except Exception as e:
            logger.warning(f"SHAP analysis failed ({model_name}): {str(e)}")

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

        for idx, (model_name, result) in enumerate(results.items()):
            cv_res = result['cv_results']
            ax = axes[idx]

            ax.scatter(cv_res['y_true'], cv_res['y_pred'], alpha=0.6)

            min_val = min(cv_res['y_true'].min(), cv_res['y_pred'].min())
            max_val = max(cv_res['y_true'].max(), cv_res['y_pred'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values (LOOCV)')
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
