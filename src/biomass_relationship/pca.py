"""
PCA (Principal Component Analysis) dimension reduction analyzer
"""
import logging
from typing import Optional, Tuple, List, Union
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class PCAnalyzer:
    """
    Principal Component Analysis (PCA) analyzer for dimension reduction

    Performs standardization and PCA transformation on input data
    """

    def __init__(self, variance_threshold: float, max_components: int, n_components: Optional[int] = None):
        """
        Initialize PCA analyzer

        Args:
            n_components: Number of components to keep. If None, keeps components 
                         explaining variance_threshold of total variance
            variance_threshold: Variance explained threshold (0-1) when n_components is None
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.max_components = max_components
        self.scaler = StandardScaler()
        self.pca = None
        self.is_fitted = False
        self.feature_names = None

        logger.info("Initialized PCAnalyzer with n_components=%s, variance_threshold=%.2f",
                   n_components, variance_threshold)

    def fit(self, data: Union[np.ndarray, pd.DataFrame], feature_names: Optional[List[str]] = None) -> None:
        """
        Fit PCA model to data

        Args:
            data: Input data with shape (n_samples, n_features)
            feature_names: Optional list of feature names
        """
        # Convert to numpy array if DataFrame
        if isinstance(data, pd.DataFrame):
            self.feature_names = data.columns.tolist()
            data_array = data.values
        else:
            data_array = data
            self.feature_names = feature_names

        if len(data_array.shape) != 2:
            raise ValueError(f"Expected 2D array, got shape {data_array.shape}")

        n_samples, n_features = data_array.shape
        logger.info("Fitting PCA on data with shape (%d samples, %d features)",
                   n_samples, n_features)

        # Standardize data
        data_scaled = self.scaler.fit_transform(data_array)

        # Fit PCA
        if self.n_components is None:
            # Fit with all components first to determine optimal number
            pca_temp = PCA()
            pca_temp.fit(data_scaled)

            # Find number of components needed for variance threshold
            cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)
            n_comp = np.argmax(cumsum_var >= self.variance_threshold) + 1
            if n_comp > self.max_components:
                n_comp = self.max_components
            self.n_components = n_comp

            logger.info("Selected %d components to explain %.1f%% variance",
                       n_comp, self.variance_threshold * 100)

        # Fit final PCA with selected components
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(data_scaled)

        self.is_fitted = True

        # Log variance explained
        total_var = np.sum(self.pca.explained_variance_ratio_) * 100
        logger.info("PCA fitted: %d components explain %.2f%% of variance",
                   self.n_components, total_var)

    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Transform data to principal components

        Args:
            data: Input data with shape (n_samples, n_features)

        Returns:
            Transformed data with shape (n_samples, n_components)
        """
        if not self.is_fitted:
            raise ValueError("PCA model not fitted. Call fit() first.")

        # Convert to numpy array if DataFrame
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = data

        # Standardize and transform
        data_scaled = self.scaler.transform(data_array)
        transformed = self.pca.transform(data_scaled)

        logger.info("Transformed data from %d to %d dimensions",
                   data_array.shape[1], transformed.shape[1])

        return transformed

    def inverse_transform(self, transformed_data: np.ndarray) -> np.ndarray:
        """
        Transform data back to original feature space

        Args:
            transformed_data: Transformed data with shape (n_samples, n_components)

        Returns:
            Data in original feature space with shape (n_samples, n_features)
        """
        if not self.is_fitted:
            raise ValueError("PCA model not fitted. Call fit() first.")

        # Inverse transform from PCA space
        data_scaled = self.pca.inverse_transform(transformed_data)

        # Inverse transform from standardization
        data_original = self.scaler.inverse_transform(data_scaled)

        return data_original

    def get_explained_variance(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get explained variance ratio for each component

        Returns:
            Tuple of (variance_ratio, cumulative_variance_ratio)
        """
        if not self.is_fitted:
            raise ValueError("PCA model not fitted. Call fit() first.")

        variance_ratio = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(variance_ratio)

        return variance_ratio, cumulative_variance

    def get_components(self) -> np.ndarray:
        """
        Get principal component vectors

        Returns:
            Component matrix with shape (n_components, n_features)
        """
        if not self.is_fitted:
            raise ValueError("PCA model not fitted. Call fit() first.")

        return self.pca.components_

    def get_feature_importance(self, component_idx: int = 0) -> pd.DataFrame:
        """
        Get feature importance (loadings) for a specific component

        Args:
            component_idx: Index of the component (0-based)

        Returns:
            DataFrame with features and their loadings, sorted by absolute value
        """
        if not self.is_fitted:
            raise ValueError("PCA model not fitted. Call fit() first.")

        if component_idx >= self.n_components:
            raise ValueError(f"Component index {component_idx} out of range [0, {self.n_components})")

        loadings = self.pca.components_[component_idx, :]

        # Create DataFrame
        if self.feature_names is not None:
            df = pd.DataFrame({
                'feature': self.feature_names,
                'loading': loadings,
                'abs_loading': np.abs(loadings)
            })
        else:
            df = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(loadings))],
                'loading': loadings,
                'abs_loading': np.abs(loadings)
            })

        # Sort by absolute loading
        df = df.sort_values('abs_loading', ascending=False).reset_index(drop=True)

        return df[['feature', 'loading', 'abs_loading']]

    def get_summary(self) -> pd.DataFrame:
        """
        Get summary of all components with variance explained

        Returns:
            DataFrame with component statistics
        """
        if not self.is_fitted:
            raise ValueError("PCA model not fitted. Call fit() first.")

        variance_ratio, cumulative_variance = self.get_explained_variance()

        summary = pd.DataFrame({
            'component': [f'PC{i+1}' for i in range(self.n_components)],
            'variance_explained': variance_ratio,
            'cumulative_variance': cumulative_variance,
            'variance_explained_pct': variance_ratio * 100,
            'cumulative_variance_pct': cumulative_variance * 100
        })

        return summary

    def reduce_dimensions(self, data: Union[np.ndarray, pd.DataFrame],
                         return_dataframe: bool = False) -> Union[np.ndarray, pd.DataFrame]:
        """
        Convenience method to reduce dimensions of new data

        Args:
            data: Input data to reduce
            return_dataframe: If True, return pandas DataFrame with component names

        Returns:
            Reduced dimension data
        """
        transformed = self.transform(data)

        if return_dataframe:
            component_names = [f'PC{i+1}' for i in range(self.n_components)]
            return pd.DataFrame(transformed, columns=component_names)

        return transformed

    def __repr__(self) -> str:
        """String representation"""
        if self.is_fitted:
            var_explained = np.sum(self.pca.explained_variance_ratio_) * 100
            return (f"PCAnalyzer(n_components={self.n_components}, "
                   f"fitted=True, variance_explained={var_explained:.2f}%)")
        else:
            return f"PCAnalyzer(n_components={self.n_components}, fitted=False)"

    def save(self, save_path: str):
        """
        Save PCA information including weights and summary to file

        Args:
            save_path: Path to save the results (should end with .xlsx or .csv)
                      For .xlsx: saves multiple sheets (summary + each component)
                      For .csv: saves summary to specified path and components to separate files
        """
        if not self.is_fitted:
            raise ValueError("PCA model not fitted. Call fit() first.")

        summary = self.get_summary()
        base_path = save_path.rsplit('.', 1)[0]

        summary_path = f"{base_path}_summary.csv"
        summary.to_csv(summary_path, index=False)
        logger.info("Saved PCA summary to %s", summary_path)

        # Save loadings for each component
        for i in range(self.n_components):
            component_name = f'PC{i+1}'
            loadings = self.get_feature_importance(component_idx=i)
            component_path = f"{base_path}_{component_name}_loadings.csv"
            loadings.to_csv(component_path, index=False)
            logger.info("Saved loadings for %s to %s", component_name, component_path)

        # Save all components matrix
        components_df = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i+1}' for i in range(self.n_components)],
            index=self.feature_names if self.feature_names else [f'feature_{i}' for i in range(self.pca.components_.shape[1])]
        )
        components_path = f"{base_path}_all_components.csv"
        components_df.to_csv(components_path)
        logger.info("Saved all components matrix to %s", components_path)

        logger.info("PCA information saved to %s (multiple files)", base_path)
