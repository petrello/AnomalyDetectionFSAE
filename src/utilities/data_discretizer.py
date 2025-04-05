import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Dict, Optional, List, Union, Tuple, Any


class DataDiscretizer:
    """
    Provides methods for discretizing numerical data and visualizing the results.

    This class offers static methods to discretize numerical columns within a Pandas DataFrame,
    using either quantile-based (qcut) or threshold-based (cut) discretization. It also includes
    visualization capabilities to compare the original and discretized data distributions,
    facilitating the understanding and evaluation of the discretization process.
    """

    @staticmethod
    def discretize_kmeans(
        df: pd.DataFrame,
        n_bins: Union[Optional[int], Optional[List[int]]] = None,
        min_bins: int = 3,
        max_bins: int = 7,
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """
        Discretizes numeric columns using K-Means binning, automatically selecting the best number of clusters
        using the Silhouette Score if n_bins is None.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the numeric columns to discretize.
        n_bins : int, list, optional
            The number of bins to use for K-Means. If None, the optimal number is determined automatically.
        min_bins : int, optional
            The minimum number of clusters to consider when auto-selecting the optimal number.
        max_bins : int, optional
            The maximum number of clusters to consider when auto-selecting the optimal number.

        Returns
        -------
        tuple of pd.DataFrame and dict
            A tuple containing the discretized DataFrame and a dictionary with discretization statistics.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> data = {'feature1': np.random.normal(0, 1, 100), 'feature2': np.random.exponential(1, 100)}
        >>> df = pd.DataFrame(data)
        >>> discretized_df, result = DataDiscretized.discretize_kmeans(df, n_bins=3)
        >>> print(discretized_df.head())
        >>> print(result['feature1'].keys())
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")
        if not all(pd.api.types.is_numeric_dtype(df[col]) for col in df.select_dtypes(include=['number']).columns):
            raise ValueError("DataFrame must contain numeric columns.")
        if n_bins is not None and (not isinstance(n_bins, int) and not isinstance(n_bins, list)):
            raise TypeError("n_bins must be an integer, a list of integers or None.")
        if not isinstance(min_bins, int) or not isinstance(max_bins, int) or min_bins < 2 or max_bins < min_bins:
            raise ValueError("min_bins and max_bins must be integers, with min_bins >= 2 and max_bins >= min_bins.")

        discretized_df = df.copy()
        result = {}

        # Normalize n_bins input
        numeric_columns = df.select_dtypes(include=['number']).columns
        if isinstance(n_bins, int):
            n_bins = [n_bins] * len(numeric_columns)
        elif n_bins is not None and len(n_bins) != len(numeric_columns):
            raise ValueError("Length of n_bins must match number of numeric columns")

        for idx, col in enumerate(df.select_dtypes(include=['number']).columns):
            print(f"Processing column: {col}")  # Debug print
            values = df[col].dropna().values.reshape(-1, 1)

            if values.shape[0] < max_bins:  # If too few unique values, skip K-Means
                print(f"Skipping {col} due to insufficient unique values.") # Debug print
                continue
            
            # Determine optimal number of bins
            if n_bins is None or n_bins[idx] is None:
                best_k = None
                best_score = -1.0
                best_kmeans: Optional[KMeans] = None

                for k in range(min_bins, max_bins + 1):
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(values)
                    score = silhouette_score(values, labels)
                    print(f"{col}: k={k}, silhouette score={score:.4f}") # Debug print

                    if score > best_score:
                        best_score = score
                        best_k = k
                        best_kmeans = kmeans
                print(f"{col}: Optimal k found: {best_k}") # Debug print
            else:
                # Use specified number of bins
                best_k = n_bins[idx]
                best_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(values)
                print(f"{col}: Using provided n_bins: {n_bins[idx]}") # Debug print

            kmeans = best_kmeans
            labels = kmeans.labels_
            cluster_centers = kmeans.cluster_centers_.flatten()
            bin_edges = np.percentile(values, np.linspace(0, 100, best_k + 1))

            # Create mapping of cluster labels to original ranges
            mapping = {i: (bin_edges[i], bin_edges[i + 1]) for i in range(best_k)}

            # Store results
            result[col] = {
                'method': 'K-Means Binning',
                'bin_edges': bin_edges.tolist(),
                'cluster_centers': cluster_centers.tolist(),
                'bin_count': best_k,
                'value_counts': pd.Series(labels).value_counts().sort_index().to_dict(),
                'mapping': mapping
            }

            discretized_df[col] = labels  # Replace with cluster labels
            print(f"{col} discretized into {best_k} bins.") # Debug print

        return discretized_df, result

    @staticmethod
    def binarize_columns(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        labels: Optional[Dict[str, List[Union[int, str]]]] = None, 
    ) -> Tuple[Dict[str, Dict[str, Union[float, List[Union[int, str]], pd.Series, float]]], pd.DataFrame]:
        """
        Binarizes numerical columns in a DataFrame based on specified thresholds.

        This static method converts numerical columns to binary values based on given thresholds.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the numerical columns to binarize.
        columns : list of str, optional
            List of column names to binarize. If None, all numeric columns are binarized.
        thresholds : dict of str to float, optional
            Dictionary mapping column names to threshold values. Default threshold is 0.5.
        labels : dict of str to list of int or str, optional
            Dictionary mapping column names to label pairs [false_label, true_label].
        
        Returns
        -------
        tuple of dict and pd.DataFrame
            Returns a tuple of a dictionary containing binarization statistics 
            for each column and the binarized DataFrame.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> # Create a sample DataFrame
        >>> data = {'feature1': np.random.normal(0, 1, 100), 'feature2': np.random.exponential(1, 100)}
        >>> df = pd.DataFrame(data)
        >>> # Binarize feature1 with a threshold of 0.5
        >>> result = DataDiscretized.binarize(df, columns=['feature1'], thresholds={'feature1': 0.5})
        >>> print(result)
        """
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if columns is None:
            columns = numeric_cols

        if thresholds is None:
            thresholds = {col: 0.5 for col in columns}

        if labels is None:
            labels = {col: [False, True] for col in columns}

        result = {}
        binarized_df = df.copy()

        for col in columns:
            threshold = thresholds.get(col, 0.5)
            col_labels = labels.get(col, [False, True])

            # Create binary column
            binary_col = col
            binarized_df[binary_col] = (df[col] > threshold).map({False: col_labels[0], True: col_labels[1]})

            # Count values
            value_counts = binarized_df[binary_col].value_counts()
            result[col] = {
                'threshold': threshold,
                'labels': col_labels,
                'value_counts': value_counts,
                'percentage_true': value_counts.get(col_labels[1], 0) / len(binarized_df) * 100
            }

        return result, binarized_df

    @staticmethod
    def discretize_columns(
        df: pd.DataFrame,
        columns_config: Optional[Dict[str, Dict[str, Union[int, List[float], List[str]]]]] = None,
        method: str = 'qcut'
    ) -> Tuple[Dict[str, Dict[str, Union[str, np.ndarray, int, pd.Series]]], pd.DataFrame]:
        """
        Discretizes specified numerical columns in a DataFrame and visualizes the results.

        This method applies either quantile-based (qcut) or threshold-based (cut) discretization to
        the specified numerical columns in the input DataFrame. It generates visualizations to compare
        the original data distribution with the discretized distribution, enhancing the understanding
        of the discretization process.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the numerical columns to discretize.
        columns_config : dict of str to dict, optional
            A dictionary defining the discretization configuration for each column.
            Keys are column names, and values are dictionaries with 'bins' and 'labels' keys.
            - For 'qcut' method: 'bins' specifies the number of quantiles, 'labels' provides custom bin labels.
            - For 'cut' method: 'bins' specifies the bin edges, 'labels' provides custom bin labels.
            If None, all numerical columns are discretized using 4 quantile bins.
        method : str, optional
            The discretization method to use: 'qcut' for quantile-based or 'cut' for threshold-based,
            default is 'qcut'.

        Returns
        -------
        tuple of dict and pd.DataFrame
            Returns a tuple containing the discretization information dictionary and the
            discretized DataFrame.

        Raises
        ------
        ValueError
            If an unknown discretization method is specified.
            If 'cut' method is used without specifying 'bins' in the columns_config.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> # Create a sample DataFrame
        >>> data = {'feature1': np.random.normal(0, 1, 100), 'feature2': np.random.exponential(1, 100), 'feature3': np.random.randint(0, 100, 100)}
        >>> df = pd.DataFrame(data)
        >>> # Discretize columns using 'cut' method
        >>> result = DataDiscretized.discretize_columns(df, columns_config={'feature3': {'bins': [0, 25, 50, 75, 100], 'labels': ['Q1', 'Q2', 'Q3', 'Q4']}}, method='cut')
        >>> print(result)
        """
        numeric_cols = df.select_dtypes(include=['number']).columns
        if columns_config is None:
            # Default config: 4 quantile bins for all numeric columns
            columns_config = {col: {'bins': 4} for col in numeric_cols}

        result = {}
        discretized_df = df.copy()

        columns = list(columns_config.keys())
        n_cols = len(columns)

        # Handle case with only one column
        if n_cols == 1:
            axes = axes.reshape(1, 2)

        for i, col in enumerate(columns):
            config = columns_config[col]

            try:
                if method == 'qcut':
                    bins = config.get('bins', 4)
                    labels = config.get('labels', None)

                    discretized_series, bin_edges = pd.qcut(
                        df[col], bins, labels=labels, retbins=True, duplicates='drop'
                    )
                    method_name = "Quantile-based"

                elif method == 'cut':
                    bins = config.get('bins')
                    if bins is None:
                        raise ValueError(f"For 'cut' method, 'bins' must be specified for column {col}")

                    labels = config.get('labels', None)

                    discretized_series, bin_edges = pd.cut(
                        df[col], bins, labels=labels, retbins=True, include_lowest=True
                    )
                    method_name = "Threshold-based"

                else:
                    raise ValueError(f"Unknown method: {method}. Use 'qcut' or 'cut'")

                discretized_df[col] = discretized_series

                result[col] = {
                    'method': method_name,
                    'bin_edges': bin_edges,
                    'bin_count': len(bin_edges) - 1,
                    'value_counts': discretized_series.value_counts().sort_index()
                }

            except Exception as e:
                print(f"Error discretizing {col}: {e}")

        return result, discretized_df