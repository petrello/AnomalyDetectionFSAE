import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from typing import Tuple, Optional, List, Dict, Union, Any

class DataVisualizer:
    """
    A class providing visualization methods for generic data visualization.
    
    Contains static methods for creating various plots and visual representations
    of data to aid in analysis and interpretation.
    """

    @staticmethod
    def plot_error_statistics(error_stats: Dict[str, Dict[int, int]]) -> None:
        """
        Plots a bar chart of error statistics.

        Parameters
        ----------
        error_stats : Dict[str, Dict[int, int]]
            Dictionary where keys are error types, and values are dictionaries with counts of 0s and 1s.

        Raises
        ------
        TypeError
            If error_stats is not a dictionary.
        ValueError
            If any value in error_stats is not a dictionary or if the inner dictionaries do not contain integers.
        """
        if not isinstance(error_stats, dict):
            raise TypeError("error_stats must be a dictionary.")

        if not error_stats:
            print("No error statistics to plot.")
            return

        # Prepare data for plotting
        error_types = []
        values_0 = []
        values_1 = []

        for error_key, counts in error_stats.items():
            if not isinstance(counts, dict):
                raise ValueError(f"Counts for {error_key} must be a dictionary.")
            if not all(isinstance(k, int) and isinstance(v, int) for k, v in counts.items()):
                raise ValueError(f"Counts for {error_key} must contain integer keys and values.")
            error_types.append(error_key)
            values_0.append(counts.get(0, 0))  # Count of '0' occurrences
            values_1.append(counts.get(1, 0))  # Count of '1' occurrences

        # Plotting
        plt.figure(figsize=(12, 6))
        bar_width = 0.4
        x_positions = range(len(error_types))

        plt.bar(x_positions, values_0, width=bar_width, label="No Error (0)", color="green")
        plt.bar([x + bar_width for x in x_positions], values_1, width=bar_width, label="Error (1)", color="red")

        # Labels & Title
        plt.xlabel("Error Types")
        plt.ylabel("Count")
        plt.title("Error Statistics by Key")
        plt.xticks([x + bar_width / 2 for x in x_positions], error_types, rotation=45, ha="right")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Show plot
        plt.tight_layout()
        plt.show()

    @staticmethod
    def display_dataset_info(
        full_dataset: pd.DataFrame, 
        normal_dataset: pd.DataFrame, 
        anomalous_dataset: pd.DataFrame, 
        encoding_mappings: Dict[str, Dict[int, Any]]
    ) -> None:
        """
        Displays information about the transformed datasets, including shapes, memory usage, and encoding mappings.

        This static method prints detailed information about the full, normal, and anomalous datasets,
        such as their shapes, memory usage, sample column names, time slices included, NaN value counts,
        and a sample of encoding mappings.

        Parameters
        ----------
        full_dataset : pd.DataFrame
            The full Dynamic Bayesian Network (DBN) dataset.
        normal_dataset : pd.DataFrame
            The normal DBN dataset (filtered).
        anomalous_dataset: pd.DataFrame
            The anomalous DBN dataset (filtered).
        encoding_mappings : dict of str to dict of int to Any
            Dictionary mapping encoded values to original values.

        Examples
        --------
        >>> import pandas as pd
        >>> # Create sample DataFrames and encoding mappings
        >>> full_data = pd.DataFrame({'A_0': [1, 2, 3], 'B_0': [4, 5, 6], 'A_1': [7, 8, 9], 'B_1': [10, 11, 12]})
        >>> normal_data = pd.DataFrame({'A_0': [1, 2], 'B_0': [4, 5], 'A_1': [7, 8], 'B_1': [10, 11]})
        >>> anomalous_data = pd.DataFrame({'A_0': [3], 'B_0': [6], 'A_1': [9], 'B_1': [12]})
        >>> mappings = {'A': {0: 'a', 1: 'b'}, 'B': {0: 'c', 1: 'd'}}
        >>> # Display dataset information
        >>> DataVisualizer.display_dataset_info(full_data, normal_data, anomalous_data, mappings)
        """
        print("Full Dataset Information:")
        print(f"Shape: {full_dataset.shape}")
        print(f"Memory usage: {full_dataset.memory_usage().sum() / 1024 / 1024:.2f} MB")
        print("\nSample column names:")
        for col in list(full_dataset.columns[:5]):
            print(f"  {col}")

        print("\nNormal Dataset Information:")
        print(f"Shape: {normal_dataset.shape}")
        print(f"Memory usage: {normal_dataset.memory_usage().sum() / 1024 / 1024:.2f} MB")
        print("\nSample column names:")
        for col in list(normal_dataset.columns[:5]):
            print(f"  {col}")

        print("\nAnomalous Dataset Information:")
        print(f"Shape: {anomalous_dataset.shape}")
        print(f"Memory usage: {anomalous_dataset.memory_usage().sum() / 1024 / 1024:.2f} MB")
        print("\nSample column names:")
        for col in list(anomalous_dataset.columns[:5]):
            print(f"  {col}")

        # Count of each time slice in full dataset
        time_slices = set([t for _, t in full_dataset.columns])
        print(f"\nTime slices included: {sorted(time_slices)}")

        # Check for NaN values
        print(f"\nNaN values in full dataset: {full_dataset.isna().sum().sum()}")
        print(f"NaN values in normal dataset: {normal_dataset.isna().sum().sum()}")
        print(f"NaN values in anomalous dataset: {anomalous_dataset.isna().sum().sum()}")

        # Sample of encoding mappings
        print("\nSample of encoding mappings:")
        sample_vars = list(encoding_mappings.keys())[:3]
        for var in sample_vars:
            print(f"  {var}: {encoding_mappings[var]}")

    @staticmethod
    def plot_discrete_distributions(
        df: pd.DataFrame,
        fault_col: str = "InverterFault"
    ) -> Tuple[plt.Figure, Dict[str, Dict[str, Union[Dict, Dict[int, Dict]]]]]:
        """
        Visualize discrete numeric columns with multiple distribution plots and return summary statistics.

        This static method generates a set of visualizations for each discrete numeric column in the DataFrame,
        including count plots for overall distribution and distributions categorized by a fault column, as well as
        box plots to identify outliers. It also calculates and returns summary statistics for each column.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing discrete numeric columns.
        fault_col : str, optional
            Name of the fault column for categorization, default is "InverterFault".

        Returns
        -------
        Tuple[plt.Figure, Dict[str, Dict[str, Union[Dict, Dict[int, Dict]]]]]
            A tuple containing:
            - The matplotlib Figure object with the generated plots.
            - A dictionary containing summary statistics for each column, including value counts and outlier information.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> # Create a sample DataFrame
        >>> data = {'feature1': np.random.randint(0, 5, 100), 'feature2': np.random.randint(0, 3, 100), 'InverterFault': np.random.randint(0, 2, 100)}
        >>> df = pd.DataFrame(data)
        >>> # Plot discrete distributions
        >>> fig, mapping = DataVisualizer.plot_discrete_distributions(df, fault_col="InverterFault")
        """
        # Identify numeric columns excluding fault column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != fault_col]

        # Create subplots
        n_features = len(numeric_cols)
        fig, axes = plt.subplots(n_features, 4, figsize=(20, 4 * n_features))

        # Ensure axes is 2D array
        if n_features == 1:
            axes = axes.reshape(1, -1)

        for i, col in enumerate(numeric_cols):
            # Frequency counts
            counts = df[col].value_counts().sort_index()

            # 1. Overall Count Plot
            sns.barplot(x=counts.index, y=counts.values, ax=axes[i, 0], palette="viridis", hue=counts.index, legend=False)
            axes[i, 0].set_title(f"Count Plot for {col}")
            axes[i, 0].set_xlabel(col)
            axes[i, 0].set_ylabel("Frequency")

            # 2. Box Plot with Fault Categorization
            sns.boxplot(x=fault_col, y=col, data=df, ax=axes[i, 1], palette='Set2', hue=fault_col, legend=False)
            axes[i, 1].set_title(f"Box Plot: {col} by {fault_col}")
            axes[i, 1].set_xlabel(fault_col)
            axes[i, 1].set_ylabel(col)

            # 3. Count Plot for Fault = 0
            counts_fault_0 = df[df[fault_col] == 0][col].value_counts().sort_index()
            sns.barplot(x=counts_fault_0.index, y=counts_fault_0.values, ax=axes[i, 2], palette='Blues_d', hue=counts_fault_0.index, legend=False)
            axes[i, 2].set_title(f"Count Plot for {col} (Fault = 0)")
            axes[i, 2].set_xlabel(col)
            axes[i, 2].set_ylabel("Frequency")

            # 4. Count Plot for Fault = 1
            counts_fault_1 = df[df[fault_col] == 1][col].value_counts().sort_index()
            sns.barplot(x=counts_fault_1.index, y=counts_fault_1.values, ax=axes[i, 3], palette='Oranges_d', hue=counts_fault_1.index, legend=False)
            axes[i, 3].set_title(f"Count Plot for {col} (Fault = 1)")
            axes[i, 3].set_xlabel(col)
            axes[i, 3].set_ylabel("Frequency")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def correlation_heatmap(
        df: pd.DataFrame, 
        title: str = 'Correlation Between Variables',
        figsize: Tuple[float, float] = (12, 10)
    ) -> List[Tuple[str, str, float]]:
        """
        Display correlation heatmap between numeric variables and return highly correlated pairs.

        This static method generates a heatmap visualization of the correlation matrix for numeric
        columns in the input DataFrame. It highlights the correlation coefficients between variables,
        aiding in the identification of related patterns. Additionally, it returns a list of variable
        pairs with correlation coefficients exceeding a threshold (absolute value > 0.7).

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the data to visualize.
        figsize : tuple of float, optional
            Custom figure size (width, height) in inches, default is (12, 10).

        Returns
        -------
        list of tuple of str, str, float
            A sorted list of tuples, each containing two column names and their correlation coefficient,
            for pairs with absolute correlation > 0.7.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> # Create a sample DataFrame
        >>> data = {'feature1': np.random.normal(0, 1, 100), 'feature2': np.random.normal(0, 1, 100) * 2, 'feature3': np.random.normal(0, 1, 100) + 5}
        >>> df = pd.DataFrame(data)
        >>> # Plot correlation heatmap and get highly correlated pairs
        >>> correlated_pairs = DataVisualizer.correlation_heatmap(df)
        >>> print(correlated_pairs)
        """
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        plt.figure(figsize=figsize)
        corr_matrix = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=True,
                    fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
        plt.title(title)
        plt.tight_layout()
        plt.show()

        # Return highly correlated pairs
        corr_pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    corr_pairs.append((numeric_cols[i], numeric_cols[j], corr_matrix.iloc[i, j]))

        return sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)

    @staticmethod
    def plot_distributions_with_kde_and_boxplot(
        df: pd.DataFrame, 
        fault_col: str = None,
        title: str = 'Feature Distributions and Fault Correlation'
    ) -> None:
        """
        Plots histograms and boxplots for numeric columns in a DataFrame to aid in discretization decisions.

        This static method generates a histogram with Kernel Density Estimation (KDE) and a boxplot
        for each numeric column in the input DataFrame, in a single figure. These visualizations are helpful in
        understanding the distribution of data, identifying potential skewness, modality, and outliers,
        which are crucial factors to consider when discretizing features.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the numeric columns to visualize.
        fault_col : str, optional
            Column name to use for grouping/coloring the data (e.g., a categorical feature).
            If provided, boxplots will be grouped by this column similar to the provided example.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> # Create a sample DataFrame
        >>> data = {'feature1': np.random.normal(0, 1, 100), 
        ...         'feature2': np.random.exponential(1, 100), 
        ...         'category': ['A'] * 50 + ['B'] * 50}
        >>> df = pd.DataFrame(data)
        >>> # Plot distributions for numeric columns
        >>> DataVisualizer.plot_distributions_with_kde_and_boxplot(df, fault_col='category')
        """
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.difference(['InverterFault'])
        
        # Create a single figure with multiple subplots
        n_cols = len(numeric_cols)
        if n_cols == 0:
            print("No numeric columns found in the DataFrame.")
            return
        
        # Create a figure with n_cols rows and 2 columns (hist and box for each feature)
        fig, axes = plt.subplots(n_cols, 2, figsize=(12, 5 * n_cols))
        
        # If there's only one numeric column, axes won't be 2D, so reshape it
        if n_cols == 1:
            axes = axes.reshape(1, -1)
        
        # Loop through each numeric column
        for i, col in enumerate(numeric_cols):
            col_data = df[col].dropna()
            
            if col_data.empty:
                continue
            
            # Local style settings for this plot only (won't affect global settings)
            with plt.style.context('default'):
                # Plot 1: Histogram with KDE
                if fault_col and fault_col in df.columns:
                    # Filter out rows where hue column is NaN
                    valid_data = df.dropna(subset=[fault_col])
                    # For histplot with hue, use multiple=True for overlapping distributions
                    hist_plot = sns.histplot(data=valid_data, x=col, hue=fault_col, kde=True, 
                                alpha=0.6, ax=axes[i, 0], multiple="layer")
                else:
                    hist_plot = sns.histplot(col_data, kde=True, ax=axes[i, 0], color='#5975A4', alpha=0.7)
                
                # Enhance histogram aesthetics without changing global settings
                hist_plot.set_title(f'Distribution of {col}', fontweight='bold', size=12)
                hist_plot.set_xlabel(col, fontweight='bold', size=11)
                hist_plot.set_ylabel('Frequency', fontweight='bold', size=11)
                hist_plot.tick_params(labelsize=10)
                hist_plot.grid(True, linestyle='--', alpha=0.7)
                
                # Plot 2: Boxplot with improved aesthetics matching the example
                if fault_col and fault_col in df.columns:
                    # Filter out rows where hue column is NaN
                    valid_data = df.dropna(subset=[fault_col])
                    # Use the hue column for grouping
                    box_plot = sns.boxplot(data=valid_data, x=fault_col, y=col, ax=axes[i, 1], 
                            palette=['#8ABB8C', '#D69C4E'], hue=fault_col, legend=False, width=0.5, linewidth=1)
                else:
                    box_plot = sns.boxplot(x=col_data, ax=axes[i, 1], color='#8ABB8C', width=0.5, linewidth=1)
                    
                # Enhance boxplot aesthetics without changing global settings
                box_title = f'{col} Distribution by {fault_col}' if fault_col else f'Boxplot of {col}'
                box_plot.set_title(box_title, fontweight='bold', size=12)
                
                if fault_col:
                    box_plot.set_xlabel(fault_col, fontweight='bold', size=11)
                    box_plot.set_ylabel(col, fontweight='bold', size=11)
                else:
                    box_plot.set_xlabel(col, fontweight='bold', size=11)
                
                num_ticks = len(box_plot.get_xticklabels())

                box_plot.tick_params(labelsize=10)
                box_plot.set_xticks(range(num_ticks))
                if num_ticks == 2:
                    box_plot.set_xticklabels(['No Fault', 'Fault'])
                elif num_ticks == 1:
                    box_plot.set_xticklabels(['No Fault'] if '0' in str(box_plot.get_xticklabels()[0]) else ['Fault'])
                box_plot.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add overall title and adjust spacing
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(top=0.96)
        plt.show()
    
    @staticmethod
    def plot_distributions_overview(
        df: pd.DataFrame, 
        columns: Optional[List[str]] = None, 
        bins: int = 30,
        quantiles: List[float] = [0, 0.25, 0.5, 0.75, 1.0],
        figsize: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Dict[str, Union[int, List[float]]]]:
        """
        Display distribution plots for selected columns with improved visibility and return quantile information.

        This static method generates histograms with Kernel Density Estimation (KDE) for specified or
        all numeric columns in the DataFrame. It also calculates and visualizes mean, median, and
        quantile lines, providing a comprehensive overview of the data distribution.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the data to visualize.
        columns : list of str, optional
            List of column names to visualize. If None, all numeric columns are used.
        bins : int, optional
            Number of bins for histograms, default is 30.
        quantiles : list of float, optional
            List of quantiles to use for bin edge suggestions, default is [0, 0.25, 0.5, 0.75, 1.0].
        figsize : tuple of float, optional
            Custom figure size (width, height) in inches, defaults to auto-calculated based on columns.

        Returns
        -------
        dict of str to dict of str to int or list of float
            A dictionary containing quantile information for each visualized column.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> # Create a sample DataFrame
        >>> data = {'feature1': np.random.normal(0, 1, 100), 'feature2': np.random.exponential(1, 100), 'category': ['A'] * 100}
        >>> df = pd.DataFrame(data)
        >>> # Plot distributions for numeric columns and get quantile info
        >>> quantile_info = DataVisualizer.distribution_overview(df)
        >>> print(quantile_info)
        """
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if columns is None:
            columns = numeric_cols

        n_cols = 1  # Single column layout
        n_rows = len(columns)

        # Auto-calculate figure size if not provided
        if figsize is None:
            figsize = (16, n_rows * 6)  # Wider and taller per plot for better visibility

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = [axes]

        # Set a clean style
        plt.style.use('seaborn-v0_8-whitegrid')

        # Define colors for visual elements
        quantile_color = '#e74c3c'  # Red for quantile lines

        result = {}

        for i, col in enumerate(columns):
            # Get data without NaN values
            data = df[col].dropna()

            # Create histogram with improved styling
            sns.histplot(
                data,
                kde=True,
                bins=bins,
                ax=axes[i],
                alpha=0.7
            )

            # Calculate statistics
            mean_val = data.mean()
            median_val = data.median()
            percentiles = [q * 100 for q in quantiles[1:-1]]
            percentile_values = np.percentile(data, percentiles)

            # Get plot limits for better text positioning
            y_min, y_max = axes[i].get_ylim()
            x_min, x_max = axes[i].get_xlim()

            # Draw mean and median lines
            axes[i].axvline(mean_val, color='green', linestyle='-', alpha=0.8, linewidth=2.5)
            axes[i].axvline(median_val, color='purple', linestyle='--', alpha=0.8, linewidth=2.5)

            # Add mean and median labels with background
            def add_text_with_background(ax, x, y, text, color, alpha=0.9):
                bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=alpha)
                ax.text(x, y, text, color=color, fontsize=12, fontweight='bold',
                        ha='center', va='center', bbox=bbox_props)

            # Add mean label
            add_text_with_background(axes[i], mean_val, y_max * 0.95, f'Mean: {mean_val:.2f}', 'green')

            # Add median label
            add_text_with_background(axes[i], median_val, y_max * 0.85, f'Median: {median_val:.2f}', 'purple')

            # Add quantile lines with improved visibility
            for j, p in enumerate(percentile_values):
                axes[i].axvline(p, color=quantile_color, linestyle='--', alpha=0.8, linewidth=2)

                # Place quantile labels in better positions to avoid overlap
                vertical_position = y_max * (0.75 - j * 0.1)  # Staggered positions

                # Add quantile labels with background
                add_text_with_background(axes[i], p, vertical_position, f'{int(percentiles[j])}%: {p:.2f}', quantile_color)

            # Improve title and labels with better formatting
            axes[i].set_title(f'Distribution of {col}', fontsize=20, fontweight='bold', pad=20)
            axes[i].set_xlabel(col, fontsize=16, fontweight='bold', labelpad=15)
            axes[i].set_ylabel('Frequency', fontsize=16, fontweight='bold', labelpad=15)

            # Improve tick parameters
            axes[i].tick_params(axis='both', labelsize=14, colors='#555555')

            # Quantile information for each column
            edges = [np.percentile(data, q * 100) for q in quantiles]
            result[col] = {
                'num_bins': len(quantiles) - 1,
                'quantile_edges': edges,
                'quantiles_used': quantiles
            }

        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4)
        plt.show()

        return result

    @staticmethod
    def plot_prediction_heatmap(
        normal_predictions: ArrayLike,
        anomalous_predictions: ArrayLike,
        figsize: Tuple[float, float] = (12, 10),
        cmap: str = "Blues",
        save_path: Optional[str] = None,
        show_plot: bool = True,
        include_percentages: bool = True,
        title: str = "Prediction Heatmap"
    ) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
        """
        Creates an enhanced heatmap visualization of prediction results with 
        additional statistics and metrics.

        Parameters
        ----------
        normal_predictions : array-like
            Predictions on normal validation data (0 or 1, where 0=Normal, 1=Anomalous)
        anomalous_predictions : array-like
            Predictions on anomalous validation data (0 or 1, where 0=Normal, 1=Anomalous)
        figsize : tuple of float, optional
            Figure size (width, height) in inches, default=(10, 8)
        cmap : str, optional
            Colormap for the heatmap, default="Blues"
        save_path : str, optional
            Path to save the figure; if None, figure is not saved, default=None
        show_plot : bool, optional
            Whether to display the plot with plt.show(), default=True
        include_percentages : bool, optional
            Whether to display percentages alongside counts in the heatmap, default=True
        title : str, optional
            Title for the overall figure, default="Prediction Heatmap"

        Returns
        -------
        Tuple[plt.Figure, Dict[str, plt.Axes]]
            The matplotlib figure and a dictionary of axes objects for further customization
        
        Examples
        --------
        >>> import numpy as np
        >>> # Generate sample predictions
        >>> normal_preds = np.random.randint(0, 2, 100)  # 100 predictions for normal data
        >>> anomalous_preds = np.random.randint(0, 2, 50)  # 50 predictions for anomalous data
        >>> # Create and show the plot
        >>> fig, axes = DataVisualizer.plot_prediction_heatmap(normal_preds, anomalous_preds)
        """
        # Convert inputs to numpy arrays
        normal_array = np.asarray(normal_predictions)
        anomalous_array = np.asarray(anomalous_predictions)
        
        # Count occurrences for confusion matrix
        counts = np.array([
            [np.sum(normal_array == 0), np.sum(normal_array == 1)],  # Normal Data
            [np.sum(anomalous_array == 0), np.sum(anomalous_array == 1)]  # Anomalous Data
        ])
        
        # Calculate percentages
        total_normals = len(normal_array)
        total_anomalous = len(anomalous_array)
        
        percentages = np.array([
            [np.sum(normal_array == 0) / total_normals * 100, np.sum(normal_array == 1) / total_normals * 100],
            [np.sum(anomalous_array == 0) / total_anomalous * 100, np.sum(anomalous_array == 1) / total_anomalous * 100]
        ])
        
        # Calculate metrics
        true_negative = counts[0, 0]
        false_positive = counts[0, 1]
        false_negative = counts[1, 0]
        true_positive = counts[1, 1]
        
        total_samples = true_negative + false_positive + false_negative + true_positive
        accuracy = (true_positive + true_negative) / total_samples
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Create a figure with grid layout
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3, height_ratios=[3, 2], width_ratios=[2, 2, 2])
        
        # Create axes dictionary to return
        axes_dict = {}
        
        # Main heatmap plot
        ax_heatmap = fig.add_subplot(gs[0, :2])
        axes_dict['heatmap'] = ax_heatmap
        
        # Format annotation text with counts and percentages if requested
        if include_percentages:
            annot_text = np.array([
                [f"{counts[0, 0]}\n({percentages[0, 0]:.1f}%)", f"{counts[0, 1]}\n({percentages[0, 1]:.1f}%)"],
                [f"{counts[1, 0]}\n({percentages[1, 0]:.1f}%)", f"{counts[1, 1]}\n({percentages[1, 1]:.1f}%)"]
            ])
            sns.heatmap(counts, annot=annot_text, fmt='', cmap=cmap, xticklabels=["Pred Normal", "Pred Anomalous"], 
                        yticklabels=["True Normal", "True Anomalous"], ax=ax_heatmap)
        else:
            sns.heatmap(counts, annot=True, fmt='d', cmap=cmap, xticklabels=["Pred Normal", "Pred Anomalous"], 
                        yticklabels=["True Normal", "True Anomalous"], ax=ax_heatmap)
        
        ax_heatmap.set_xlabel("Predicted Label")
        ax_heatmap.set_ylabel("True Label")
        ax_heatmap.set_title("Confusion Matrix")
        
        # Metrics panel
        ax_metrics = fig.add_subplot(gs[0, 2])
        axes_dict['metrics'] = ax_metrics
        ax_metrics.axis('off')
        
        metrics_text = (
            f"Classification Metrics:\n\n"
            f"Accuracy: {accuracy:.3f}\n"
            f"Precision: {precision:.3f}\n"
            f"Recall: {recall:.3f}\n"
            f"F1 Score: {f1_score:.3f}\n\n"
            f"True Positive: {true_positive}\n"
            f"True Negative: {true_negative}\n"
            f"False Positive: {false_positive}\n"
            f"False Negative: {false_negative}\n\n"
            f"Total Samples: {total_samples}\n"
            f"Normal Samples: {total_normals}\n"
            f"Anomalous Samples: {total_anomalous}"
        )
        
        ax_metrics.text(
            0.5, 0.5, 
            metrics_text,
            transform=ax_metrics.transAxes, 
            verticalalignment='center', 
            horizontalalignment='center',
            bbox=dict(
                boxstyle='round',
                facecolor='white',
                alpha=0.8,
                edgecolor='gray'
            ),
            fontsize=10
        )
        
        # Normal class bar chart
        ax_normal_bar = fig.add_subplot(gs[1, 0])
        axes_dict['normal_bar'] = ax_normal_bar
        
        normal_bar_data = [counts[0, 0], counts[0, 1]]
        ax_normal_bar.bar(['Predicted Normal', 'Predicted Anomalous'], normal_bar_data, color=['lightblue', 'lightcoral'])
        ax_normal_bar.set_title('True Normal Class')
        ax_normal_bar.set_ylabel('Count')
        
        # Add percentage labels
        for i, v in enumerate(normal_bar_data):
            ax_normal_bar.text(i, v + 0.1, f"{percentages[0, i]:.1f}%", ha='center')
        
        # Anomalous class bar chart
        ax_anomalous_bar = fig.add_subplot(gs[1, 1])
        axes_dict['anomalous_bar'] = ax_anomalous_bar
        
        anomalous_bar_data = [counts[1, 0], counts[1, 1]]
        ax_anomalous_bar.bar(['Predicted Normal', 'Predicted Anomalous'], anomalous_bar_data, color=['lightblue', 'lightcoral'])
        ax_anomalous_bar.set_title('True Anomalous Class')
        
        # Add percentage labels
        for i, v in enumerate(anomalous_bar_data):
            ax_anomalous_bar.text(i, v + 0.1, f"{percentages[1, i]:.1f}%", ha='center')
        
        # Add rates visualization
        ax_rates = fig.add_subplot(gs[1, 2])
        axes_dict['rates'] = ax_rates
        
        rates = [
            percentages[0, 0],  # True negative rate
            percentages[1, 1],  # True positive rate
            percentages[0, 1],  # False positive rate
            percentages[1, 0]   # False negative rate
        ]
        
        rate_labels = ['TNR', 'TPR', 'FPR', 'FNR']
        rate_colors = ['forestgreen', 'royalblue', 'crimson', 'darkorange']
        
        ax_rates.bar(rate_labels, rates, color=rate_colors)
        ax_rates.set_title('Classification Rates')
        ax_rates.set_ylabel('Percentage')
        ax_rates.set_ylim(0, 100)
        
        # Add percentage labels
        for i, v in enumerate(rates):
            ax_rates.text(i, v + 1, f"{v:.1f}%", ha='center')
        
        # Add figure title
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show the plot if requested
        if show_plot:
            plt.show()
        
        return fig, axes_dict

    @staticmethod
    def plot_prediction_bar_chart(
        normal_predictions: ArrayLike,
        anomalous_predictions: ArrayLike,
        figsize: Tuple[float, float] = (10, 10),
        save_path: Optional[str] = None,
        show_plot: bool = True,
        colors: Tuple[str, str] = ('royalblue', 'crimson'),
        title: str = "Prediction Distribution Analysis"
    ) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
        """
        Creates an enhanced bar chart visualization of prediction distributions with 
        multiple perspectives and statistical breakdown.

        Parameters
        ----------
        normal_predictions : array-like
            Predictions on normal validation data (0 or 1, where 0=Normal, 1=Anomalous)
        anomalous_predictions : array-like
            Predictions on anomalous validation data (0 or 1, where 0=Normal, 1=Anomalous)
        figsize : tuple of float, optional
            Figure size (width, height) in inches, default=(10, 8)
        save_path : str, optional
            Path to save the figure; if None, figure is not saved, default=None
        show_plot : bool, optional
            Whether to display the plot with plt.show(), default=True
        colors : tuple of str, optional
            Colors for normal and anomalous visualization, default=('royalblue', 'crimson')
        title : str, optional
            Title for the overall figure, default="Prediction Distribution Analysis"

        Returns
        -------
        Tuple[plt.Figure, Dict[str, plt.Axes]]
            The matplotlib figure and a dictionary of axes objects for further customization
        
        Examples
        --------
        >>> import numpy as np
        >>> # Generate sample predictions
        >>> normal_preds = np.random.randint(0, 2, 100)  # 100 predictions for normal data
        >>> anomalous_preds = np.random.randint(0, 2, 50)  # 50 predictions for anomalous data
        >>> # Create and show the plot
        >>> fig, axes = DataVisualizer.plot_prediction_bar_chart(normal_preds, anomalous_preds)
        """
        # Convert inputs to numpy arrays
        normal_array = np.asarray(normal_predictions)
        anomalous_array = np.asarray(anomalous_predictions)
        
        # Count occurrences
        normal_counts = np.array([np.sum(normal_array == 0), np.sum(normal_array == 1)])
        anomalous_counts = np.array([np.sum(anomalous_array == 0), np.sum(anomalous_array == 1)])
        
        # Calculate percentages
        total_normals = len(normal_array)
        total_anomalous = len(anomalous_array)
        
        normal_percentages = normal_counts / total_normals * 100
        anomalous_percentages = anomalous_counts / total_anomalous * 100
        
        # Calculate total predictions
        total_pred_normal = normal_counts[0] + anomalous_counts[0]
        total_pred_anomalous = normal_counts[1] + anomalous_counts[1]
        
        # Create a figure with grid layout
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3, height_ratios=[3, 2])
        
        # Create axes dictionary to return
        axes_dict = {}
        
        # Main stacked bar chart
        ax_stacked = fig.add_subplot(gs[0, :])
        axes_dict['stacked_bar'] = ax_stacked
        
        labels = ["Predicted Normal", "Predicted Anomalous"]
        width = 0.7
        
        # Create stacked bar chart
        ax_stacked.bar(labels, normal_counts, width, label="True Normal", color=colors[0], alpha=0.7)
        ax_stacked.bar(labels, anomalous_counts, width, bottom=normal_counts, label="True Anomalous", color=colors[1], alpha=0.7)
        
        # Add count labels
        for i, label in enumerate(labels):
            # Label for normal portion
            ax_stacked.text(i, normal_counts[i]/2, f"{normal_counts[i]}\n({normal_percentages[i]:.1f}%)", 
                            ha='center', va='center', color='white', fontweight='bold')
            
            # Label for anomalous portion
            ax_stacked.text(i, normal_counts[i] + anomalous_counts[i]/2, 
                            f"{anomalous_counts[i]}\n({anomalous_percentages[i]:.1f}%)", 
                            ha='center', va='center', color='white', fontweight='bold')
            
            # Total label
            total = normal_counts[i] + anomalous_counts[i]
            ax_stacked.text(i, total, f"Total: {total}", ha='center', va='bottom')
        
        ax_stacked.set_ylabel("Number of Predictions")
        ax_stacked.set_title("Stacked Bar Chart of Predictions")
        ax_stacked.legend(loc='upper right')
        
        # Pie chart for true normal class
        ax_normal_pie = fig.add_subplot(gs[1, 0])
        axes_dict['normal_pie'] = ax_normal_pie
        
        ax_normal_pie.pie(normal_counts, 
                        labels=[f"Pred Normal\n{normal_percentages[0]:.1f}%", f"Pred Anomalous\n{normal_percentages[1]:.1f}%"],
                        colors=[colors[0], colors[1]], 
                        autopct='%1.1f%%',
                        startangle=90,
                        wedgeprops={'alpha': 0.7})
        ax_normal_pie.set_title(f"True Normal Class\n(n={total_normals})")
        
        # Pie chart for true anomalous class
        ax_anomalous_pie = fig.add_subplot(gs[1, 1])
        axes_dict['anomalous_pie'] = ax_anomalous_pie
        
        ax_anomalous_pie.pie(anomalous_counts, 
                            labels=[f"Pred Normal\n{anomalous_percentages[0]:.1f}%", f"Pred Anomalous\n{anomalous_percentages[1]:.1f}%"],
                            colors=[colors[0], colors[1]], 
                            autopct='%1.1f%%',
                            startangle=90,
                            wedgeprops={'alpha': 0.7})
        ax_anomalous_pie.set_title(f"True Anomalous Class\n(n={total_anomalous})")
        
        # Summary statistics panel
        ax_stats = fig.add_subplot(gs[1, 2])
        axes_dict['stats'] = ax_stats
        ax_stats.axis('off')
        
        # Calculate metrics
        true_negative = normal_counts[0]
        false_positive = normal_counts[1]
        false_negative = anomalous_counts[0]
        true_positive = anomalous_counts[1]
        
        total_samples = true_negative + false_positive + false_negative + true_positive
        
        stats_text = (
            f"Summary Statistics:\n\n"
            f"Total Samples: {total_samples}\n"
            f"  - Normal: {total_normals} ({total_normals/total_samples*100:.1f}%)\n"
            f"  - Anomalous: {total_anomalous} ({total_anomalous/total_samples*100:.1f}%)\n\n"
            f"Predicted Normal: {total_pred_normal}\n"
            f"  - Correct: {true_negative} ({true_negative/total_pred_normal*100:.1f}%)\n"
            f"  - Incorrect: {false_negative} ({false_negative/total_pred_normal*100:.1f}%)\n\n"
            f"Predicted Anomalous: {total_pred_anomalous}\n"
            f"  - Correct: {true_positive} ({true_positive/total_pred_anomalous*100:.1f}%)\n"
            f"  - Incorrect: {false_positive} ({false_positive/total_pred_anomalous*100:.1f}%)"
        )
        
        ax_stats.text(
            0.5, 0.5, 
            stats_text,
            transform=ax_stats.transAxes, 
            verticalalignment='center', 
            horizontalalignment='center',
            bbox=dict(
                boxstyle='round',
                facecolor='white',
                alpha=0.8,
                edgecolor='gray'
            ),
            fontsize=9
        )
        
        # Add figure title
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show the plot if requested
        if show_plot:
            plt.show()
        
        return fig, axes_dict

    @staticmethod
    def plot_log_likelihood_kde_with_zoom(
    normal_ll: ArrayLike,
    anomalous_ll: ArrayLike,
    zoom_xlim: Tuple[float, float] = (-10, 10),
    zoom_ylim: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (12, 10),
    save_path: Optional[str] = None,
    color_normal: str = 'green',
    color_anomalous: str = 'red',
    show_plot: bool = True
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
        """
        Generate KDE plots for log-likelihood distributions with a zoomed-in view to compare
        normal and anomalous data distributions.
        
        Parameters
        ----------
        normal_ll : array-like
            Log-likelihood values for normal data
        anomalous_ll : array-like
            Log-likelihood values for anomalous data
        zoom_xlim : tuple of float, optional
            X-axis limits for the zoomed plot, default=(-10, 10)
        zoom_ylim : tuple of float, optional
            Y-axis limits for the zoomed plot, default=None (auto-determined)
        figsize : tuple of float, optional
            Figure size (width, height) in inches, default=(12, 10)
        save_path : str, optional
            Path to save the figure; if None, figure is not saved, default=None
        color_normal : str, optional
            Color for normal data visualization, default='green'
        color_anomalous : str, optional
            Color for anomalous data visualization, default='red'
        show_plot : bool, optional
            Whether to display the plot with plt.show(), default=True
        
        Returns
        -------
        Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]
            The matplotlib figure and axes objects for further customization if needed
        
        Examples
        --------
        >>> import numpy as np
        >>> # Generate sample data
        >>> normal_data = np.random.normal(-2, 1, 1000)
        >>> anomalous_data = np.random.normal(-5, 2, 200)
        >>> # Create and show the plot
        >>> fig, axes = DataVisualizer.plot_log_likelihood_kde_with_zoom(normal_data, anomalous_data)
        """
        # Convert inputs to numpy arrays for consistent handling
        normal_array = np.asarray(normal_ll)
        anomalous_array = np.asarray(anomalous_ll)
        
        # Calculate statistics once to avoid redundant computation
        normal_stats = {
            'median': float(np.median(normal_array)),
            'min': float(np.min(normal_array)),
            'max': float(np.max(normal_array)),
            'mean': float(np.mean(normal_array)),
            'std': float(np.std(normal_array))
        }
        
        anomalous_stats = {
            'median': float(np.median(anomalous_array)),
            'min': float(np.min(anomalous_array)),
            'max': float(np.max(anomalous_array)),
            'mean': float(np.mean(anomalous_array)),
            'std': float(np.std(anomalous_array))
        }
        
        # Create a figure with a grid layout for better flexibility
        fig = plt.figure(figsize=figsize)
        
        # Create a 2x2 grid, with the main plot taking up the top row
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 2], width_ratios=[2, 1])
        
        # Main plot spans the entire top row
        ax1 = fig.add_subplot(gs[0, :])
        # Zoom plot is in the bottom left
        ax2 = fig.add_subplot(gs[1, 0])
        # Add a third subplot for legend and stats
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')  # Hide axis for legend panel
        
        # --- Full range KDE plot (main plot) ---
        sns.kdeplot(
            normal_array, 
            ax=ax1, 
            label='Normal', 
            fill=True, 
            alpha=0.5, 
            color=color_normal,
            bw_adjust=1.0  # Default bandwidth
        )
        
        sns.kdeplot(
            anomalous_array, 
            ax=ax1, 
            label='Anomalous', 
            fill=True, 
            alpha=0.5, 
            color=color_anomalous,
            bw_adjust=1.0
        )
        
        # Add vertical lines for medians with appropriate styling
        ax1.axvline(
            normal_stats['median'], 
            color=color_normal, 
            linestyle='dashed'
        )
        
        ax1.axvline(
            anomalous_stats['median'], 
            color=color_anomalous, 
            linestyle='dashed'
        )
        
        # Set title and labels for top subplot
        ax1.set_title("Log-Likelihood Distributions", fontweight='bold')
        ax1.set_xlabel("Log-Likelihood")
        ax1.set_ylabel("Density")
        ax1.grid(True, alpha=0.3)
        
        # Highlight the zoom region on the main plot
        ax1.axvspan(
            zoom_xlim[0], 
            zoom_xlim[1], 
            alpha=0.1, 
            color='gray',
            zorder=0  # Ensure it's behind other elements
        )
        
        # --- Zoomed-in KDE plot (bottom left subplot) ---
        sns.kdeplot(
            normal_array, 
            ax=ax2, 
            fill=True, 
            alpha=0.5, 
            color=color_normal,
            legend=False  # No need for legend in this plot
        )
        
        sns.kdeplot(
            anomalous_array, 
            ax=ax2, 
            fill=True, 
            alpha=0.5, 
            color=color_anomalous,
            legend=False
        )
        
        # Set zoom limits for x-axis
        ax2.set_xlim(zoom_xlim)
        
        # Set zoom limits for y-axis if provided, otherwise auto-determine
        if zoom_ylim:
            ax2.set_ylim(zoom_ylim)
        
        # Add the same median lines to the zoomed plot
        ax2.axvline(normal_stats['median'], color=color_normal, linestyle='dashed')
        ax2.axvline(anomalous_stats['median'], color=color_anomalous, linestyle='dashed')
        
        # Set title and labels for zoomed subplot
        ax2.set_title("Zoomed View", fontweight='bold')
        ax2.set_xlabel("Log-Likelihood")
        ax2.set_ylabel("Density")
        ax2.grid(True, alpha=0.3)
        
        # --- Legend and statistics panel (bottom right) ---
        # Create a custom legend
        legend_elements = [
            plt.Line2D([0], [0], color=color_normal, lw=4, alpha=0.5, label='Normal'),
            plt.Line2D([0], [0], color=color_anomalous, lw=4, alpha=0.5, label='Anomalous'),
            plt.Line2D([0], [0], color=color_normal, lw=2, linestyle='dashed', 
                    label=f"Normal Median: {normal_stats['median']:.2f}"),
            plt.Line2D([0], [0], color=color_anomalous, lw=2, linestyle='dashed', 
                    label=f"Anomalous Median: {anomalous_stats['median']:.2f}")
        ]
        
        ax3.legend(handles=legend_elements, loc='upper center', fontsize=10)
        
        # Add statistics below the legend
        stats_text = (
            f"Normal Distribution:\n"
            f"  Mean: {normal_stats['mean']:.2f}\n"
            f"  Std Dev: {normal_stats['std']:.2f}\n"
            f"  Min: {normal_stats['min']:.2f}\n"
            f"  Max: {normal_stats['max']:.2f}\n\n"
            f"Anomalous Distribution:\n"
            f"  Mean: {anomalous_stats['mean']:.2f}\n"
            f"  Std Dev: {anomalous_stats['std']:.2f}\n"
            f"  Min: {anomalous_stats['min']:.2f}\n"
            f"  Max: {anomalous_stats['max']:.2f}"
        )
        
        ax3.text(
            0.5, 0.4, 
            stats_text,
            transform=ax3.transAxes, 
            verticalalignment='center', 
            horizontalalignment='center',
            bbox=dict(
                boxstyle='round',
                facecolor='white',
                alpha=0.8,
                edgecolor='gray'
            ),
            fontsize=9
        )
        
        # Add better spacing between subplots
        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show the plot if requested
        if show_plot:
            plt.show()
        
        return fig, (ax1, ax2)