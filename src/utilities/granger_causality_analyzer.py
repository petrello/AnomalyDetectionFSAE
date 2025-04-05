from typing import Dict, List, Optional, Tuple, Union, Literal
import numpy as np
import pandas as pd
import pandas.io.formats.style as pdstyle
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.preprocessing import OrdinalEncoder
from statsmodels.tsa.stattools import grangercausalitytests
import warnings


class GrangerCausalityAnalyzer:
    """
    A class containing utilities for performing and visualizing Granger causality tests.
    
    This class provides static methods to analyze time series data for Granger causality 
    relationships, visualize the results through heatmaps, and organize significant 
    relationships into formatted tables.
    
    All methods are static or class methods, meaning they can be called directly from the class
    without instantiation.
    """
    
    @staticmethod
    def granger_causality_matrix(
        data: pd.DataFrame, 
        variables: List[str], 
        max_lag: int = 2, 
        test: Literal['ssr_chi2test', 'ssr_ftest', 'lrtest', 'params_ftest'] = 'ssr_chi2test', 
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Compute the Granger causality between all pairs of variables in the dataset.
        
        Parameters
        ----------
        data : pd.DataFrame
            Pandas DataFrame containing the time series variables
        variables : List[str]
            List of variable names to test
        max_lag : int, default=2
            Maximum lag to test for
        test : {'ssr_chi2test', 'ssr_ftest', 'lrtest', 'params_ftest'}, default='ssr_chi2test'
            Which test to use for determining Granger causality
        verbose : bool, default=False
            If True, print detailed results of each test
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing p-values for Granger causality tests, with rows representing
            causes (from) and columns representing effects (to)
        """
        # Suppress warnings
        warnings.filterwarnings('ignore')
        
        # Create a copy of the data to avoid modifying the original
        df = data.copy()
        encoder = OrdinalEncoder()
        
        # Check if any columns are not numeric and encode them
        non_numeric_cols = [col for col in variables if not pd.api.types.is_numeric_dtype(df[col])]
        if non_numeric_cols:
            df[non_numeric_cols] = encoder.fit_transform(df[non_numeric_cols])
        
        # Select only the specified variables
        df = df[variables]
        
        # Initialize the p-values matrix with ones
        p_values = pd.DataFrame(
            np.ones((len(variables), len(variables))),
            index=variables, 
            columns=variables
        )
        
        # Calculate Granger causality for each combination of variables
        for c in itertools.combinations(variables, 2):
            # For X->Y (c[0] causes c[1])
            test_result = grangercausalitytests(df[[c[1], c[0]]], maxlag=max_lag, verbose=verbose)
            p_values.loc[c[0], c[1]] = test_result[max_lag][0][test][1]
            
            # For Y->X (c[1] causes c[0])
            test_result = grangercausalitytests(df[[c[0], c[1]]], maxlag=max_lag, verbose=verbose)
            p_values.loc[c[1], c[0]] = test_result[max_lag][0][test][1]
        
        return p_values

    @staticmethod
    def plot_granger_heatmap(
        p_values: pd.DataFrame, 
        significance_level: float = 0.05, 
        title: str = "Granger Causality P-values"
    ) -> plt.Axes:
        """
        Plot a lower triangular heatmap of Granger causality p-values.
        
        Parameters
        ----------
        p_values : pd.DataFrame
            DataFrame containing p-values from Granger causality tests
        significance_level : float, default=0.05
            P-value threshold for significance
        title : str, default="Granger Causality P-values"
            Title for the plot
            
        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib Axes object for further customization if needed
        """
        # Create a mask for the upper triangle and diagonal
        mask = np.zeros_like(p_values, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        
        # Create figure with appropriate size
        plt.figure(figsize=(18, 14))
        
        # Create a more distinguishable colormap for better visualization
        cmap = sns.color_palette("YlOrRd_r", as_cmap=True)
        
        # Determine an appropriate maximum value for the colormap
        # Cap for better color contrast
        v_max = min(0.2, p_values.where(~mask).max().max())
        
        # Plot the heatmap (lower triangle only)
        ax = sns.heatmap(
            p_values, 
            annot=True,  # Show p-values on the heatmap
            cmap=cmap,
            vmin=0,  # Minimum p-value is 0
            vmax=v_max,
            mask=mask,
            fmt=".3f",  # Format p-values to 3 decimal places
            linewidths=0.5,
            annot_kws={"size": 11},
            cbar_kws={'label': 'p-value'}
        )
        
        # Add significance level line on colorbar for reference
        cbar = ax.collections[0].colorbar
        cbar.ax.axhline(y=significance_level, color='blue', linestyle='--', linewidth=2)
        cbar.ax.text(
            3.0, significance_level, 
            f'α = {significance_level}', 
            color='blue', ha='left', va='center', fontsize=12
        )
        
        # Add titles and labels
        plt.title(title, fontsize=24, pad=20)
        plt.xlabel("Effect (to)", fontsize=16, labelpad=10)
        plt.ylabel("Cause (from)", fontsize=16, labelpad=10)
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        return ax

    @staticmethod
    def create_significant_relationships_table(
        p_values: pd.DataFrame, 
        significance_level: float = 0.05
    ) -> pd.DataFrame:
        """
        Create a formatted table of significant Granger causal relationships.
        
        Parameters
        ----------
        p_values : pd.DataFrame
            DataFrame containing p-values from Granger causality tests
        significance_level : float, default=0.05
            P-value threshold for significance
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing significant relationships sorted by p-value,
            or a message if no significant relationships are found
        """
        # Extract significant relationships as list of dictionaries
        significant_pairs = []
        for i in p_values.index:
            for j in p_values.columns:
                # Check if p-value is below significance level and not comparing to itself
                if i != j and p_values.loc[i, j] < significance_level:
                    significant_pairs.append({
                        'From': i,
                        'To': j,
                        'P-value': p_values.loc[i, j]
                    })
        
        # Create DataFrame from the significant pairs
        if significant_pairs:
            # Create DataFrame and sort by p-value (ascending)
            result_df = pd.DataFrame(significant_pairs).sort_values('P-value')
            
            # Add significance level indicators (* for p<0.05, ** for p<0.01, *** for p<0.001)
            def get_significance_stars(p_value: float) -> str:
                """Helper function to add significance stars based on p-value."""
                if p_value < 0.001:
                    return "***"
                elif p_value < 0.01:
                    return "**"
                elif p_value < 0.05:
                    return "*"
                return ""
            
            result_df['Significance'] = result_df['P-value'].apply(get_significance_stars)
            
            return result_df
        else:
            # Return DataFrame with a message if no significant relationships found
            return pd.DataFrame({'Message': [f'No significant relationships found (α = {significance_level})']})

    @staticmethod
    def display_granger_results(
        data: pd.DataFrame, 
        variables: List[str], 
        max_lag: int = 2, 
        significance_level: float = 0.05,
        title: str = 'Granger Causality P-values',
        test: Literal['ssr_chi2test', 'ssr_ftest', 'lrtest', 'params_ftest'] = 'ssr_chi2test'
    ) -> Tuple[pd.DataFrame, Union[pdstyle.Styler, pd.DataFrame]]:
        """
        Compute and display Granger causality results.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data containing time series variables
        variables : List[str]
            List of variable names to analyze
        max_lag : int, default=2
            Maximum lag for Granger test
        significance_level : float, default=0.05
            Significance threshold
        test : {'ssr_chi2test', 'ssr_ftest', 'lrtest', 'params_ftest'}, default='ssr_chi2test'
            Test type for Granger causality
            
        Returns
        -------
        Tuple[pd.DataFrame, Union[pdstyle.Styler, pd.DataFrame]]
            A tuple containing:
            - p_values: DataFrame with all p-values
            - styled_table: Styled DataFrame with significant relationships or message DataFrame
        """
        # Compute Granger causality matrix
        p_values = GrangerCausalityAnalyzer.granger_causality_matrix(
            data, variables, max_lag, test
        )
        
        # Plot the heatmap
        GrangerCausalityAnalyzer.plot_granger_heatmap(
            p_values, 
            significance_level, 
            title=f"{title} (max_lag={max_lag})"
        )
        plt.show()
        
        # Create table of significant relationships
        sig_relationships = GrangerCausalityAnalyzer.create_significant_relationships_table(
            p_values, significance_level
        )
        
        # Display results based on whether significant relationships were found
        if 'Message' in sig_relationships.columns:
            # No significant relationships found
            print(sig_relationships['Message'].iloc[0])
            return p_values, sig_relationships
        else:
            # Count significant relationships by significance level
            strong = sum(sig_relationships['P-value'] < 0.001)
            medium = sum((sig_relationships['P-value'] >= 0.001) & (sig_relationships['P-value'] < 0.01))
            weak = sum((sig_relationships['P-value'] >= 0.01) & (sig_relationships['P-value'] < 0.05))
            
            # Print summary of findings
            print(f"Found {len(sig_relationships)} significant Granger causal relationships:")
            print(f"  - Strong (p < 0.001): {strong}")
            print(f"  - Medium (p < 0.01): {medium}")
            print(f"  - Weak (p < 0.05): {weak}")
            print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05\n")
            
            # Create styled table with color gradient
            styled_table = sig_relationships.style.background_gradient(
                cmap='YlOrRd_r', subset=['P-value']
            ).format({'P-value': '{:.4f}'})
            
            return p_values, styled_table

    @staticmethod
    def filter_table(
        styled_table: pdstyle.Styler, 
        exclude_from: Optional[Union[str, List[str]]] = None, 
        exclude_to: Optional[Union[str, List[str]]] = None
    ) -> pd.DataFrame:
        """
        Get filtered DataFrame from a styled table.
        
        Parameters
        ----------
        styled_table : pdstyle.Styler
            Styled DataFrame to filter
        exclude_from : str or List[str], optional
            Values to exclude from 'From' column
        exclude_to : str or List[str], optional
            Values to exclude from 'To' column
            
        Returns
        -------
        pd.DataFrame
            Filtered DataFrame (not styled)
        """
        # Get the underlying DataFrame from the styled table
        df = styled_table.data.copy()
        
        # Filter From column if specified
        if exclude_from is not None:
            if isinstance(exclude_from, str):
                exclude_from = [exclude_from]
            df = df[~df['From'].isin(exclude_from)]
        
        # Filter To column if specified
        if exclude_to is not None:
            if isinstance(exclude_to, str):
                exclude_to = [exclude_to]
            df = df[~df['To'].isin(exclude_to)]
        
        # Check if the filtered DataFrame is empty
        if df.empty:
            print("No rows remain after filtering!")
            return pd.DataFrame({'Message': ['No data after filtering']})
    
        # Re-apply styling
        styled_df = df.style.background_gradient(
            cmap='YlOrRd_r', subset=['P-value']
        ).format({'P-value': '{:.4f}'})

        return styled_df
    
    @staticmethod
    def filter_and_style_table(
        sig_relationships: pd.DataFrame, 
        filter_conditions: Dict[str, Union[str, List[str]]]
    ) -> pdstyle.Styler:
        """
        Filter a DataFrame and then apply styling.
        
        Parameters
        ----------
        sig_relationships : pd.DataFrame
            The DataFrame with Granger causality results
        filter_conditions : Dict[str, Union[str, List[str]]]
            Dictionary with column names as keys and filter conditions as values
            
        Returns
        -------
        pdstyle.Styler
            The filtered and styled DataFrame, or a message if no rows remain
        """
        # Create a copy to avoid modifying the original
        df = sig_relationships.copy()
        
        # Apply each filter condition in sequence
        for column, value in filter_conditions.items():
            if isinstance(value, list):
                # If value is a list, exclude all rows where the column value is in the list
                df = df[~df[column].isin(value)]
            else:
                # If value is a single item, exclude rows where the column equals that value
                df = df[df[column] != value]
        
        # Check if the filtered DataFrame is empty
        if df.empty:
            print("No rows remain after filtering.")
            return pd.DataFrame({'Message': ['No rows remain after filtering.']}).style
        
        # Make sure P-Value is numeric for proper gradient coloring
        p_value_col = 'P-value'
        if p_value_col in df.columns and df[p_value_col].dtype == object:
            # Create a numeric column for the gradient
            df['P-value_numeric'] = df[p_value_col].astype(float)
            
            # Apply styling with gradient background
            styled = df.style.background_gradient(
                cmap='YlOrRd_r', subset=['P-value_numeric']
            ).format({p_value_col: '{:.4f}'})
            
            # Hide the numeric column used for styling
            styled = styled.hide(axis='columns', names=['P-value_numeric'])
        else:
            # Apply styling directly if P-value is already numeric
            styled = df.style.background_gradient(
                cmap='YlOrRd_r', subset=[p_value_col]
            ).format({p_value_col: '{:.4f}'})
        
        return styled