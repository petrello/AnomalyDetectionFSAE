import pandas as pd

class DBNDataTransformer:
    """
    Provides methods for transforming data to formats suitable for Dynamic Bayesian Networks (DBNs).
    """

    @staticmethod
    def create_time_series_dbn_data(df: pd.DataFrame, time_slices: int = 2) -> pd.DataFrame:
        """
        Transforms a dataset into a time-series format required for DBN training.

        This static method converts a standard DataFrame into a time-series format compatible with
        Dynamic Bayesian Network (DBN) training. It creates MultiIndex columns representing
        variables at different time slices.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame without explicit time slices.
        time_slices : int, optional
            The number of consecutive time slices to include in the transformed DataFrame,
            default is 2.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame with MultiIndex columns, where each column is a tuple
            of (variable_name, time_slice).

        Raises
        ------
        ValueError
            If time_slices is less than 2, as DBNs require at least two time slices.

        Examples
        --------
        >>> import pandas as pd
        >>> data = {'A': [1, 2, 3, 4, 5], 'B': [6, 7, 8, 9, 10]}
        >>> df = pd.DataFrame(data)
        >>> transformed_df = DBNDataTransformer.create_time_series_dbn_data(df, time_slices=2)
        >>> print(transformed_df)
        """
        if time_slices < 2:
            raise ValueError("DBN requires at least two consecutive time slices (0 and 1).")

        transformed_data = {}

        # Generate time-slice shifted copies
        for t in range(time_slices):
            df_shifted = df.shift(-t)  # Align timestamps
            # Create a tuple of (variable_name, time_slice) for each column
            tuple_columns = [(col, t) for col in df.columns]
            df_shifted.columns = pd.MultiIndex.from_tuples(tuple_columns)
            transformed_data[t] = df_shifted

        # Concatenate all time slices
        df_time_series = pd.concat(transformed_data.values(), axis=1)

        # Remove rows with NaN values
        df_time_series = df_time_series.dropna()

        # Reset index to ensure consecutive indices
        df_time_series = df_time_series.reset_index(drop=True)

        return df_time_series