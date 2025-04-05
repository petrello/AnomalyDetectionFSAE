import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from typing import Dict, Tuple, Literal, Union

class DataEncoder:
    """
    Provides methods for encoding categorical and boolean columns in a DataFrame.
    """

    @staticmethod
    def encode_categorical_columns(
        df: pd.DataFrame, 
        encoding_strategy: Literal['ordinal', 'label'] = 'ordinal', 
        handle_unknown: Literal['error', 'use_encoded_value'] = 'error'
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Union[str, list, dict]]]]:
        """
        Encodes categorical and boolean columns in a DataFrame to integer representations.

        This static method transforms categorical and boolean columns within a DataFrame into numerical
        representations. It supports two encoding strategies: ordinal and label encoding, and provides
        options for handling unknown categories during the encoding process.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing categorical and boolean columns to be encoded.
        encoding_strategy : {'ordinal', 'label'}, optional
            The encoding strategy to apply to categorical columns:
            - 'ordinal': Uses OrdinalEncoder to preserve any inherent order in categories.
            - 'label': Uses LabelEncoder to assign arbitrary integer labels.
            Default is 'ordinal'.
        handle_unknown : {'error', 'use_encoded_value'}, optional
            Specifies how to handle unknown categories during the encoding transform:
            - 'error': Raises an error if unknown categories are encountered.
            - 'use_encoded_value': Replaces unknown categories with a specified encoded value.
            Default is 'error'.

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, Dict[str, Union[str, list, dict]]]]
            A tuple containing:
            - The DataFrame with encoded categorical and boolean columns.
            - A dictionary containing encoding details for each encoded column.

        Raises
        ------
        ValueError
            If an invalid encoding strategy is specified.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> # Create a sample DataFrame
        >>> data = {'category': ['A', 'B', 'C', 'A'], 'boolean': [True, False, True, False]}
        >>> df = pd.DataFrame(data)
        >>> # Encode categorical columns using ordinal encoding
        >>> encoded_df, encoding_info = DataEncoder.encode_categorical_columns(df)
        >>> print(encoded_df)
        >>> print(encoding_info)
        """
        encoded_df = df.copy()
        encoding_info = {}
        cat_bool_columns = df.select_dtypes(include=['category', 'object', 'bool']).columns

        for col in cat_bool_columns:
            if df[col].dtype == 'bool':
                encoded_df[col] = df[col].astype(np.int8)
                encoding_info[col] = {
                    'type': 'boolean',
                    'mapping': {False: 0, True: 1}
                }
                continue

            try:
                if encoding_strategy == 'ordinal':
                    encoder = OrdinalEncoder(handle_unknown=handle_unknown, dtype=np.int8)
                    encoded_col = encoder.fit_transform(df[[col]])
                    encoded_df[col] = encoded_col.flatten().astype(np.int8)
                    encoding_info[col] = {
                        'type': 'ordinal',
                        'categories': list(encoder.categories_[0]),
                        'mapping': dict(zip(encoder.categories_[0], range(len(encoder.categories_[0]))))
                    }

                elif encoding_strategy == 'label':
                    encoder = LabelEncoder()
                    encoded_col = encoder.fit_transform(df[col])
                    encoded_df[col] = encoded_col.astype(np.int8)
                    encoding_info[col] = {
                        'type': 'label',
                        'categories': list(encoder.classes_),
                        'mapping': dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
                    }

                else:
                    raise ValueError(f"Invalid encoding strategy: {encoding_strategy}")

            except Exception as e:
                print(f"Error encoding column {col}: {e}")
                raise

        return encoded_df, encoding_info