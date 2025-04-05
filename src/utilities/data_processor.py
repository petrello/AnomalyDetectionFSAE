import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import os

class DataProcessor:
    """
    A static class for processing, cleaning, and transforming JSON datasets 
    into structured formats.
    """

    # Mapping from raw variable names (both versions) to unified names
    VARIABLE_MAPPING = {
        # Inverter speeds
        "BORG_1_RL_Speed_RPM": "InverterSpeed_RearLeft_RPM",
        "ECUMeas_ECUInverter_BORG1RL_Speed": "InverterSpeed_RearLeft_RPM",
        "ECU_Meas_ECUInverter_BORG1RL_Speed": "InverterSpeed_RearLeft_RPM",
        "BORG_2_RR_Speed_RPM": "InverterSpeed_RearRight_RPM",
        "ECUMeas_ECUInverter_BORG2RR_Speed": "InverterSpeed_RearRight_RPM",
        "ECU_Meas_ECUInverter_BORG2RR_Speed": "InverterSpeed_RearRight_RPM",

        # Battery parameters
        "ECU_CAN_Pack_Voltage_V": "BatteryVoltage_V",
        "ECUMeas_ECUCAN_PackVoltage": "BatteryVoltage_V",
        "ECU_Meas_ECUCAN_PackVoltage": "BatteryVoltage_V",
        "ECU_CAN_Pack_Current_A": "BatteryCurrent_A",
        "ECUMeas_ECUCAN_PackCurrent": "BatteryCurrent_A",
        "ECU_Meas_ECUCAN_PackCurrent": "BatteryCurrent_A",

        # Inverter fault analysis variables
        "BORG_1_RL_Id_Ref_A": "Inverter_Id_Ref_RearLeft_A",
        "ECUMeas_ECUInverter_BORG1RL_Id_Ref": "Inverter_Id_Ref_RearLeft_A",
        "ECU_Meas_ECUInverter_BORG1RL_Id_Ref": "Inverter_Id_Ref_RearLeft_A",
        "BORG_2_RR_Id_Ref_A": "Inverter_Id_Ref_RearRight_A",
        "ECUMeas_ECUInverter_BORG2RR_Id_Ref": "Inverter_Id_Ref_RearRight_A",
        "ECU_Meas_ECUInverter_BORG2RR_Id_Ref": "Inverter_Id_Ref_RearRight_A",
        "BORG_1_RL_Iq_Ref_A": "Inverter_Iq_Ref_RearLeft_A",
        "ECUMeas_ECUInverter_BORG1RL_Iq_Ref": "Inverter_Iq_Ref_RearLeft_A",
        "ECU_Meas_ECUInverter_BORG1RL_Iq_Ref": "Inverter_Iq_Ref_RearLeft_A",
        "BORG_2_RR_Iq_Ref_A": "Inverter_Iq_Ref_RearRight_A",
        "ECUMeas_ECUInverter_BORG2RR_Iq_Ref": "Inverter_Iq_Ref_RearRight_A",
        "ECU_Meas_ECUInverter_BORG2RR_Iq_Ref": "Inverter_Iq_Ref_RearRight_A",

        # Motor and inverter temperatures
        "BORG_1_RL_T_Mot_degC": "MotorTemp_RearLeft_C",
        "ECUMeas_ECUInverter_BORG1RL_TMot": "MotorTemp_RearLeft_C",
        "ECU_Meas_ECUInverter_BORG1RL_TMot": "MotorTemp_RearLeft_C",
        "BORG_2_RR_T_Mot_degC": "MotorTemp_RearRight_C",
        "ECUMeas_ECUInverter_BORG2RR_TMot": "MotorTemp_RearRight_C",
        "ECU_Meas_ECUInverter_BORG2RR_TMot": "MotorTemp_RearRight_C",
        "BORG_1_RL_T_Heatsink_degC": "InverterTemp_RearLeft_C",
        "ECUMeas_ECUInverter_BORG1RL_THeatsink": "InverterTemp_RearLeft_C",
        "ECU_Meas_ECUInverter_BORG1RL_THeatsink": "InverterTemp_RearLeft_C",
        "BORG_2_RR_T_Heatsink_degC": "InverterTemp_RearRight_C",
        "ECUMeas_ECUInverter_BORG2RR_THeatsink": "InverterTemp_RearRight_C",
        "ECU_Meas_ECUInverter_BORG2RR_THeatsink": "InverterTemp_RearRight_C",

        # Battery pack temperature
        "Pack_Calc_Tmax_packdegC": "BatteryPackTemp_C",
        "ECUMeas_ECUMisc_PackCalc_Tmaxpack": "BatteryPackTemp_C",
        "ECU_Meas_ECUMisc_PackCalc_Tmaxpack": "BatteryPackTemp_C",

        # Failure indicators
        "Errors_Inverter_Fault": "InverterFault",
        "ECU_Meas_ECUMisc_Driver_Errors_InverterFault": "InverterFault",
        "ECUMeas_ECUMisc_Driver_Errors_InverterFault": "InverterFault",
        "outputcluster_ECUMisc_Driver_Errors_InverterFault": "InverterFault",
        # "Errors_Pack_Error": "PackError",
        # "ECU_Meas_ECUMisc_Driver_Errors_PackError": "PackError",
        # "ECUMeas_ECUMisc_Driver_Errors_PackError": "PackError",
        # "outputcluster_ECUMisc_Driver_Errors_PackError": "PackError"
    }


    @staticmethod
    def load_and_concat_json(file_paths: List[str]) -> pd.DataFrame:
        """
        Loads multiple JSON files and concatenates them into a single DataFrame.

        Parameters
        ----------
        file_paths : List[str]
            List of paths to JSON files.

        Returns
        -------
        pd.DataFrame
            Concatenated DataFrame of all raw data.

        Raises
        ------
        ValueError
            If no valid JSON files are loaded.
        """
        data_frames = []

        for path in file_paths:
            try:
                with open(path, 'r') as file:
                    data = json.load(file)
                data_frames.append(pd.DataFrame(data))
                print(f"Loaded {path}, shape: {data_frames[-1].shape}")
            except Exception as e:
                print(f"Error loading {path}: {e}")

        if not data_frames:
            raise ValueError("No valid JSON files loaded.")
        
        concatenated_df = pd.concat(data_frames, ignore_index=True)
        print(f"Total concatenated shape: {concatenated_df.shape}")
        return concatenated_df

    @staticmethod
    def unify_variable_names(df: pd.DataFrame) -> pd.DataFrame:
        """
        Renames columns in the DataFrame to unify names across different file versions.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with raw variable names.

        Returns
        -------
        pd.DataFrame
            DataFrame with unified column names.
        """
        existing_columns = df.columns
        rename_mapping = {col: new_name for col, new_name in DataProcessor.VARIABLE_MAPPING.items() if col in existing_columns}

        df = df.rename(columns=rename_mapping)
        print(f"Renamed {len(rename_mapping)} columns.")
        print(f"Renamed columns: {rename_mapping.keys()}")
        return df

    @staticmethod
    def save_dataset(df: pd.DataFrame, output_path: str, file_format: str = "csv") -> None:
        """
        Saves the DataFrame to disk in the specified format.

        Parameters
        ----------
        df : pd.DataFrame
            The final processed DataFrame.
        output_path : str
            Path to save the dataset.
        file_format : str, optional
            Format for saving ('csv'), default is "csv".
        """
        try:
            if file_format == "csv":
                df.to_csv(output_path, index=False)
            else:
                raise ValueError("Unsupported file format. Choose 'csv'.")

            print(f"Dataset with shape {df.shape}, saved successfully at {output_path} ({file_format}).")
        except Exception as e:
            print(f"Error saving dataset: {e}")
            raise

    @staticmethod
    def filter_dataset_channels(df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters the DataFrame to include only columns specified in the variable mapping.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.

        Returns
        -------
        pd.DataFrame
            The filtered DataFrame.

        Raises
        ------
        Exception
            If an error occurs during filtering.
        """
        try:
            if df.empty:
                print("Empty DataFrame provided. Returning unmodified DataFrame.")
                return df

            valid_column_names = set(DataProcessor.VARIABLE_MAPPING.values())
            filtered_columns = [col for col in df.columns if col in valid_column_names]

            if not filtered_columns:
                print("No matching columns found. Returning unmodified DataFrame.")
                return df

            print(f"Columns retained in DataFrame: {filtered_columns}")
            df = df[filtered_columns]
            return df

        except Exception as e:
            print(f"Error filtering and renaming dataset: {e}")
            raise

    @staticmethod
    def concatenate_datasets(dataset_folder: str, dataset_files: Dict[str, str], output_file: str) -> pd.DataFrame:
        """
        Concatenates multiple datasets into a single DataFrame while adding a 'Test_Day' column.

        Parameters
        ----------
        dataset_folder : str
            Path to the folder containing dataset files.
        dataset_files : Dict[str, str]
            Dictionary mapping test days (str) to dataset filenames (str).
        output_file : str
            Path to save the final concatenated dataset.

        Returns
        -------
        pd.DataFrame
            The concatenated dataset.

        Raises
        ------
        FileNotFoundError
            If no valid datasets are found for concatenation.
        Exception
            If an error occurs during loading or saving.
        """
        df_list = []

        for test_day, filename in dataset_files.items():
            file_path = os.path.join(dataset_folder, filename)

            if not os.path.exists(file_path):
                print(f"File not found: {file_path}. Skipping...")
                continue

            try:
                df = pd.read_csv(file_path)
                df["Test_Day"] = test_day
                df_list.append(df)
                print(f"Loaded dataset: {file_path} with shape {df.shape}")
            except Exception as e:
                print(f"Failed to load {file_path}: {str(e)}")
        
        if not df_list:
            raise FileNotFoundError("No valid datasets found for concatenation. Check file paths and formats.")

        final_dataset = pd.concat(df_list, ignore_index=True)
        print(f"Final dataset shape after concatenation: {final_dataset.shape}")

        try:
            final_dataset.to_csv(output_file, index=False)
            print(f"Final concatenated dataset saved at {output_file}")
        except Exception as e:
            print(f"Failed to save final dataset: {str(e)}")
            raise

        return final_dataset

    @staticmethod
    def aggregate_time_series(df: pd.DataFrame, window_size: int = 10, method: str = 'mean') -> pd.DataFrame:
        """
        Aggregates time series data while preserving the original order.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing sequential data.
        window_size : int, optional
            Number of rows per aggregation window, default is 10.
        method : str, optional
            Aggregation method ('mean', 'median', 'min', 'max'), default is 'mean'.

        Returns
        -------
        pd.DataFrame
            DataFrame with aggregated values while preserving original structure.

        Raises
        ------
        ValueError
            If an invalid aggregation method is specified.
        """
        valid_methods = ['mean', 'median', 'min', 'max']
        if method not in valid_methods:
            raise ValueError(f"Invalid aggregation method: {method}. Choose from {valid_methods}")
        
        if window_size < 2:
            print("Window size is less than 2. No aggregation performed.")
            return df.copy()
        
        result_df = df.copy()
        numeric_columns = result_df.columns.difference(['InverterFault'])
        print("numeric_columns", numeric_columns)
        binary_columns = ['InverterFault']
        
        result_df['group_index'] = np.arange(len(result_df)) // window_size
        
        aggregated = result_df.groupby('group_index').agg({
            **{col: method for col in numeric_columns},
            **{col: 'max' for col in binary_columns}
        }).reset_index()
        
        final_result = []
        
        for group_index in sorted(result_df['group_index'].unique()):
            agg_row = aggregated[aggregated['group_index'] == group_index]
            
            if not agg_row.empty:
                group_rows = result_df[result_df['group_index'] == group_index]
                final_row = group_rows.iloc[0].copy()
                
                for col in list(numeric_columns) + binary_columns:
                    final_row[col] = agg_row[col].values[0]
                
                final_result.append(final_row)
        
        result_df = pd.DataFrame(final_result).drop(columns='group_index').reset_index(drop=True)
        print(f"Aggregated data using {method} over windows of size {window_size}. New shape: {result_df.shape}")
        
        return result_df