import re
import json
import scipy.io as sio
import numpy as np
from typing import Any, Dict, Union, Optional


class MatToJsonConverter:
    """
    A class for converting MATLAB .mat files to JSON format.
    
    This class provides methods to transform MATLAB data structures, including nested
    structs and arrays, into JSON-compatible Python objects and save them to a file.
    The converter handles MATLAB-specific data types and performs appropriate conversions
    to ensure JSON compatibility.
    
    Examples:
        Basic usage:
        >>> MatToJsonConverter.convert_file("input.mat", "output.json")
        
        With formatted output:
        >>> MatToJsonConverter.convert_file("input.mat", "output.json", minified=False)
        
        With debugging information:
        >>> MatToJsonConverter.convert_file("input.mat", "output.json", debug=True)
    """
    
    @staticmethod
    def __mat_struct_to_dict(mat_obj: Any) -> Any:
        """
        Recursively converts a MATLAB struct object into a Python dictionary.
        
        This method handles the conversion of MATLAB-specific data structures into
        Python native types that can be easily serialized to JSON.
        
        Args:
            mat_obj: The MATLAB struct, array or other data type to be converted
        
        Returns:
            A Python dictionary, list, or native type representing the input data
        """
        result: Dict[str, Any] = {}
        
        # Handle MATLAB struct objects
        if isinstance(mat_obj, sio.matlab._mio5_params.mat_struct):
            for field_name in mat_obj._fieldnames:
                result[field_name] = MatToJsonConverter.__mat_struct_to_dict(getattr(mat_obj, field_name))
                
        # Handle NumPy arrays
        elif isinstance(mat_obj, np.ndarray):
            if mat_obj.dtype.names:  # Structured array with named fields
                return {field: MatToJsonConverter.__mat_struct_to_dict(mat_obj[field]) 
                        for field in mat_obj.dtype.names}
            else:  # Regular array
                return mat_obj.tolist()
                
        # Convert NumPy scalars to Python native types
        elif isinstance(mat_obj, (np.generic, np.number)):
            return mat_obj.item()
            
        # Return as-is for strings, None, etc.
        else:
            return mat_obj
            
        return result

    @staticmethod
    def __convert_for_json(value: Any) -> Any:
        """
        Recursively converts NumPy arrays and other non-serializable objects to JSON-compatible types.
        
        Args:
            value: The value to be converted to a JSON-serializable type
            
        Returns:
            A JSON-serializable version of the input value
        """
        if isinstance(value, np.ndarray):
            return value.tolist()  # Convert NumPy array to a list
        elif isinstance(value, dict):
            return {k: MatToJsonConverter.__convert_for_json(v) for k, v in value.items()}
        elif isinstance(value, (np.generic, np.number)):
            return value.item()  # Convert NumPy scalar to Python scalar
        else:
            return value  # Return as-is for standard Python types

    @staticmethod
    def convert_file(source_filename: str, dest_filename: str, *, 
                     minified: bool = True, debug: bool = False) -> bool:
        """
        Converts a MATLAB .mat file into a JSON file.
        
        This method handles the loading of MATLAB data, extraction of relevant structures,
        and saving to JSON format. It can specifically process FSAE log files with
        ECU_Meas data.
        
        Args:
            source_filename: Path to the input MATLAB .mat file
            dest_filename: Path to the output JSON file
            minified: If True, the JSON output will be minified. Defaults to True
            debug: If True, enables debug mode with additional console output. Defaults to False
            
        Returns:
            bool: True if conversion was successful, False otherwise
        """
        print(f"Loading MATLAB file: {source_filename}")
        
        try:
            # Load the MATLAB file with struct_as_record=False to get mat_struct objects
            mat_data = sio.loadmat(source_filename, struct_as_record=False, squeeze_me=True)
        except Exception as e:
            print(f"Error loading MATLAB file: {e}")
            return False
        
        if debug:
            print(f"Top-Level Keys in .mat file: {list(mat_data.keys())}")

        # Remove MATLAB system keys
        for key_to_remove in ['__header__', '__version__', '__globals__']:
            mat_data.pop(key_to_remove, None)

        # Look for FSAE log key pattern (e.g., "FSAE_Log_123")
        fsae_log_key = next((key for key in mat_data.keys() 
                           if re.match(r"^FSAE_Log_\d+", key)), None)

        if fsae_log_key:
            print(f"Found FSAE log key: {fsae_log_key}")
            
            # Extract ECU_Meas data if available
            ecu_meas = getattr(mat_data[fsae_log_key], "ECU_Meas", None)

            if ecu_meas is not None:
                mat_data = ecu_meas
            else:
                print(f"'ECU_Meas' key not found inside '{fsae_log_key}'. Unable to process.")
                return False

        # Convert MATLAB structure to Python dictionary
        extracted_data = MatToJsonConverter.__mat_struct_to_dict(mat_data)

        # Prepare data for JSON serialization
        converted_data = {key: MatToJsonConverter.__convert_for_json(value) 
                         for key, value in extracted_data.items()}

        if debug:
            print(f"Extracted Data Structure: {list(converted_data.keys())}")

        try:
            # Write to JSON file with appropriate formatting
            with open(dest_filename, "w") as f:
                json.dump(
                    converted_data, 
                    f, 
                    separators=(',', ':') if minified else None, 
                    indent=4 if not minified else None
                )
        except Exception as e:
            print(f"Error saving JSON file: {e}")
            return False

        print(f"Successfully converted MATLAB file to JSON: {dest_filename}")
        return True

    @staticmethod
    def extract_data(source_filename: str, *, debug: bool = False) -> Optional[Dict[str, Any]]:
        """
        Extracts data from a MATLAB .mat file without saving to JSON.
        
        This method is useful when you want to process the extracted data in memory
        without writing to a file.
        
        Args:
            source_filename: Path to the input MATLAB .mat file
            debug: If True, enables debug mode with additional console output. Defaults to False
            
        Returns:
            Dict or None: Converted data dictionary if successful, None otherwise
        """
        print(f"Loading MATLAB file: {source_filename}")
        
        try:
            mat_data = sio.loadmat(source_filename, struct_as_record=False, squeeze_me=True)
        except Exception as e:
            print(f"Error loading MATLAB file: {e}")
            return None
            
        if debug:
            print(f"Top-Level Keys in .mat file: {list(mat_data.keys())}")

        # Remove MATLAB system keys
        for key_to_remove in ['__header__', '__version__', '__globals__']:
            mat_data.pop(key_to_remove, None)

        # Look for FSAE log key pattern
        fsae_log_key = next((key for key in mat_data.keys() 
                           if re.match(r"^FSAE_Log_\d+", key)), None)

        if fsae_log_key:
            print(f"Found FSAE log key: {fsae_log_key}")
            
            ecu_meas = getattr(mat_data[fsae_log_key], "ECU_Meas", None)

            if ecu_meas is not None:
                mat_data = ecu_meas
            else:
                print(f"'ECU_Meas' key not found inside '{fsae_log_key}'. Unable to process.")
                return None

        # Convert MATLAB structure to Python dictionary
        extracted_data = MatToJsonConverter.__mat_struct_to_dict(mat_data)

        # Prepare data for JSON serialization
        converted_data = {key: MatToJsonConverter.__convert_for_json(value) 
                         for key, value in extracted_data.items()}
                         
        if debug:
            print(f"Extracted Data Structure: {list(converted_data.keys())}")
            
        return converted_data