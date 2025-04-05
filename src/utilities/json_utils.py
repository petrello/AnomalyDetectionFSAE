import json
import os
import re
from typing import Any, Dict, List, Optional, Union, cast


class JSONUtils:
    """
    A utility class for working with JSON data structures.
    
    This class provides static methods to explore, extract, and manipulate JSON data.
    All methods are designed to be called directly from the class without instantiation.
    """
    
    @staticmethod
    def print_keys(data: Union[Dict[str, Any], List[Any]], prefix: str = "") -> None:
        """
        Print all keys present in the JSON data with their hierarchy.
        
        Parameters
        ----------
        data : Union[Dict[str, Any], List[Any]]
            Dictionary or list containing JSON data
        prefix : str, default=""
            Prefix to track hierarchy in nested structures
        """
        if isinstance(data, dict):
            for key, value in data.items():
                full_key = f"{prefix}.{key}" if prefix else key
                print(full_key)  # Print the key with its hierarchy
                JSONUtils.print_keys(value, full_key)  # Recursively process nested structures
                
        elif isinstance(data, list) and data:
            # Avoid printing a prefix that has already been printed
            if prefix:
                print(f"{prefix}[]")  # Indicate that this is a list
            
            # Analyze only the first element to understand the structure
            first_element = data[0] if data and isinstance(data[0], (dict, list)) else None
            if first_element:
                JSONUtils.print_keys(first_element, prefix)

    @staticmethod
    def get_keys(data: Dict[str, Any], regex_pattern: Optional[str] = None) -> List[str]:
        """
        Extract top-level keys from a Python dictionary with optional regex filtering.
        
        Parameters
        ----------
        data : Dict[str, Any]
            The JSON-like dictionary to extract keys from
        regex_pattern : Optional[str], default=None
            A regex pattern to filter the keys. If None, returns all top-level keys
            
        Returns
        -------
        List[str]
            A list of top-level keys matching the regex pattern (or all keys if no regex is provided)
            
        Raises
        ------
        ValueError
            If the input data is not a dictionary
        """
        if not isinstance(data, dict):
            raise ValueError("Input data must be a dictionary")

        # Get all top-level keys
        keys = list(data.keys())

        # Apply regex filtering if a pattern is provided
        if regex_pattern:
            regex = re.compile(regex_pattern, re.IGNORECASE)
            keys = [key for key in keys if regex.search(key)]

        return keys

    @staticmethod
    def extract_value(data: Union[Dict[str, Any], List[Any]], target_key: str) -> Dict[str, Any]:
        """
        Search for a specific key in a JSON structure and return all associated values.
        
        If the found values are lists, they are flattened to avoid nested lists.
        
        Parameters
        ----------
        data : Union[Dict[str, Any], List[Any]]
            The JSON data to search, which can be a dictionary or a list
        target_key : str
            The key to search for
            
        Returns
        -------
        Dict[str, Any]
            A dictionary with the key and a flattened list of all found values.
            If only one value is found, it returns that value directly instead of a list.
            If no values are found, returns an empty dictionary.
        """
        results: List[Any] = []

        def _search(sub_data: Union[Dict[str, Any], List[Any]]) -> None:
            """Recursively search for the key in the JSON structure."""
            if isinstance(sub_data, dict):
                for key, value in sub_data.items():
                    if key == target_key:
                        if isinstance(value, list):
                            results.extend(value)  # Flatten lists
                        else:
                            results.append(value)  # Append non-list values
                    if isinstance(value, (dict, list)):
                        _search(value)  # Continue search recursively
            elif isinstance(sub_data, list):
                for item in sub_data:
                    if isinstance(item, (dict, list)):
                        _search(item)

        _search(data)

        # If results contain only one element, return that element directly instead of a list
        return {target_key: results[0] if len(results) == 1 else results} if results else {}

    @staticmethod
    def save_query_result(
        data: Any, 
        key: str, 
        *, 
        minified: bool = True, 
        output_folder: str = "results"
    ) -> str:
        """
        Save the result of a JSON query to a separate file.
        
        Parameters
        ----------
        data : Any
            Data to save
        key : str
            Name of the key that was searched for (used for filename)
        minified : bool, default=True
            If True, the JSON output will be minified
        output_folder : str, default="results"
            Output folder where the file will be saved
            
        Returns
        -------
        str
            Path of the saved file, or empty string if saving failed
        """
        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, f"{key}.json")

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                if minified:
                    json.dump(data, f, separators=(',', ':'))
                else:
                    json.dump(data, f, indent=4)
            print(f"Results saved to: {output_file}")
            return output_file
        except Exception as e:
            print(f"Error saving JSON file: {e}")
            return ""

    @staticmethod
    def load_json(file_path: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
        """
        Load a JSON file into memory.
        
        Parameters
        ----------
        file_path : str
            Path to the JSON file
            
        Returns
        -------
        Optional[Union[Dict[str, Any], List[Any]]]
            The content of the JSON file, or None if loading failed
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return cast(Union[Dict[str, Any], List[Any]], json.load(f))
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return None
    
    @staticmethod
    def dump_json(obj: Union[Dict[str, Any], List[Any]], file_path: str) -> bool:
        """
        Dump a Python object into a JSON file.

        Parameters
        ----------
        obj : Union[Dict[str, Any], List[Any]]
            The Python object to dump into the JSON file.
        file_path : str
            Path to the JSON file where the object will be dumped.

        Returns
        -------
        bool
            True if the object was successfully dumped, False otherwise.
        """
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=4)
            return True
        except Exception as e:
            print(f"Error dumping JSON file: {e}")
            return False