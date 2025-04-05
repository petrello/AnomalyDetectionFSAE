# Import all utility classes for easy access when importing the package

from .data_discretizer import DataDiscretizer
from .data_encoder import DataEncoder
from .data_processor import DataProcessor
from .data_visualizer import DataVisualizer
from .dbn_data_transformer import DBNDataTransformer
from .dbn_model import DBNModel
from .granger_causality_analyzer import GrangerCausalityAnalyzer
from .json_utils import JSONUtils
from .mat_to_json_converter import MatToJsonConverter

# Define what gets imported with "from utilities import *"
__all__ = [
    'DataDiscretizer',
    'DataEncoder',
    'DataProcessor',
    'DataVisualizer',
    'DBNDataTransformer',
    'DBNModel',
    'GrangerCausalityAnalyzer',
    'JSONUtils',
    'MatToJsonConverter'
]

# Package metadata
__version__ = '0.1.0'


# # Example usage
# # Import the whole package
# import utilities

# # Use a class
# processor = utilities.DataProcessor()

# # Or import specific classes
# from utilities import DataVisualizer, GrangerCausalityAnalyzer

# # Or import everything (only imports what's in __all__)
# from utilities import *