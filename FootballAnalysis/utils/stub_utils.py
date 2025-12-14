import pickle
import os

def read_stub(read_from_stub, stub_path):
    """
    Read data from a stub file if requested and file exists.
    
    Args:
        read_from_stub (bool): Whether to read from stub file
        stub_path (str): Path to the stub file
    
    Returns:
        The loaded data if successful, None otherwise
    """
    if read_from_stub and stub_path is not None and os.path.exists(stub_path):
        try:
            with open(stub_path, 'rb') as f:
                return pickle.load(f)
        except (pickle.PickleError, IOError) as e:
            print(f"Error reading stub file {stub_path}: {e}")
            return None
    return None

def save_stub(stub_path, data):
    """
    Save data to a stub file.
    
    Args:
        stub_path (str): Path to save the stub file
        data: Data to save
    """
    if stub_path is not None:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(data, f)
        except (pickle.PickleError, IOError) as e:
            print(f"Error saving stub file {stub_path}: {e}")
