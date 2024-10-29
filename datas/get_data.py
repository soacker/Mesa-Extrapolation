import os
import json

def get_data(data_path, args=None):
    if "passkey" in data_path:
        file_path = os.path.join(os.getcwd(), data_path)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise NotImplementedError
    return data