import urllib.request
import os

DATASET_LINKS = {
    "rcv1.binary": {
        "train": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2",
        "test": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_test.binary.bz2"
    },
    "w8a": {
        "train": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a",
        "test": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a.t"
    },
    "ijcnn1": {
        "train": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.tr.bz2",
        "test": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.t.bz2"
    }
}

def download_libsvm_dataset(dataset_name):
    """
    Downloads a dataset from the LIBSVM website and saves it to a local directory.
    
    Args:
        dataset_name (str): The name of the dataset to download. Must be one of 'rcv1.binary', 'w8a', or 'ijcnn1'.
    """
    data_dir = "./data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    print(f"Downloading {dataset_name} dataset...")
    
    for data_type, url in DATASET_LINKS[dataset_name].items():
        # Extract the file extension from the URL (if any)
        file_ext = os.path.splitext(url)[-1]
        
        # Construct the file path for the downloaded file, preserving the file extension
        file_path = os.path.join(data_dir, f"{dataset_name}_{data_type}{file_ext}")
        
        urllib.request.urlretrieve(url, file_path)
    
    print(f"{dataset_name} dataset downloaded successfully!")

def get_dataset_paths(dataset_name):
    """
    Returns the file paths of the train and test data for the specified dataset.

    Args:
    - dataset_name: str, name of the dataset

    Returns:
    - train_path: str, file path of the train data
    - test_path: str, file path of the test data
    """

    data_dir = "./data"

    if dataset_name not in DATASET_LINKS:
        raise ValueError(f"Dataset {dataset_name} not found.")

    paths = []

    for data_type, url in DATASET_LINKS[dataset_name].items():
        file_ext = os.path.splitext(url)[-1]
        file_path = os.path.join(data_dir, f"{dataset_name}_{data_type}{file_ext}")
        paths.append(file_path)

    return paths


if __name__ == "__main__":
    # download_libsvm_dataset("w8a")
    # download_libsvm_dataset("ijcnn1")
    # download_libsvm_dataset("rcv1.binary")
    train_path, test_path = get_dataset_paths('rcv1.binary')
    train_path, test_path = get_dataset_paths('w8a')
    train_path, test_path = get_dataset_paths('ijcnn1')