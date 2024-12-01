import os


def test_directories_exist_and_not_empty(fins_dataset_path):
    directories_to_check = ["1056", "random"]

    for directory in directories_to_check:
        dir_path = os.path.join(fins_dataset_path, directory)

        # check if exists
        assert os.path.exists(dir_path), f"Directory '{directory}' does not exist in '{fins_dataset_path}'."

        # check if empty
        assert os.listdir(dir_path), f"Directory '{directory}' in '{fins_dataset_path}' is empty."
