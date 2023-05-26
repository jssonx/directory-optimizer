import os


def list_files(startpath):
    """List all files and folders in a directory tree."""
    if not os.path.exists(startpath):
        print(f"Path not found: {startpath}")
        return

    if startpath.endswith(os.sep):
        startpath = startpath[:-1]
    print(startpath.split(os.sep)[-1])
    try:
        list_files_rec(startpath, [False])
    except PermissionError:
        print(f"No permission to access path: {startpath}")


def list_files_rec(path, is_last_history):
    """Recursively list files and folders in a directory tree."""
    try:
        file_and_folders = sorted(os.listdir(path))
        for i, file in enumerate(file_and_folders):
            is_last = i == len(file_and_folders) - 1
            full_path = os.path.join(path, file)
            prefix = ''.join(['    ' if is_last_item else '│   ' for is_last_item in is_last_history[:-1]])
            prefix += '└── ' if is_last else '├── '
            print(prefix + file)
            if os.path.isdir(full_path):
                list_files_rec(full_path, is_last_history + [is_last])
    except PermissionError:
        print(f"No permission to access path: {path}")


start_path = './langchain'  # replace with your path
list_files(start_path)