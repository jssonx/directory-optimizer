import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
folder_path = "./indeed_htmls"  # 文件夹路径

# 检查文件夹是否存在
if os.path.exists(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # 检查路径是否是文件
        if os.path.isfile(file_path):
            # 删除文件
            os.remove(file_path)
            print(f"已删除文件: {file_path}")
else:
    print(f"文件夹不存在: {folder_path}")