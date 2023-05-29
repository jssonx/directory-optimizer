import pkg_resources
import os

# 将当前环境中的所有包及其版本保存为一个字典
installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

# 读取原有的requirements.txt文件
with open('requirements.txt', 'r') as f:
    lines = f.readlines()

# 创建新的requirements.txt文件
with open('new_requirements.txt', 'w') as f:
    for line in lines:
        line = line.strip()
        # 如果这个包在当前环境中安装了，就添加版本号
        if line in installed_packages:
            f.write(line + '==' + installed_packages[line] + '\n')
        else:
            f.write(line + '\n')