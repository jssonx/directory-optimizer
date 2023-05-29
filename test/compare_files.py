import filecmp

def compare_files(file1, file2):
    # 使用 filecmp 库的 cmp 方法进行文件对比
    comparison = filecmp.cmp(file1, file2)

    if comparison:
        print("两个文件完全一样。")
    else:
        print("两个文件不完全一样。")

# 指定你要对比的两个 HTML 文件的路径
file1 = './indeed_jobs/indeed_jobs_4.html'
file2 = './indeed_jobs/indeed_jobs_5.html'

compare_files(file1, file2)
