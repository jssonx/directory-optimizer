import os

def collect_files(path):
    python_files = []
    
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))

    return python_files

def write_summary_file(python_files, output_filename):
    with open(output_filename, "w", encoding="utf-8") as summary_file:
        for file in python_files:
            summary_file.write("#" * 80 + "\n")
            summary_file.write(f"# FILE: {file}\n")
            summary_file.write("#" * 80 + "\n\n")
            
            with open(file, "r", encoding="utf-8") as source_file:
                summary_file.write(source_file.read())
                summary_file.write("\n\n")

if __name__ == "__main__":
    path = "../LangBase"
    output_filename = "summary.py"
    python_files = collect_files(path)
    write_summary_file(python_files, output_filename)