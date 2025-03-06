import os

def count_files_in_directory(directory):
    # 获取目录下的所有文件和文件夹
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return len(files)

def compare_folder_file_count(folder1, folder2):
    count1 = count_files_in_directory(folder1) // 4
    count2 = count_files_in_directory(folder2)
    
    # 打印每个文件夹的文件数量
    print(f"Folder 1 ({folder1}) contains {count1} files.")
    print(f"Folder 2 ({folder2}) contains {count2} files.")
    
    # 比较文件数量
    if count1 == count2:
        print("Both folders have the same number of files.")
    else:
        print("The folders have a different number of files.")

# 示例使用

for i in ["00", "01", "02", "03", "04", "05", "06", "07", "09", "08", "10"]:
    
    folder1 = os.path.join('/u/home/caoh/datasets/SemanticKITTI/dataset/labels',i)
    folder2 = os.path.join('/u/home/caoh/datasets/SemanticKITTI/dataset/pred/MonoScene',i)
    compare_folder_file_count(folder1, folder2)

    print('===============')
