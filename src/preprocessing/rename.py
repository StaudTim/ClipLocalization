import os


def rename_one_dir(folder_path):
    """
    Renames all files which are in a specific directory and ignores folders within this directory.
    :return:
    """
    files = os.listdir(folder_path)

    # Loop through each file and append the folder name to the beginning of the file name
    for file in files:
        # Ignore any subdirectories in the folder
        if os.path.isdir(os.path.join(folder_path, file)):
            continue

        # Get the current file name and the folder name
        current_name = os.path.join(folder_path, file)
        folder_name = os.path.basename(folder_path)

        # Generate the new file name with the folder name appended
        new_name = os.path.join(folder_path, folder_name + "_" + file)

        # Rename the file with the new name
        os.rename(current_name, new_name)


def rename_whole_dir():
    """
    Renames every file which are in the 'dataset' directory. Also search for files in directories inside
    the 'dataset' directory.
    :return:
    """
    path = '../../dataset'
    new_path = '../../dataset/clip'
    os.makedirs(new_path, exist_ok=True)

    for root, dirs, files in os.walk(path):
        if 'clip' in dirs:
            dirs.remove('clip')

        for file in files:
            if file.endswith('.png'):
                # move image to 'clip' directory
                file_path = os.path.join(root, file)
                folder_name = os.path.basename(root)
                new_file_name = folder_name + '_' + file
                new_file_path = os.path.join(new_path, new_file_name)
                os.rename(file_path, new_file_path)

                # move annotation to 'clip' directory
                annotation_path = file_path.replace('.png', '.txt')
                new_annotation_path = new_file_path.replace('.png', '.txt')
                os.rename(annotation_path, new_annotation_path)


if __name__ == "__main__":
    rename_whole_dir()