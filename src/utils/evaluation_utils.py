import os

"""
These functions are used to get specific files from a directory.
Depending on your use case you would need one or the other.
"""


def get_files_dir(directory, extensions=['*']):
    ret = []
    for extension in extensions:
        if extension == '*':
            ret += [f for f in os.listdir(directory)]
            continue
        elif extension is None:
            # accepts all extensions
            extension = ''
        elif '.' not in extension:
            extension = f'.{extension}'
        ret += [f for f in os.listdir(directory) if f.lower().endswith(extension.lower())]
    return ret


def get_files_recursively(directory, extension="txt"):
    files = [
        os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(directory)
        for f in get_files_dir(directory, [extension])
    ]
    # Disconsider hidden files, such as .DS_Store in the MAC OS
    ret = [f for f in files if not os.path.basename(f).startswith('.')]
    return ret


def get_annotation_files(file_path):
    # Path can be a directory containing all files or a directory containing multiple files
    if file_path is None:
        return []
    annotation_files = []
    if os.path.isfile(file_path):
        annotation_files = [file_path]
    elif os.path.isdir(file_path):
        annotation_files = get_files_recursively(file_path)
    return sorted(annotation_files)
