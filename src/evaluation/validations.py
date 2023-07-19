import os

"""
These functions are used to check if the provided format is valid.
"""


def is_text(file_path):
    # Text file with annotations can be a .txt file or have no extension
    return os.path.splitext(file_path)[-1].lower() in ['.txt', '']


def is_absolute_text_format(file_path, num_blocks=[6, 5], blocks_abs_values=[4]):
    if not is_text(file_path):
        return False
    if not is_empty_file(file_path):
        return all_lines_have_blocks(file_path,
                                     num_blocks=num_blocks) and all_blocks_have_absolute_values(
            file_path, blocks_abs_values=blocks_abs_values)
    return True


def is_relative_text_format(file_path, num_blocks=[6, 5], blocks_rel_values=[4]):
    if not is_text(file_path):
        return False
    if not is_empty_file(file_path):
        return all_lines_have_blocks(file_path,
                                     num_blocks=num_blocks) and all_blocks_have_relative_values(
            file_path, blocks_rel_values=blocks_rel_values)
    return True


def all_lines_have_blocks(file_path, num_blocks=[]):
    with open(file_path, 'r+') as f:
        for line in f:
            line = line.replace('\n', '').strip()
            if line == '':
                continue
            passed = False
            for block in num_blocks:
                if len(line.split(' ')) == block:
                    passed = True
            if passed is False:
                return False
    return True


def all_blocks_have_absolute_values(file_path, blocks_abs_values=[]):
    with open(file_path, 'r+') as f:
        for line in f:
            line = line.replace('\n', '').strip()
            if line == '':
                continue
            passed = False
            splitted = line.split(' ')
            for block in blocks_abs_values:
                if len(splitted) < block:
                    return False
                try:
                    if float(splitted[block]) == int(float(splitted[block])):
                        passed = True
                except:
                    passed = False
            if passed is False:
                return False
    return True


def all_blocks_have_relative_values(file_path, blocks_rel_values=[]):
    with open(file_path, 'r+') as f:
        for line in f:
            line = line.replace('\n', '').strip()
            if line == '':
                continue
            passed = False
            splitted = line.split(' ')
            for block in blocks_rel_values:
                if len(splitted) < block:
                    return False
                try:
                    float(splitted[block])
                    passed = True
                except:
                    passed = False
            if passed is False:
                return False
    return True


def is_empty_file(file_path):
    # An empty file is considered a file with empty lines or spaces
    with open(file_path, 'r+') as f:
        for line in f:
            if line.strip() != '':
                return False
    return True
