"""
Script for extracting features from commits.
"""

__author__ = "Mohammed Shehab and Wahab Hamou-Lhadj"
__copyright__ = "Copyright (c) 2019 Wahab Hamou-Lhadj and Mohammed Shehab"

import gc
import json
import os
import re

import pandas as pd
from pygit2 import Repository, GIT_SORT_REVERSE, GIT_SORT_TOPOLOGICAL
from tqdm import tqdm

from core.config import repository_branch, repository_path_location, project_name, szz_results, by_tag
from core.size import load_fixed_bugs
from downloader.loader import create_nested_dir


def get_file_lines_of_code(repo, tree, dfile):
    """
    Function that counts the number of lines of code in a file.
    """

    tloc = []
    try:
        blob = repo[tree[dfile.path].id]
        tloc = blob.data.decode('utf-8')
    except Exception as _:
        return tloc
    return tloc


def get_commit(repo, head, key):
    commits = list(repo.walk(head.target, GIT_SORT_TOPOLOGICAL | GIT_SORT_REVERSE))

    for commit in commits[1:]:
        if commit.hex == key:
            return commit
    pass


def process_log(text):
    lines = text.split('\n')

    start_build = False
    block_l = ''
    block_r = ''
    blocks_l = []
    blocks_r = []
    reg_patter = "\+[0-9]+"
    right_index = -1
    left_index = -1

    for idx, line in enumerate(lines):

        if start_build and '@@' not in line:
            if line.startswith('+'):
                block_r += line + '\n'
                if right_index < 0:
                    right_index = idx
            elif line.startswith('-'):
                block_l += line + '\n'
                if left_index < 0:
                    left_index = idx
            else:
                # line = line.replace('{','-{')
                # line = line.replace('}','-}')
                block_r += line + '\n'
                block_l += line + '\n'

        if '@@' in line:
            start_build = True

            if len(block_l) > 0:
                blocks_l.append({'location': location, 'str': block_l})

            if len(block_r) > 0:
                blocks_r.append({'location': location, 'str': block_r})

            x = re.search(reg_patter, line)
            if x:
                ms = re.findall(reg_patter, line)
                location = int(ms[0])
            gc.collect()

    if len(block_l) > 0:
        blocks_l.append({'location': location, 'str': block_l})

    if len(block_r) > 0:
        blocks_r.append({'location': location, 'str': block_r})

    return blocks_l, blocks_r


def build_block(fix_commit, bug_commit, file_names, pid=0):
    repo = Repository(repository_path_location)
    if by_tag:
        branch = "refs/tags/" + repository_branch  # "refs/heads/trunk"#ARGS.branch
    else:
        branch = "refs/heads/" + repository_branch  # "refs/heads/trunk"#ARGS.branch

    head = repo.references.get(branch)

    if bug_commit == '-1':
        return

    commit_a = get_commit(repo, head, fix_commit)
    commit_b = get_commit(repo, head, bug_commit)

    export_path = './results/' + project_name
    diff = repo.diff(commit_b, commit_a)
    # Extract Size features
    tree_bug = commit_b.tree

    patches = [p for p in diff]
    ls = []
    rs = []
    for fileName in file_names:
        for patch in patches:
            # Skip binary files
            if patch.delta.is_binary or patch.delta.new_file.path != fileName:
                continue
            filename, file_extension = os.path.splitext(fileName)

            old_file = patch.delta.old_file
            text_file_lines_old = get_file_lines_of_code(repo, tree_bug, old_file)
            write_text(text_file_lines_old, export_path + '/buggy_commits_db/', bug_commit + '_' + fix_commit,
                       file_extension)

            l, r = process_log(patch.text)
            if len(l) > 1:
                ls.append({'extension': file_extension, 'segment_bug': l})
            if len(r) > 1:
                rs.append({'extension': file_extension, 'segment_fix': r})
    return {'id': bug_commit + '_' + fix_commit, 'segments_bug': ls, 'segments_fix': rs}


def test_nicad5(commit_hash, files_path, nicad_path, repo, extension='.java'):
    parents = commit_hash.parents
    from shutil import copyfile, rmtree
    from xml.dom import minidom
    import subprocess
    files = os.listdir(files_path)
    current_directory = os.getcwd()
    current_directory = current_directory.replace(':', '')
    current_directory = current_directory.replace('\\', '/').lower()
    current_directory = '/mnt/' + current_directory + nicad_path.replace('.', '')

    bash_command = 'bash -c "nicad5 blocks ' + extension.replace('.',
                                                                 '') + ' ' + current_directory + ' type3-2c-report"'

    for parent in parents:
        diff = repo.diff(commit_hash, parent)
        tree_bug = commit_hash.tree
        tree_parent = parent.tree
        stats = diff.stats
        patches = [p for p in diff]
        for patch in patches:
            # Skip binary files
            if patch.delta.is_binary or not patch.delta.new_file.path.endswith(extension):
                continue
            new_file = patch.delta.new_file
            textFile_lines_new = get_file_lines_of_code(repo, tree_bug, new_file)

            filename, file_extension = os.path.splitext('buggy_commits_db' + extension)
            # Export the change code from commit
            write_text(textFile_lines_new, nicad_path, filename, file_extension)

            # Search for similar code from the database

            for file in files:
                if file.endswith(extension):
                    src = os.path.join(files_path, file)
                    print(src)
                    copyfile(src, nicad_path + file)
                    subprocess.call(bash_command, shell=True)
                    print('finish subprocess ...')
                    # nicad_blocks-consistent-clones
                    xml_path = nicad_path.replace('nicad', 'nicad_blocks-consistent-clones')
                    if os.path.exists(xml_path + 'nicad_blocks-consistent-clones-0.50.xml'):
                        # parse an xml file by name
                        mydoc = minidom.parse(xml_path + 'nicad_blocks-consistent-clones-0.50.xml')
                        items = mydoc.getElementsByTagName('clone')
                        for item in items:
                            blocks = item.getElementsByTagName('source')
                            if blocks[0].attributes['file'].value != blocks[1].attributes['file'].value:
                                return item.attributes['similarity'].value
                                # print(item.attributes['similarity'].value)
                    # Delete report data of Nicad
                    rmtree(xml_path)
                    temp_files = os.listdir(nicad_path.replace('nicad/', ''))
                    for item in temp_files:
                        if item.endswith(".log") or item.endswith(".xml"):
                            os.remove(os.path.join(nicad_path.replace('nicad/', ''), item))
                    # Delete copied file
                    os.remove(nicad_path + file)
    return 0


def worker(dataset_path, test_size=0.50):
    # Create folders to export files.
    create_nested_dir('./results/' + project_name + '/buggy_commits_db/')
    create_nested_dir('./results/' + project_name + '/recommended_fixes/')
    create_nested_dir('./results/' + project_name + '/nicad/')
    # List of files to export
    db = []
    # Load the bug reports
    szz_re = load_fixed_bugs(szz_results + project_name, project_name + '_SZZ_results')
    # Split dataset
    features = ["commit_hex", "exp", "rexp", "sexp", "NDEV", "AGE", "NUC", "LA", "LD", "NF", "LT", "entropy", "NS",
                "ND", "commit_time", "fix", 'label']
    df = pd.read_csv(dataset_path)

    from sklearn.model_selection import train_test_split
    train_features, test_features, train_labels, test_labels = train_test_split(df[features], df['label'],
                                                                                test_size=test_size, random_state=42)
    # Filter data with defects only.
    train_features = train_features.loc[df['label'] == 1]

    train_arr = train_features['commit_hex'].values
    try:
        data = int(read_text('./results/' + project_name + '/recommended_fixes/last_process.txt')[0])
    except:
        data = 0

    if data <= 0:
        for i, row in enumerate(tqdm(train_arr)):
            hash_intro = row
            for d_row in szz_re:
                if len(d_row) > 1 and hash_intro == d_row['hash_intro']:
                    db.append(build_block(d_row['hash_fix'], d_row['hash_intro'],
                                          d_row['files'], pid=0))
                    break

        write_json(db, './results/' + project_name + '/recommended_fixes/', 'db_' + project_name)
        write_text(str(len(db)), './results/' + project_name + '/recommended_fixes/', 'last_process', 'txt')


def write_text(data, final_directory, file_name, extension):
    try:
        extensions_needs = [".java", ".txt", ".cpp", ".h"]
        if extension.lower() in extensions_needs:
            # if extension == '.java' or extension == '.txt':
            with open(final_directory + '/' + file_name + extension, 'w', encoding="utf-8") as file:
                file.write(data)
        return True
    except:
        print('Error writing to the file')
        return False


def read_text(file_name):
    current_directory = os.getcwd()
    final_path = current_directory + '\\' + file_name
    with open(file_name) as f:
        content = f.readlines()
    return content


def write_json(json_object, final_directory, file_name):
    with open(final_directory + '/' + file_name + '.json', 'w') as json_file:
        json.dump(json_object, json_file)
