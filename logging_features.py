"""
Script for extracting commit features that characterize the logging of the commits.
"""

from pygit2 import Repository, GIT_SORT_TOPOLOGICAL, GIT_SORT_REVERSE
from tqdm import tqdm

from config import repository_branch, code_repository_path, by_tag


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


def process_code(code):
    log_add = 0
    log_del = 0
    log_max = 0
    log_min = 1000

    log_types = ['error', 'info', 'warn', 'debug', 'verbose', 'fatal', 'trace', 'severe', 'config']
    counts_type = {'error_add': 0, 'error_del': 0,
                   'info_add': 0, 'info_del': 0,
                   'warn_add': 0, 'warn_del': 0,
                   'debug_add': 0, 'debug_del': 0,
                   'verbose_add': 0, 'verbose_del': 0,
                   'fatal_add': 0, 'fatal_del': 0,
                   'trace_add': 0, 'trace_del': 0,
                   'severe_add': 0, 'severe_del': 0,
                   'config_add': 0, 'config_del': 0
                   }
    lines = code.split('\n')
    for l in lines:
        # If the code line contains log
        if '*' not in l and 'log' in l:

            code_type = l[0]
            if code_type == '+':
                log_add += 1
            elif code_type == '-':
                log_del += 1

            for t in log_types:
                if t in l and code_type == '+':
                    try:
                        val = l.split('(', 1)[1].split(')')[0]
                        if log_max < len(val):
                            log_max = len(val)
                        if log_min > len(val):
                            log_min = len(val)
                    except:
                        print(f'No Parameter on {l}', end='\n')
                    counts_type[t + "_add"] += 1
                elif t in l and code_type == '-':
                    try:
                        val = l.split('(', 1)[1].split(')')[0]
                        if log_max < len(val):
                            log_max = len(val)
                        if log_min > len(val):
                            log_min = len(val)
                    except:
                        print(f'No Parameter on {l}', end='\n')
                    counts_type[t + "_del"] += 1

    log_err_add = counts_type['error_add']
    log_err_del = counts_type['error_del']
    log_inf_add = counts_type['info_add']
    log_inf_del = counts_type['info_del']
    log_war_add = counts_type['warn_add']
    log_war_del = counts_type['warn_del']
    log_deb_add = counts_type['debug_add']
    log_deb_del = counts_type['debug_del']
    log_ver_add = counts_type['verbose_add']
    log_ver_del = counts_type['verbose_del']
    log_fat_add = counts_type['fatal_add']
    log_fat_del = counts_type['fatal_del']
    log_trc_add = counts_type['trace_add']
    log_trc_del = counts_type['trace_del']
    log_ser_add = counts_type['severe_add']
    log_ser_del = counts_type['severe_del']
    log_con_add = counts_type['config_add']
    log_con_del = counts_type['config_del']

    return log_add, log_del, log_err_add, log_err_del, log_inf_add, log_inf_del, \
           log_war_add, log_war_del, log_deb_add, log_deb_del, log_ver_add, log_ver_del, \
           log_fat_add, log_fat_del, log_trc_add, log_trc_del, log_ser_add, log_ser_del, \
           log_con_add, log_con_del, log_max, log_min


def get_logging():
    """
   Extract the purpose features for each commit.
   """
    if by_tag:
        branch = "refs/tags/" + repository_branch  # "refs/heads/trunk"#ARGS.branch
    else:
        branch = "refs/heads/" + repository_branch  # "refs/heads/trunk"#ARGS.branch
    repo = Repository(code_repository_path)
    head = repo.references.get(branch)

    commits = list(
        repo.walk(head.target, GIT_SORT_TOPOLOGICAL | GIT_SORT_REVERSE))

    features = [{} for c in range(len(commits))]
    for i, commit in enumerate(tqdm(commits[1:])):
        diff = repo.diff(commits[i], commit)

        patches = [p for p in diff]
        for patch in patches:
            # Skip binary files
            if patch.delta.is_binary:
                continue
            new_file = patch.delta.new_file
            text_file_lines_new = get_file_lines_of_code(repo, commit.tree, new_file)
            extension = new_file.raw_path.decode('utf-8')
            if len(text_file_lines_new) > 0 and '.java' in extension:
                text = patch.text
                log_add, log_del, log_err_add, log_err_del, log_inf_add, log_inf_del, \
                log_war_add, log_war_del, log_deb_add, log_deb_del, log_ver_add, log_ver_del, \
                log_fat_add, log_fat_del, log_trc_add, log_trc_del, log_ser_add, log_ser_del, \
                log_con_add, log_con_del, log_max, log_Min = process_code(text)
                # print(extension)
                # print(text_file_lines_new)
        if log_Min == 1000:
            log_Min = 0
        feature_dict = {"commit_hex": str(commit.hex),
                        "log_add": log_add,
                        "log_del": log_del,
                        "log_err_add": log_err_add,
                        "log_err_del": log_err_del,
                        "log_inf_add": log_inf_add,
                        "log_inf_del": log_inf_del,
                        "log_war_add": log_war_add,
                        "log_war_del": log_war_del,
                        "log_deb_add": log_deb_add,
                        "log_deb_del": log_deb_del,
                        "log_ver_add": log_ver_add,
                        "log_ver_del": log_ver_del,
                        "log_fat_add": log_fat_add,
                        "log_fat_del": log_fat_del,
                        "log_trc_add": log_trc_add,
                        "log_trc_del": log_trc_del,
                        "log_ser_add": log_ser_add,
                        "log_ser_del": log_ser_del,
                        "log_con_add": log_con_add,
                        "log_con_del": log_con_del,
                        "log_max": log_max,
                        "log_Min": log_Min
                        }
        features[i].update(feature_dict)
    return features
