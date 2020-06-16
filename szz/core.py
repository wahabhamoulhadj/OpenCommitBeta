"""
Script implementing the SZZ algorithm using parallel processing.
"""

__author__ = "Mohammed Shehab and Wahab Hamou-Lhadj"
__copyright__ = "Copyright (c) 2019 Wahab Hamou-Lhadj and Mohammed Shehab"

import json
import os
import re
import time
from multiprocessing import Process, Manager, cpu_count

from dulwich.diff_tree import tree_changes
from dulwich.repo import Repo
from tqdm import tqdm

from core.config import repository_path_location, project_name, bugs_report_path, szz_results, enable_parallel, \
    bug_pattern
from downloader.loader import create_nested_dir

repo = None
issue_list = None

bug_regular_expressions = ['bug[# \t]*[0-9]+', 'pr[# \t]*[0-9]+', 'show\_bug\.cgi\?id=[0-9]+', '\[[0-9]+\]',
                           'fix(e[ds])?|bugs?|defects?|patch']


def init_szz() -> None:
    """
    Initialize SZZ by loading the repository to the global repository.
    """

    global repo
    repo = Repo(repository_path_location)


def load_bugs_reports() -> None:
    """
    Load the stored issue reports to the global issue list dictionary.
    """

    full_report_path = bugs_report_path + project_name
    global issue_list
    issue_list = {}
    for file_name in os.listdir(full_report_path):
        with open(full_report_path + '/' + file_name) as f:
            temp_data = json.loads(f.read())
            issue_list.update(temp_data)
            x = len(issue_list)


def link_commits_with_report() -> None:
    """
    The first step in SZZ is to find the list of commits linked to issue reports
    using the issue identifier.
    """

    load_bugs_reports()
    global issue_list
    global repo
    global bug_regular_expressions

    # print("=============== Start linking commits with issue reports ===============")

    r = repo
    prev = None
    walker = r.get_graph_walker()
    cset = walker.next()

    count = len(issue_list)
    count_found = 0
    count_not_found = 0
    json_out = []

    # extensions_needs = ('.java','.cpp','.c','.py','.cs','.asp','.aspx','.shtml','.php')
    extensions_needs = ('.txt', '.xml', '.sh')
    if count <= 0:
        raise Exception(f"- Please check folder {bugs_report_path + project_name}, no issue reports found.")
    # start navigation the git log
    while cset is not None:
        # if count_bugs_found % 10 == 0:
        print("Total number of bug reports = " + str(count) + " % " + (
            str(round(count_found / count, 2))) + " Number of bug reports linked = " + str(
            count_found) + " Not linked = " + str(
            count_not_found), end='\n')
        # get the commit by its hash
        commit = r.get_object(cset)
        # convert commit log message from bytes to string
        log_message = commit.message.decode("utf-8").lower()
        committer = commit.committer.decode("utf-8").lower()

        if prev is None:
            prev = commit.tree
            cset = walker.next()
            continue

        delta = tree_changes(r, prev, commit.tree)
        changes = 0
        for x in delta:
            if x.new.path:
                if not x.new.path.decode("utf-8").lower().endswith(extensions_needs):
                    changes += 1

        for reg in bug_regular_expressions:
            x = re.search(reg, log_message)
            if x:
                bug_id = re.findall(bug_pattern, log_message)
                if bug_id:
                    found = False
                    for b_id in bug_id:
                        bug_identifier = b_id.upper().replace('#', "")
                        if bug_identifier in issue_list:
                            info = issue_list[bug_identifier]
                            json_dict = {"commit_hash": cset.decode("utf-8"),
                                         "log_message": log_message,
                                         "bug_patter": bug_identifier,
                                         # "changes":changes,
                                         "creationdate": info['creationdate'],
                                         "resolutiondate": info['resolutiondate'],
                                         "description": info['description'],
                                         "summary": info['summary'],
                                         "confidence": get_confidence(log_message, info['description'], committer,
                                                                      changes)}
                            json_out.append(json_dict)
                            count_found += 1
                            found = True
                            break
                    if not found:
                        count_not_found += 1

        prev = commit.tree
        cset = walker.next()

    # End navigation
    print("Total number of bug reports = " + str(count) + " % " + (
        str(round(count_found / count, 2))) + " Number of bug reports linked = " + str(
        count_found) + " Not linked = " + str(
        count_not_found), end='\n')

    #  print("=============== Exporting results ===============")

    create_nested_dir(szz_results + project_name)
    write_json(json_out, szz_results + project_name, project_name + '_commits_linked_reports')

    # print("=============== Finish linking ===============")


def run_szz(export_file_name: str) -> None:
    """
    This function is used to start SZZ.
    It takes the path to export the file results.
    :param export_file_name: location on hard drive where to extract JSON
    """

    global repo
    export_location = szz_results + project_name
    linked_commits = szz_results + project_name + '/' + project_name + "_commits_linked_reports.json"
    json_reports = read_json(linked_commits)

    manager = Manager()
    res = manager.dict()

    # Test single process

    if not enable_parallel:
        start_time = time.time()
        parse_szz(0, json_reports, res, 0)
        end_time = time.time()
    else:
        # Check how many processes that could be spawned
        cpus = cpu_count()
        # Divide the commits equally between the processes.
        quote, remainder = divmod(len(json_reports), cpus)

        # Note: Repo is not work with multi- process, so we need to init it in code of function itself
        # in other words, each single process will create its own repo object.
        processes = [
            Process(
                target=parse_szz,
                args=(i, json_reports, res, i * quote + min(i, remainder),
                      (i + 1) * quote + min(i + 1, remainder))) for i in range(cpus)
        ]

        for process in processes:
            process.start()

        start_time = time.time()
        for process in processes:
            process.join()
        end_time = time.time()

    # Assemble the results
    final_results = []
    for _, feat in res.items():
        final_results.extend(feat)
    final_results = list(reversed(final_results))

    print("Saving SZZ results.")
    write_json(final_results, export_location, export_file_name)


def parse_szz(pid: int, json_reports: object, res: object, start: int, stop: int = -1) -> None:
    """
    This function splits SZZ processing into tasks, excuted by parrallel processes
    :param pid: process ID
    :param json_reports: shared read JSON data, each process has a copy
    :param res: Shared memory between sub processes
    :param start: start segment location
    :param stop: end segment location
    """
    start = start - 1 if (start > 0) else start
    json_reports = json_reports[start:stop] if (stop != -1) else json_reports[start:]

    fix_intro = [{} for c in range(len(json_reports))]

    for i, json_report in enumerate(tqdm(json_reports[1:], position=pid)):
        r = szz(str.encode(json_report['commit_hash']), json_report)
        if r is not None and len(r) > 0:
            fix_intro[i].update(r)

    res[pid] = fix_intro


def szz(issue_hash: str, json_report: object) -> dict:
    """
    This function is executed for a single commit, the input it the fix commit.
    The function searches for the commit that introduced a bug/defect
    :param issue_hash: fix commit ID
    :param json_report: json for linked commits with issue reports
    :return: dict of fix commit, bug commit and the intercept files
    """

    repo = Repo(repository_path_location)
    report_issue = json_report['creationdate']
    r = repo
    cset = issue_hash
    extensions_needs = ('.txt', '.xml', '.sh')
    commit_changes = []

    temp_parents = []
    go_branch = False
    # Start navigation the git log
    while cset is not None:
        # get the commit by its hash
        commit = r.get_object(cset)

        if len(commit.parents) <= 0:
            if len(temp_parents) > 0:
                cset = temp_parents.pop(0)
                commit = r.get_object(cset)
                go_branch = True
            else:
                return {}
        if len(commit.parents) > 1:
            if not go_branch:
                for p in commit.parents[1:]:
                    temp_parents.append(p)
        prev = r.get_object(commit.parents[0])

        if report_issue > prev.commit_time:
            delta = tree_changes(r, prev.tree, commit.tree)
            changes = []
            if len(commit_changes) == 0:
                add_flag = True
            else:
                add_flag = False
            for x in delta:
                if x.new.path:
                    if not x.new.path.decode("utf-8").lower().endswith(extensions_needs):
                        if add_flag:
                            commit_changes.append(x.new.path.decode("utf-8"))
                        else:
                            changes.append(x.new.path.decode("utf-8"))

            set_intersection = list(set(commit_changes) & set(changes))

            if len(set_intersection) > 0:
                intro_report = {"bug_patter": json_report['bug_patter'],
                                "hash_fix": issue_hash.decode("utf-8"),
                                "hash_intro": cset.decode("utf-8"),
                                "files": set_intersection}
                break

        cset = prev.id

    return intro_report


def write_json(json_object: object, final_directory: str, file_name: str, extension: str = '.json') -> None:
    """
    Function used to export the JSON object to the hard drive
    :param json_object: json object
    :param final_directory: export location
    :param file_name: json file name
    :param extension: the extension of file must start with "." (default is json)
    """
    with open(final_directory + '/' + file_name + extension, 'w') as json_file:
        json.dump(json_object, json_file)


def read_json(file_name: str) -> object:
    """
    Function to load JSON  file
    :param file_name: json file name
    :return: json object as dictionary
    """

    # current_directory = os.getcwd()
    # final_path = current_directory + '\\' + file_name
    with open(file_name) as f:
        json_fixed = json.loads(f.read())
    return json_fixed


def get_confidence(log_message: str, bug_description: str, committer: str, files: int) -> int:
    """
    log_message: log message from repository
    bug_description: short description from bug report
    committer: the name of person who committed the code (transaction)

    return int value:
        1-The bug has been resolved as Fixed
        2-The short description from bug report contains a log message
        3-The author assigned appears in the bug description
        4-One or more code files are effected
    """

    # because on get bugs report we retrieve only the fixed ones
    conf = 1

    if bug_description is None:
        return 1

    if log_message.find(bug_description) > 0:
        conf += 1

    if committer in bug_description:
        conf += 1

    if files > 1:
        conf += 1

    return conf
