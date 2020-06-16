"""
Main script
"""

__author__ = "Wahab Hamou-Lhadj and Mohammed Shehab"
__copyright__ = "Copyright (c) 2019 Wahab Hamou-Lhadj and Mohammed Shehab"

import os

from core.code_similarity import worker as block_builder
from core.config import *
from core.extract_features import extract_features
from core.machine_learning import TrainModel as model
from downloader.loader import jira_connection
from szz.core import init_szz, run_szz, link_commits_with_report


def main():
    print("Start commit assistant...")

    # Step 0: Download the Github repository to your hard drive
    # Access Github and clone the repository to the hard drive

    # Step 1: Download issue reports from Jira if not already downloaded
    if not os.path.exists(bugs_report_path + project_name):
        start_at = 1
        is_running = True

        while is_running:
            # For jira bug tracking system
            is_running = jira_connection(start_at)
            # For github bug tracking system
            # is_running = github_connection(start_at)
            start_at += max_results

    else:
        absolute_file_path = os.path.abspath(bugs_report_path + project_name)
        print("Project {} reports already exist in {}. if you want to download them again, delete folder {}."
              .format(project_name, absolute_file_path, project_name))
    # End Step 1

    # Step 2: Execute SZZ
    init_szz()
    linked_commits_file = szz_results + project_name + "/" + project_name + '_commits_linked_reports.json'
    if not os.path.exists(linked_commits_file):
        link_commits_with_report()
    export_file_name = project_name + "_SZZ_results"
    export_path_file_name = szz_results + project_name + '/' + export_file_name + ".json"
    if not os.path.exists(export_path_file_name):
        print('Run SZZ algorithm...')
        run_szz(export_file_name)
    else:
        print("Skip SZZ, the SZZ results are in {}".format(export_path_file_name))

    # Step 3: Extract the features for ML and labeled data using SZZ
    print("Extract features for ML and labeled datea using SZZ")
    extract_features()

    if enable_nicad:
        # Step 4: Build databse of defect-introducing commits for nicad tool process
        print("Build database of defect-introducing commits")
        database = szz_results + project_name + '/buggy_commits_db/'
        if not os.path.exists(database):
            # Build the database of defect-introducing commits from the training data.
            # The training data size is 70% from the total data.
            block_builder(szz_results + project_name + '_sample_features.csv', test_size=0.30)

    # Step 5: Train and test the Random Forest model and NICAD
    ml_worker = model()
    ml_worker.train_model(dataset_path=szz_results + project_name + '_sample_features.csv',
                          model_name='rf',
                          data_method='down',
                          test_size=0.30)


def check_configuration():
    if issue_tracking_system_url == "":
        print("Error: Please set the issue tracking system url.")
        exit(1)

    if project_name == "":
        print("Error: Please set the project name.")
        exit(1)

    if repository_path_location == "":
        print("Error: Please set the repository path to your hard drive.")
        exit(1)


if __name__ == "__main__":
    check_configuration()
    main()
