"""
Configuration variables
"""

__author__ = "Wahab Hamou-Lhadj and Mohammed Shehab"
__copyright__ = "Copyright (c) 2019 Wahab Hamou-Lhadj and Mohammed Shehab"

# This variable is used to put the endpoint restful api url
issue_tracking_system_url = "issues.apache.org/jira"
# e.g. (issue_tracking_system_url = "api.github.com", issue_tracking_system_url = "issues.apache.org/jira")

# This variable is used to set software name
# project_name = "kafka"
project_name = "helix"
# This variable is used to set software repository location in your hard drive
repository_path_location = "D:/helix"  # "E:/CeleverDataset/androidannotations"

# The location for issue reports
bugs_report_path = "./issues/"

# The location for the SZZ algorithm
szz_results = "./results/"

# This variable is used to set the bug pattern inside issue reports. (e.g. "[a-z]+[-\t]+[0-9]+" is for Jira apache
# systems,"\#[0-9]+" for Github and "[a-z]+[-\t]+[0-9]+" for Jira).
bug_pattern = "[a-z]+[-\t]+[0-9]+"

# This variable is used to set software repository branch
repository_branch = "master"
# This flag used to retrieve commits of specific release(False will return all commits of branch)
by_tag = False

# Generalk settings
enable_parallel = True
max_results = 1_000
clone_threshold = "0.50"
enable_nicad = False
# .java for java programming language and .cpp for c++ programming language.
programming_extension = ".cpp"
# Github settings
# Please add your own github access token. For more information see https://developer.github.com/v3/#authentication
github_access_token = ""
# Must be in following format (organization name/project name) e.g. google/guava for
# project https://github.com/google/guava
github_project_organization_owner = "PX4/Firmware"
