"""
Script for connecting to JIRA reports using RESTful API.
"""

__author__ = "Mohammed Shehab and Wahab Hamou-Lhadj"
__copyright__ = "Copyright (c) 2019 Wahab Hamou-Lhadj and Mohammed Shehab"

import calendar
import datetime
import json
import os

import requests

from core.config import bugs_report_path, issue_tracking_system_url, project_name, max_results, github_access_token, \
    github_project_organization_owner

state = True


def jira_connection(start_at: int) -> bool:
    """
    Function used to connect with RESTful API
    :param start_at: reports start from
    :return: if the last connection returns empty or error code
    then return false, otherwise return true to continue connection with new increment
    start_at parameter
    """
    output_location = bugs_report_path + project_name

    global state
    if state:
        create_nested_dir(output_location)
        print("Downloading Jira reports of Project {}...".format(project_name))

    req_body = {
        "jql": "project = " + project_name + " AND issuetype = Bug AND status in (Resolved, Closed) AND resolution = "
                                             "Fixed  AND created <= '2019-05-16' ORDER BY created DESC",
        "startAt": start_at,
        "maxResults": max_results,
        "fields": ["summary",
                   "status",
                   "assignee",
                   "description",
                   "name",
                   "created",
                   "updated",
                   "resolutiondate",
                   "issuetype",
                   "project",
                   "resolution",
                   "watches",
                   "priority",
                   "labels",
                   "issuelinks",
                   "assignee",
                   "components",
                   "creator",
                   "reporter",
                   "votes"]
    }
    request_url = "https://" + issue_tracking_system_url + "/rest/api/2/search"
    resp = requests.post(request_url, json=req_body)
    if resp.status_code != 200 and resp.status_code != 201:
        print("Error connection to end point {}".format(request_url))
        print("Connection code = {}".format(str(resp.status_code)))
        return False
    else:
        # print("Connection code {}".format(str(resp.status_code)))
        return parser_jira_json(resp.json(), output_location, start_at)


def parser_jira_json(response_json: object, output_location: str, start_at: int) -> bool:
    """
    Function used to serialize the JSON response  from the RESTful API
    :param response_json: the JSON data
    :param output_location: the export location
    :param start_at: the page id (split id, where each split = 1,000 rows)
    :return:
    """

    if len(response_json['issues']) > 1:
        issue_list = {}
        for issue in response_json['issues']:
            issue_list[issue['key']] = {}

            created_date = issue['fields']['created']
            issue_list[issue['key']]['creationdate'] = convert_timestamp_jira(created_date)

            res_date = issue['fields']['resolutiondate']
            issue_list[issue['key']]['resolutiondate'] = convert_timestamp_jira(res_date)

            summary = issue['fields']['summary']
            issue_list[issue['key']]['summary'] = summary
            description = issue['fields']['description']
            issue_list[issue['key']]['description'] = description

        with open(output_location + '/' + project_name + "_" + str(start_at) + '.json', 'w') as json_file:
            json.dump(issue_list, json_file)
        return True
    else:
        print("Download of Jira reports of Project {} is complete.".format(project_name))
        return False


def github_connection(start_at):
    """
   Function used to connect with RESTful API
   :param start_at: reports start from
   :return: if the last connection returns empty or error code
   then return false, otherwise return true to continue connection with new increment
   start_at parameter
   """
    output_location = bugs_report_path + project_name
    global state
    if state:
        create_nested_dir(output_location)
        print("Downloading Github reports of Project {}...".format(project_name))

    per_page_num = 100
    # We need to use authorization or github will block connection
    # please generate your access token and set it on config.py file.
    header = {"authorization": "bearer " + github_access_token}

    for page in range((start_at // per_page_num) + 1, (start_at // per_page_num) + 11):
        # each extract 10 pages(1000 records)
        request_url = "https://" + issue_tracking_system_url + "/repos/" + github_project_organization_owner + \
                      "/issues?page=" + str(page) + "&state=closed&per_page=" + str(per_page_num)
        # URL=api.github.com
        resp = requests.get(request_url, headers=header)
        if not any(resp.json()):
            return False
        # print("page: "+str(page))
        if resp.status_code != 200 and resp.status_code != 201:
            # print("Connection code {}".format(str(resp.status_code)))
            return False
        else:
            # print("Connection code {}".format(str(resp.status_code)))
            if not (parser_github_json(resp.json(), output_location,
                                       start_at + (page - 1 - start_at // per_page_num) * 100)):
                return False
    return True


def parser_github_json(response_json: object, output_location: str, start_at: int):
    """
      Function used to serialize the JSON response  from the RESTful API
      :param response_json: the JSON data
      :param output_location: the export location
      :param start_at: the page id (split id, where each split = 1,000 rows)
      :return:
  """
    # The init value for main dictionary
    issue_list = {}
    for i in response_json:
        issue_list[i['number']] = {}
        issue_list[i['number']]['resolutiondate'] = convert_timestamp_github(i['closed_at'])
        issue_list[i['number']]['creationdate'] = convert_timestamp_github(i['created_at'])
        issue_list[i['number']]['summary'] = i['title']
        issue_list[i['number']]['description'] = i['body']

    if len(issue_list) > 1:
        with open(output_location + '/' + project_name + "_" + str(start_at) + '.json', 'w') as json_file:
            json.dump(issue_list, json_file)
        # print("total : {}".format(str(len(issue_list))))
        return True
    else:
        print("Download of Github reports of Project {} is complete.".format(project_name))
        return False


def convert_timestamp_jira(string_time: str) -> int:
    """
    Function used to convert formatted time from data time to
    int as unix timestamp unit
    :param string_time:
    :return: int unix timestamp
    """

    if string_time:
        time_utc = datetime.datetime.strptime(string_time, '%Y-%m-%dT%H:%M:%S.%f%z')
        return int(calendar.timegm(time_utc.utctimetuple()))
    else:
        return None


def convert_timestamp_github(t):
    if t:
        time_utc = datetime.datetime.strptime(t, '%Y-%m-%dT%H:%M:%SZ')
        return int(calendar.timegm(time_utc.utctimetuple()))
    else:
        return t


def create_nested_dir(nested_path: str) -> None:
    """
    Function used to create folder inside folder
    it helps developers create specific location
    for their results
    :type nested_path: str
    """

    folder_output = ''
    global state
    state = True
    for index, dir in enumerate(nested_path.split('/')):
        if '.' not in dir:
            folder_output += dir + '/'
            try:
                current_directory = os.getcwd()
                final_directory = os.path.join(current_directory, folder_output)
                # Create target Directory
                os.mkdir(final_directory)
                # print("Folder ", final_directory, " is created.")
            except FileExistsError:
                # print("Folder ", final_directory, " already exists.")
                state = False
