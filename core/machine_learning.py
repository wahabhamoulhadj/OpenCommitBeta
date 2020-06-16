"""
Script for applying the machine learning algorithms.
"""

__author__ = "Wahab Hamou-Lhadj and Mohammed Shehab"
__copyright__ = "Copyright (c) 2019 Wahab Hamou-Lhadj and Mohammed Shehab"

import gc

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from pygit2 import Repository, GIT_SORT_REVERSE, GIT_SORT_TOPOLOGICAL
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.utils import resample

from core.code_similarity import write_text
from core.config import repository_path_location, repository_branch, project_name, enable_nicad, \
    by_tag, programming_extension
from downloader.loader import create_nested_dir


class TrainModel:
    # The main features that used for train model
    features = ["exp", "rexp", "sexp", "NDEV", "AGE", "NUC", "LA", "LD", "NF", "LT", "entropy", "NS", "ND",
                "commit_time", "fix"]

    def __init__(self):
        print("+ Start training the model.")
        self.repo_path = repository_path_location
        self.branch = repository_branch
        self.project_name = project_name

    def get_commit(self, repo, head, key):
        commits = list(repo.walk(head.target, GIT_SORT_TOPOLOGICAL | GIT_SORT_REVERSE))

        for commit in commits[1:]:
            if commit.hex == key:
                return commit
        pass

    def get_file_lines_of_code(self, repo, tree, dfile):
        tloc = []
        try:
            blob = repo[tree[dfile.path].id]

            tloc = blob.data.decode('utf-8')
            # tloc = tloc.split('\n')
        except Exception as _:
            return tloc
        return tloc

    # def get_similarity(self, commit_hash, files_path, nicad_path, repo, extension='.java'):
    #     parents = commit_hash.parents
    #     from shutil import copyfile, rmtree
    #     # from xml.dom import minidom
    #     import subprocess
    #     files = os.listdir(files_path)
    #     current_directory = os.getcwd()
    #     current_directory = current_directory.replace(':', '')
    #     current_directory = current_directory.replace('\\', '/').lower()
    #     current_directory = '/mnt/' + current_directory + nicad_path.replace('.', '')
    #     current_directory = current_directory.replace(' ','\ ')
    #     bash_command = 'bash -c "nicad5 functions ' + extension.replace('.',
    #                                                                  '') + ' ' + current_directory + ' type3-2c-report"'
    #     for parent in parents:
    #         diff = repo.diff(commit_hash, parent)
    #         tree_bug = commit_hash.tree
    #         tree_parent = parent.tree
    #         stats = diff.stats
    #         patches = [p for p in diff]
    #         for patch in patches:
    #             # Skip binary files
    #             if patch.delta.is_binary or not patch.delta.new_file.path.endswith(extension):
    #                 continue
    #             new_file = patch.delta.new_file
    #             text_file_lines_new = self.get_file_lines_of_code(repo, tree_bug, new_file)
    #             if text_file_lines_new is None or len(text_file_lines_new) <= 0:
    #                 continue
    #             filename, file_extension = os.path.splitext('buggy' + extension)
    #             # Export the change code from commit
    #             write_state = write_text(text_file_lines_new, nicad_path, filename, file_extension)
    #             # check if the code file is written without error
    #             if not write_state:
    #                 # if there is error in format skip this file
    #                 continue
    #             # try to check file size
    #             state_info = os.stat(nicad_path + filename + file_extension)
    #             # if file size is greater than 1 MB, then mostly it contains defects
    #             f_size = state_info.st_size / 1024
    #             if f_size > 1000:
    #                 os.remove(nicad_path + filename + file_extension)
    #                 return 1
    #             # Now we need to search for similar code from
    #             #
    #             #
    #             # uilt database
    #             found = False
    #             for file in files:
    #                 # compare the files with same type
    #                 if file.endswith(extension):
    #                     # Copy file from defect database to the test folder
    #                     src = os.path.join(files_path, file)
    #                     copyfile(src, nicad_path + file)
    #                     subprocess.call(bash_command, shell=True)
    #                     # nicad_blocks-consistent-clones or nicad_functions-consistent-clones
    #                     xml_path = nicad_path.replace('nicad', 'nicad_functions-consistent-clones')
    #                     # read the results of nicad
    #                     if os.path.exists(
    #                             xml_path + 'nicad_functions-consistent-clones-' + clone_threshold + '-classes.xml'):
    #                         # parse an xml file by name
    #                         try:
    #                             mydoc = minidom.parse(
    #                                 xml_path + 'nicad_functions-consistent-clones-' + clone_threshold + '-classes.xml')
    #                             items = mydoc.getElementsByTagName('clones')
    #                             for item in items:
    #                                 # similarity = item.attributes['similarity'].value
    #
    #                                 blocks = item.getElementsByTagName('source')
    #                                 if len(blocks) > 0:
    #                                     if blocks[0].attributes['file'].value != blocks[1].attributes['file'].value:
    #                                         found = True
    #                                         break
    #                                         # return 1 #item.attributes['similarity'].value
    #                                         # print(item.attributes['similarity'].value)
    #                         except Exception as inst:
    #                             print('Error: Failed to parse XML exported file.')
    #                     try:
    #                         # Now delete report data of Nicad
    #                         rmtree(xml_path)
    #                     except:
    #                         print('Error: no Results path.')
    #                     temp_files = os.listdir(nicad_path.replace('nicad/', ''))
    #                     for item in temp_files:
    #                         if item.endswith(".log") or item.endswith(".xml"):
    #                             os.remove(os.path.join(nicad_path.replace('nicad/', ''), item))
    #                     # Now delete copied file
    #                     os.remove(nicad_path + file)
    #                     if found:
    #                         return 1
    #                         # pass
    #     return 0

    def export_results(self, df, predictions, labels, test_size=0.5, random_state=42):
        features = ["commit_hex", "exp", "rexp", "sexp", "NDEV", "AGE", "NUC", "LA", "LD", "NF", "LT", "entropy", "NS",
                    "ND", "commit_time", "fix"]
        train_features, test_features, train_labels, test_labels = train_test_split(df[features], df['label'],
                                                                                    test_size=test_size,
                                                                                    random_state=random_state)

        hex_code = test_features['commit_hex'].values
        data = list(zip(hex_code, labels, predictions))
        similarity = []
        # Nicad section
        repo = Repository(self.repo_path)
        if by_tag:
            head = repo.references.get("refs/tags/" + self.branch)  # "refs/heads/trunk"#ARGS.branch
        else:
            head = repo.references.get('refs/heads/' + self.branch)  # "refs/heads/trunk"#ARGS.branch
        # head = repo.references.get('refs/heads/' + self.branch)

        create_nested_dir('./results/' + self.project_name + '/nicad/')
        le = len(data)
        for i, d in enumerate(data):
            if d[2] == 1:
                commit = self.get_commit(repo, head, d[0])
                similarity.append(self.get_similarity(commit, './results/' + self.project_name + '/buggy_commits_db/',
                                                      './results/' + self.project_name + '/nicad/', repo,
                                                      programming_extension))
                # print('{} / {}'.format(i, le))
                gc.collect()
            else:
                similarity.append(d[2])
                # write_text(str(i) + '\r\n', './results/' + self.project_name, str('i'), '.txt')
                # print('similarity = {}, prediction = 1, label = {}'.format(str(similarity),str(d[1])))

        data = list(zip(hex_code, labels, predictions, similarity))

        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for d in data:
            if d[1] == d[3] and d[1] == 1:
                tp += 1
            elif d[1] == d[3] and d[1] == 0:
                tn += 1
            elif d[1] == 0 and d[3] == 1:
                fp += 1
            elif d[1] == 1 and d[3] == 0:
                fn += 1

        report_ml = "TP = {}, TN = {}, FP = {}, FN = {}".format(tp, tn, fp, fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        report_ml += '\r\n' + 'Precision = {} , Recall = {}'.format(precision, recall)
        write_text(report_ml, './results/' + self.project_name, project_name + 'nicad_results', '.txt')

        df = pd.DataFrame(data, columns=['hex_code', 'labels', 'predictions', 'similarity'])
        df.to_csv('./' + self.project_name + '_results.csv', header=True, index=False)
        print('Nicad completed.')

    def random_forest(self, df, test_size=0.5, print_report=False):

        train_features, test_features, train_labels, test_labels = train_test_split(df[self.features], df['label'],
                                                                                    test_size=test_size,
                                                                                    random_state=42)

        rf = RandomForestClassifier(random_state=2018, n_jobs=-1, n_estimators=1000, max_depth=7)
        rf.fit(train_features, train_labels)
        print('Training random forest model complete.')

        print("Start testing the model.")
        predictions = rf.predict(test_features)

        labels = ['0', '1']
        tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
        print('Random Forest results:')
        print(classification_report(test_labels, predictions, target_names=labels))
        print("TP = {}, TN = {}, FP = {}, FN = {}".format(tp, tn, fp, fn))
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        print('Precision = {} , Recall = {}'.format(precision, recall))

        report_ml = "TP = {}, TN = {}, FP = {}, FN = {}".format(tp, tn, fp, fn)
        report_ml += '\r\n' + 'Precision = {} , Recall = {}'.format(precision, recall)
        write_text(report_ml, './results/' + self.project_name, self.project_name + '_ml_results_random_forest', '.txt')
        if enable_nicad:
            print('Start Nicad.')
            self.export_results(df, predictions, test_labels, test_size)

    def logistic_regression(self, df, test_size=0.5, print_report=False):

        train_features, test_features, train_labels, test_labels = train_test_split(df[self.features], df['label'],
                                                                                    test_size=test_size,
                                                                                    random_state=42)

        logreg = LogisticRegression(random_state=2018, solver='liblinear')

        logreg.fit(train_features, train_labels)
        print('Training logistic regression model complete.')

        predictions = logreg.predict(test_features)

        labels = ['0', '1']
        tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
        print('Logistic Regression results:')
        print(classification_report(test_labels, predictions, target_names=labels))

        print("TP = {}, TN = {}, FP = {}, FN = {}".format(tp, tn, fp, fn))
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        print('Precision = {} , Recall = {}'.format(precision, recall))

        report_ml = "TP = {}, TN = {}, FP = {}, FN = {}".format(tp, tn, fp, fn)
        report_ml += '\r\n' + 'Precision = {} , Recall = {}'.format(precision, recall)
        write_text(report_ml, './results/' + self.project_name, self.project_name + '_ml_results_logistic_regression',
                   '.txt')

        if enable_nicad:
            print('Start Nicad.')
            self.export_results(df, predictions, test_labels, test_size)

    def support_vector(self, df, test_size=0.5, print_report=False):
        train_features, test_features, train_labels, test_labels = train_test_split(df[self.features], df['label'],
                                                                                    test_size=test_size,
                                                                                    random_state=42)

        clf = SVC(C=0.3, random_state=2018, tol=0.0001, kernel='poly', max_iter=-1)

        clf.fit(train_features, train_labels)
        print('Training support vector model complete.')

        predictions = clf.predict(test_features)

        labels = ['0', '1']
        tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
        print('Support Vector results:')
        print(classification_report(test_labels, predictions, target_names=labels))
        print("TP = {}, TN = {}, FP = {}, FN = {}".format(tp, tn, fp, fn))
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        print('Precision = {} , Recall = {}'.format(precision, recall))

        report_ml = "TP = {}, TN = {}, FP = {}, FN = {}".format(tp, tn, fp, fn)
        report_ml += '\r\n' + 'Precision = {} , Recall = {}'.format(precision, recall)
        write_text(report_ml, './results/' + self.project_name, self.project_name + '_ml_results_support_vector',
                   '.txt')

        if enable_nicad:
            print('Start Nicad.')
            self.export_results(df, predictions, test_labels, test_size)

    def train_model(self, dataset_path: str, model_name: str, data_method: str = 'down',
                    test_size: float = 0.50) -> None:
        """
        The main function to train machine learning models
        :param dataset_path: the location for csv file that contains dataset, the default location is in results folder e.g. (./results/druid_sample_features.csv)
        :param model_name: the name of machine learning model (rf-> random_forest, lf-> logistic_regression and svm-> support vectors).
        :param data_method: the method used to solve unbalanced data. (down-> down-sampling and over-> for up-sampling).
        :param test_size: the size for testing phase.
        :return: None
        """
        df = pd.read_csv(dataset_path)

        if df is None:
            print("Dataset is not found.")
            return
        if data_method == 'over':
            df_fix = df[df.label == 0]
            df_bug = df[df.label == 1]

            bug_up_sampled = resample(df_bug,
                                      replace=True,  # sample with replacement
                                      n_samples=len(df_fix),  # match number in majority class
                                      random_state=48)  # reproducible results
            df = pd.concat([df_fix, bug_up_sampled])
        else:
            df_fix = df[df.label == 0]
            df_bug = df[df.label == 1]

            bug_up_sampled = resample(df_fix,
                                      replace=False,  # sample with replacement
                                      n_samples=len(df_bug),  # match number in majority class
                                      random_state=48)  # reproducible results
            df = pd.concat([df_bug, bug_up_sampled])

        if model_name.upper() == 'LR':
            self.logistic_regression(df, test_size)
        elif model_name.upper() == 'RF':
            self.random_forest(df, test_size)
        elif model_name.upper() == 'SVM':
            self.support_vector(df, test_size)
        else:
            print(
                "Error : model name is not supported, the support models are: random forest -> RF or logistic "
                "regression -> LR")
            print("please use model name RF for random forest or LR for logistic regression")
            return

# Prepare data
# Balance data
# Train model
# Test model
