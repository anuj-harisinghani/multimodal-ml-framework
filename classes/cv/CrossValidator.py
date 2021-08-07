from abc import ABC
from classes.factories.TrainersFactory import TrainersFactory
from classes.handlers.ParamsHandler import ParamsHandler

import numpy as np
import os
import csv
import pandas as pd
import operator


class CrossValidator(ABC):
    def __init__(self, mode: str, classifiers: list):
        self.__trainer = TrainersFactory().get(mode)
        self.mode = mode
        self.classifiers = classifiers

    def cross_validate(self, tasks_data: dict):
        new_features_results_prefix = 'results_new_features'
        task_fusion_prefix = 'results_task_fusion'
        feature_importance = False

        # running trainer for each of the tasks
        if self.mode == 'single_tasks':
            for task in tasks_data.keys():
                print("\nTask %s" % task)
                print("---------------")

                task_params = ParamsHandler.load_parameters(task)
                feature_sets = task_params['features']

                # running trainer for each modality separately
                for modality, modality_data in tasks_data[task].items():
                    modality_feature_set = list(feature_sets[modality].keys())[0]

                    # splitting data then training the models
                    trained_models = {clf: self.__trainer.train(data=modality_data, clf=clf, feature_set=modality_feature_set,
                                                                feature_importance=False) for clf in self.classifiers}

                    # saving results
                    CrossValidator.save_results(trained_models=trained_models, feature_set=modality_feature_set,
                                                prefix=new_features_results_prefix, method='default', saveToCSV=True,
                                                getPrediction=True, feature_importance=False)

        # currently this is the same as single_tasks, checking if it would make a difference to keep them separate or not
        elif self.mode == 'fusion':
            trained_models = []
            method = 'task_fusion'

            # running the trainer for each of the tasks
            for task in tasks_data.keys():
                print("\nTask %s" % task)
                print("---------------")

                trained_models_task = []
                task_params = ParamsHandler.load_parameters(task)
                feature_sets = task_params['features']

                # running trainer for each modality separately
                for modality, modality_data in tasks_data[task].items():
                    modality_feature_set = list(feature_sets[modality].keys())[0]

                    # splits, x_columns = self.__splitter.make_splits(data=modality_data, nfolds=nfolds)
                    trained_models_modality = {clf: self.__trainer.train(data=modality_data, clf=clf, feature_set=modality_feature_set,
                                                                         feature_importance=feature_importance) for clf in self.classifiers}

                    # saving each modality's results
                    CrossValidator.save_results(trained_models=trained_models_modality, feature_set=modality_feature_set,
                                                prefix=task_fusion_prefix, method=method, saveToCSV=True,
                                                getPrediction=True, feature_importance=feature_importance)

                    trained_models_task.append(trained_models_modality)

                # aggregating modality-wise results to make task-level results
                if len(trained_models_task) > 1:
                    data = {}
                    for clf in self.classifiers:
                        data[clf] = CrossValidator.aggregate_results(data=trained_models_task, model=clf)
                    trained_models_task = [data]

                trained_models.append(trained_models_task[0])

                # re-calculate performance metrics after aggregation of modality-wise data
                trained_models_results = {clf: self.__trainer.calculate_task_fusion_results(data=trained_models_task[0][clf])
                                          for clf in self.classifiers}

                CrossValidator.save_results(trained_models=trained_models_results, feature_set=task,
                                            prefix=task_fusion_prefix, method=method, saveToCSV=True,
                                            getPrediction=True, feature_importance=feature_importance)

            # compiling the data from all tasks here then aggregating them
            final_trained_models = {}
            for clf in self.classifiers:
                final_trained_models[clf] = CrossValidator.aggregate_results(data=trained_models, model=clf)

            # recalculating results after aggregation of data from all tasks
            final_trained_models_results = {clf: self.__trainer.calculate_task_fusion_results(data=final_trained_models[clf])
                                            for clf in self.classifiers}

            # saving results after full aggregation
            CrossValidator.save_results(trained_models=final_trained_models_results, feature_set='',
                                        prefix=task_fusion_prefix, method=method, saveToCSV=True,
                                        getPrediction=True, feature_importance=feature_importance)


    @staticmethod
    def save_results(trained_models, feature_set, prefix, if_exists='replace', method='default',
                     saveToCSV=False, getPrediction=False, feature_importance=False):

        # required values
        feat_csv_writer = None
        feat_f = None
        feat_fold_csv_writer = None
        pred_csv_writer = None
        params = ParamsHandler.load_parameters('settings')
        output_folder = params['output_folder']
        random_seed = params['random_seed']

        prediction_prefix = 'predictions'
        feature_fold_prefix = 'features_fold'
        feature_prefix = 'features'
        metrics = ['acc', 'roc', 'fms', 'precision', 'recall', 'specificity']
        results_path = os.path.join(os.getcwd(), 'results', output_folder, str(random_seed))

        if not os.path.exists(results_path):
            os.makedirs(results_path)

        dfs = []
        name = "%s_%s" % (prefix, feature_set)
        pred_f = None
        feat_fold_f = None
        if getPrediction:
            pred_f = open('%s_%s.csv' % (os.path.join(results_path, prediction_prefix), name), 'w')
            pred_csv_writer = csv.writer(pred_f)
            headers = ['model', 'PID', 'prob_0', 'prob_1', 'pred']
            pred_csv_writer.writerow(headers)

        if feature_importance:
            feat_fold_f = open('%s_%s.csv' % (os.path.join(results_path, feature_fold_prefix), name), 'w')
            feat_fold_csv_writer = csv.writer(feat_fold_f)
            headers_fold = ['model', 'fold', 'feature', 'score1', 'score2', 'odds_ratio', 'CI_low', 'CI_high', 'p_value']
            feat_fold_csv_writer.writerow(headers_fold)

            feat_f = open('%s_%s.csv' % (os.path.join(results_path, feature_prefix), name), 'w')
            feat_csv_writer = csv.writer(feat_f)
            headers = ['model', 'feature', 'score1', 'score2', 'odds_ratio', 'CI_low', 'CI_high', 'p_value']
            feat_csv_writer.writerow(headers)

        for model in trained_models:
            cv = trained_models[model]
            k_range = cv.best_k[method]['k_range']
            k_range = [1]
            for metric in metrics:
                if metric in cv.results[method].keys():
                    results = cv.results[method][metric]
                    # print('@@', metric, results)
                    df = pd.DataFrame(results, columns=k_range)
                    df['metric'] = metric
                    df['model'] = model
                    dfs += [df]
            if getPrediction:
                for pid, prob in cv.pred_probs[method].items():
                    prob_0 = prob[0]
                    prob_1 = prob[1]
                    pred = cv.preds[method][pid]
                    row = [model, pid, prob_0, prob_1, pred]
                    pred_csv_writer.writerow(row)
            if feature_importance:
                if cv.feature_scores_fold[method] is not None and \
                        cv.feature_scores_all[method] is not None:

                    i = 0
                    for feat_score in cv.feature_scores_fold[method]:
                        sorted_feat_score = sorted(feat_score.items(),
                                                   key=operator.itemgetter(1), reverse=True)
                        for feat, score in sorted_feat_score:
                            row = [model, i, feat, score[0], score[1], score[2], score[3], score[4], score[5]]
                            feat_fold_csv_writer.writerow(row)
                        i += 1
                    sorted_feat_score_all = \
                        sorted(cv.feature_scores_all[method].items(),
                               key=operator.itemgetter(1),
                               reverse=True)
                    for feat, score in sorted_feat_score_all:
                        row = [model, feat, score[0], score[1], score[2], score[3], score[4], score[5]]
                        feat_csv_writer.writerow(row)

        df = pd.concat(dfs, axis=0, ignore_index=True)
        # typedict = {col_name: types.Float(precision=5, asdecimal=True) \
        #             for col_name in df}
        # typedict['metric'] = types.NVARCHAR(length=255)
        # typedict['model'] = types.NVARCHAR(length=255)
        # df.to_sql(name, cnx, if_exists=if_exists, dtype=typedict)
        if saveToCSV:
            df.to_csv(os.path.join(results_path, name + '.csv'), index=False)
        if getPrediction:
            pred_f.close()
        if feature_importance:
            feat_f.close()
            feat_fold_f.close()

    @staticmethod
    def aggregate_results(data: list, model: str):
        method = 'task_fusion'
        avg_preds = {}
        avg_pred_probs = {}

        # this portion gets activated when across_tasks or across modalities aggregation is required
        # since the model being passed is a single model (either GNB, or RF, or LR)
        # if type(model) == str:
        new_data = data[-1][model]
        num = len(data)
        sub_data = np.array([data[t][model] for t in range(num)])

        # this portion gets activated when within_tasks aggregation is required
        # since the models being passed will be more than one
        # elif type(model) == list:
        #     new_data = data[model[-1]]
        #     num = len(model)
        #     sub_data = np.array([data[m] for m in model])

        # sub_data will hold all the DementiaCV instances for a particular model, across all tasks
        # so for task='PupilCalib+CookieTheft+Reading+Memory':
        #        sub_data[0] = DementiaCV class for PupilCalib, some model
        #        sub_data[1] = DementiaCV class for CookieTheft, some model.. so on.

        # pred_probs --------------------------------------------------------------------------------------------------

        # find the union of all pids across all tasks
        union_pids = np.unique(np.concatenate([list(sub_data[i].pred_probs[method].keys()) for i in range(num)]))
        pred_probs_dict = {}

        # averaging the pred_probs for a certain PID whenever it's seen across all tasks
        for i in union_pids:
            pred_probs_sum_list = np.zeros(3)
            for t in range(num):
                if i in sub_data[t].pred_probs[method]:
                    pred_probs_sum_list[0] += sub_data[t].pred_probs[method][i][0]
                    pred_probs_sum_list[1] += sub_data[t].pred_probs[method][i][1]
                    pred_probs_sum_list[2] += 1
            pred_probs_dict[i] = np.array([pred_probs_sum_list[0] / pred_probs_sum_list[2], pred_probs_sum_list[1] / pred_probs_sum_list[2]])

        avg_pred_probs[method] = pred_probs_dict
        new_data.pred_probs = avg_pred_probs

        # preds ------------------------------------------------------------------------------------------------------

        # assigning True or False for preds based on what the averaged pred_probs were found in the previous step
        preds_dict = {}
        for i in avg_pred_probs[method]:
            preds_dict[i] = avg_pred_probs[method][i][0] < avg_pred_probs[method][i][1]

        avg_preds[method] = preds_dict
        new_data.preds = avg_preds

        # returned the updated new_data - only pred_probs and preds are changed, the rest are the same as the initially chosen new_data
        return new_data
