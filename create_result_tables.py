import sys
import os
import pandas as pd

METRICS = 'metric'
MODEL = 'model'
ACCURACY = 'acc'
ROC = 'roc'
F1_SCORE = 'f1'
PRECISION = 'precision'
RECALL = 'recall'
SETTINGS = 'settings'
SPECIFICITY = 'specificity'

ACCURACY_SD = 'acc_sd'
ROC_SD = 'roc_sd'
F1_SD = 'f1_sd'
PREC_SD = 'prec_sd'
REC_SD = 'rec_sd'
SPEC_SD = 'spec_sd'

RESULT_COLUMNS = [SETTINGS, MODEL, ACCURACY, ROC, F1_SCORE, PRECISION, RECALL, SPECIFICITY]
RESULT_COLUMNS2 = [SETTINGS, MODEL, ACCURACY, ACCURACY_SD, ROC, ROC_SD, F1_SCORE, F1_SD, PRECISION, PREC_SD, RECALL, REC_SD, SPECIFICITY, SPEC_SD]


def main():
    input_files = sys.argv[1]
    results_csv = pd.DataFrame(columns=RESULT_COLUMNS)
    for directory in os.listdir(input_files):
        print("seed: ", directory)
        if directory != '.DS_Store':
            for filename in os.listdir(input_files + '/' + directory):
                if filename.startswith('results'):
                    if filename[-5:-4] == '_':
                        suffix = 'overall'
                    else:
                        suffix = ''
                    results = pd.read_csv(input_files + '/' + directory + '/' + filename)
                    models = results.model.unique()
                    for model in models:
                        # print('\n ----------- new model -----------\n')
                        # print(model, directory)
                        model_info = results[results[MODEL] == model]
                        acc = model_info[model_info[METRICS] == ACCURACY]['1'].mean()
                        roc = model_info[model_info[METRICS] == ROC]['1'].mean()
                        f1_score = model_info[model_info[METRICS] == 'fms']['1'].mean()
                        precision = model_info[model_info[METRICS] == PRECISION]['1'].mean()
                        recall = model_info[model_info[METRICS] == RECALL]['1'].mean()
                        specificity = model_info[model_info[METRICS] == SPECIFICITY]['1'].mean()
                        results_csv = results_csv.append({
                            SETTINGS: filename[20:-4] + suffix,
                            MODEL: model,
                            ACCURACY: acc,
                            ROC: roc,
                            F1_SCORE: f1_score,
                            PRECISION: precision,
                            RECALL: recall,
                            SPECIFICITY: specificity
                        }, ignore_index=True)
    average_seeds(results_csv)

def average_seeds(results):
    results_csv = pd.DataFrame(columns=RESULT_COLUMNS)
    settings = results.settings.unique()
    for setting in settings:
        setting_groups = results[results[SETTINGS] == setting]
        models = results.model.unique()
        for model in models:
            setting_model_info = setting_groups[setting_groups[MODEL] == model]
            print(model, setting, setting_model_info[ACCURACY].mean())
            acc = round(setting_model_info[ACCURACY].mean(), 2)
            roc = round(setting_model_info[ROC].mean(), 2)
            f1_score = round(setting_model_info[F1_SCORE].mean(), 2)
            precision = round(setting_model_info[PRECISION].mean(), 2)
            recall = round(setting_model_info[RECALL].mean(), 2)
            specificity = round(setting_model_info[SPECIFICITY].mean(), 2)

            acc_sd = round(setting_model_info[ACCURACY].std(), 2)
            roc_sd = round(setting_model_info[ROC].std(), 2)
            f1_sd = round(setting_model_info[F1_SCORE].std(), 2)
            prec_sd = round(setting_model_info[PRECISION].std(), 2)
            rec_sd = round(setting_model_info[RECALL].std(), 2)
            spec_sd = round(setting_model_info[SPECIFICITY].std(), 2)
            results_csv = results_csv.append({
                SETTINGS: setting,
                MODEL: model,
                ACCURACY: acc,
                ROC: roc,
                F1_SCORE: f1_score,
                PRECISION: precision,
                RECALL: recall,
                SPECIFICITY: specificity,
                
                ACCURACY_SD : acc_sd,
                ROC_SD : roc_sd,
                F1_SD : f1_sd,
                PREC_SD : prec_sd,
                REC_SD : rec_sd,
                SPEC_SD : spec_sd
            }, ignore_index=True)
    results_csv.to_csv(sys.argv[2]+'.csv', index=False)

if __name__ == "__main__":
    main()

