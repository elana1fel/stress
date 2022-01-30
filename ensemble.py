import sys
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

model_path = {
    # 'wei': './result/dev_ensemble_wei.csv',
    # 'yao': './result/dev_ensemble_0.5955_yao.csv',
    # 'tommy': './result/dev_ensemble_0.5974_tommy.csv'
    'wei': './result/test_ensemble_wei.csv',
    'yao': './result/test_ensemble_0.5955_yao.csv',
    'tommy': './result/test_ensemble_0.5974_tommy.csv'
}
model_count = len(model_path.keys())

inverse_category = {
    0: 'moderate',
    1: 'severe',
    2: 'not depression'
}

# read all predictions
df = {}
for key, value in model_path.items():
    df[key] = pd.read_csv(model_path[key])

ensemble_weight = [1, 0.67, 0.69]
POWER = 4
max_f1 = 0

answer = [] 

for i in range(len(df['wei'])):
    prob = []
    if model_count == 1:
        for prob_1 in zip(df['wei'].iloc[i].values.tolist()[1:]):
            prob.append(prob_1)
    elif model_count == 2:
        for prob_1, prob_2 in \
            zip(df['wei'].iloc[i].values.tolist()[1:], df['tommy'].iloc[i].values.tolist()[1:]):
            current_prob = (prob_1**POWER) * ensemble_weight[0] + (prob_2**POWER) * ensemble_weight[1]
            prob.append(current_prob)
    else:
        # 0 is index in dataframe
        for prob_1, prob_2, prob_3 in \
            zip(df['wei'].iloc[i].values.tolist()[1:], df['yao'].iloc[i].values.tolist()[1:], df['tommy'].iloc[i].values.tolist()[1:]):
            current_prob = (prob_1**POWER) * ensemble_weight[0] + (prob_2**POWER) * ensemble_weight[1] + (prob_3**POWER) * ensemble_weight[2]
            prob.append(current_prob)

    category = prob.index(max(prob))
    pid = 'test_pid_'+ str(i+1)
    answer.append([pid, inverse_category[category]])

assert len(answer) == len(df['wei'])
answer = pd.DataFrame(answer, columns = ['pid', 'class_label'])
# answer.to_csv('answer.csv', index = False)

with open('./NYCU_TWD_ensemble.tsv','w') as write_tsv:
    write_tsv.write(answer.to_csv(sep='\t', index=False))

# # evaluate
# df_val = pd.read_csv('../data/val_np.tsv', sep='\t')
# df_predicted = pd.read_csv('./answer.csv')

# gt = df_val['Label']
# preds = df_predicted['class_label']

# labels = ['moderate', 'severe', 'not depression']

# f1 = f1_score(gt, preds, average = 'macro')
# print(confusion_matrix(gt, preds, labels = labels))
# print('Macro F1: ', f1)
