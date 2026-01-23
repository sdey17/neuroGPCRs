import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score, confusion_matrix

def cal_metrics(df):
    pred = list(df['Predictions'].apply(lambda x: [int(el) for el in x.strip("[]").split(".")[0]]))
    label = list(df['Label'])
    tn, fp, fn, tp = confusion_matrix(label, pred, labels=[0,1]).ravel()
    acc = np.round(accuracy_score(label, pred), 3)
    mcc = np.round(matthews_corrcoef(label, pred), 3)
    roc = np.round(roc_auc_score(label, pred), 3)
    sen = np.round(tp/(tp + fn), 3)
    spec = np.round(tn/(tn + fp), 3)
    return acc, spec, sen, mcc, roc

results_val = []
results_unseen_prot = []
results_unseen_lig = []

for i in range(5):
    val_df = pd.read_csv('val_pred_fine_tune_only_mol_ProtBert_MolFormer_{}.csv'.format(i), index_col=0)
    test_df_unseen_prot = pd.read_csv('test_pred_unseen_prot_fine_tune_only_mol_ProtBert_MolFormer_{}.csv'.format(i), index_col=0)
    test_df_unseen_lig = pd.read_csv('test_pred_unseen_lig_fine_tune_only_mol_ProtBert_MolFormer_{}.csv'.format(i), index_col=0)

    accuracy, specificity, sensitivity, mcc, roc = cal_metrics(val_df)
    results_val.append([i, accuracy, specificity, sensitivity, mcc, roc])

    accuracy, specificity, sensitivity, mcc, roc = cal_metrics(test_df_unseen_prot)
    results_unseen_prot.append([i, accuracy, specificity, sensitivity, mcc, roc])

    accuracy, specificity, sensitivity, mcc, roc = cal_metrics(test_df_unseen_lig)
    results_unseen_lig.append([i, accuracy, specificity, sensitivity, mcc, roc])

columns = ['Dataset', 'Accuracy', 'Specificity', 'Sensitivity', 'MCC', 'AUC']

results_val_df = pd.DataFrame(results_val, columns=columns)
results_unseen_prot_df = pd.DataFrame(results_unseen_prot, columns=columns)
results_unseen_lig_df = pd.DataFrame(results_unseen_lig, columns=columns)

results_val_df.to_csv('val_metrics_fine_tune_only_mol_ProtBert_MolFormer.csv')
results_unseen_prot_df.to_csv('test_unseen_prot_metrics_fine_tune_only_mol_ProtBert_MolFormer.csv')
results_unseen_lig_df.to_csv('test_unseen_lig_metrics_fine_tune_only_mol_ProtBert_MolFormer.csv')


