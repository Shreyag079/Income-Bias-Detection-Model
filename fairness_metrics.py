import pandas as pd
from sklearn.metrics import confusion_matrix

def compute_fairness(X, y_true, y_pred, sensitive_col):
    
    df = X.copy()
    df['actual'] = y_true
    df['pred'] = y_pred
    
    results = {}
    
    for group in df[sensitive_col].unique():
        g = df[df[sensitive_col]==group]
        
        tn, fp, fn, tp = confusion_matrix(g['actual'], g['pred']).ravel()
        
        TPR = tp/(tp+fn) if (tp+fn)>0 else 0
        FPR = fp/(fp+tn) if (fp+tn)>0 else 0
        PPR = (tp+fp)/len(g)
        
        results[group] = {
            "TPR":TPR,
            "FPR":FPR,
            "PPR":PPR
        }
    
    return results