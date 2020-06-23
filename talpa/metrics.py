from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report



def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)



def recall(y_true, y_pred):
    rec = recall_score(y_true, y_pred)
    return rec



def precision(y_true, y_pred):
    prec = precision_score(y_true, y_pred)
    return prec