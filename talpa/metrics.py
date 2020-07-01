from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix



def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def f1_measure(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')

def recall(y_true, y_pred):
    rec = recall_score(y_true, y_pred)
    return rec



def precision(y_true, y_pred):
    prec = precision_score(y_true, y_pred)
    return prec


def confusion_metric(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    #tn, fp, fn, tp = cm.ravel()
    #return tn, fp, fn, tp
    return cm