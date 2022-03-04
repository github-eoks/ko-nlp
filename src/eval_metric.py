from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pandas as pd

data_ = pd.read_csv('./data/123.csv',
                    encoding="cp949", sep=';', error_bad_lines=False)

y_true = data_['Data Set Value']
y_pred = round(data_['Expected Value for s_4'],0)

round(data_['Expected Value for s_4'],0)

# y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
# y_pred = [0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 0, 1, 1, 2, 2]

target_names = ['0', '1', '2', '4']

print(confusion_matrix(y_true, y_pred))
print(round(accuracy_score(y_true, y_pred), 4))
print(classification_report(y_true, y_pred, target_names=target_names))