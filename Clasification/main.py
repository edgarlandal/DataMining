import pandas as pd

#libreries for processing
import methods as me
def separator(data, output):
    inputs = data.drop(labels=[output], axis= 1)
    targets = data[output]
    return inputs, targets

data = pd.read_csv('iris.csv')
x_true, y_true = separator(data, 'species')


print(data.info())
me.method_hold_out(x_true, y_true, 0.4, 0.6)
me.method_hold_out(x_true, y_true, 1, 1)
me.method_random_subsampling(x_true, y_true, 30, 0.5 , 0.25, 0)
me.method_K_Fold(x_true, y_true, 10, False, True)
me.method_SCV(x_true, y_true, 10)
me.method_leave_one_out(x_true, y_true)
