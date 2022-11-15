def missing_values(data):
    missing_values_count = data.isnull().sum()
    missing_values_count[0:40]
    
    total_cells = np.product(data.shape)
    total_missing = missing_values_count.sum()
    
    percent_missing = (total_missing/total_cells) * 100
    print(percent_missing)

def data_information(data):
    print(data.info())
    print(data.head())
    print(data.describe())
    
def delete_columns(data):
    return data.dropna(axis=1)

def remplace_for_constant(data):
    return data.fillna(method="bfill", axis=0).fillna(0)

def remplace_for_mean(data):
    dataAux = data
    for i in data:
        dataAux[i].fillna(data[i].mean, inplace = True)
    return dataAux