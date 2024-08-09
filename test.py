import torch
import torch_directml
import pandas as pd

dml = torch_directml.device()


def minmax(column):
    return 2*(column-column.min())/(column.max()-column.min())-1.0

def convert_value_set_to_number(column):
    unique_values = column.unique()
    replace_dict = {}
    column.replace({"?":"Unknown"})
    for i, val in enumerate(unique_values) :
        replace_dict[val] = i
    return column.replace(replace_dict)

# One hot encoding
def one_hot_encoding(dataframe, column_name:str):
    encoded_df = pd.get_dummies(data=dataframe, columns=[column_name], prefix=column_name)
    print(encoded_df.columns)
    return pd.concat([dataframe, encoded_df], axis=1).drop(column_name, axis=1)

def preprocess_input(df):
    df.columns=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']


    # Temp drop all no numerical
    df = df.drop(['education'], axis=1)
    df['income'] = df.income.replace({" <=50K" : 0, " <=50K." : 0, " >50K": 1, " >50K.": 1}) #cheeky spaces
    df['age'] = minmax(df['age'])
    df['fnlwgt'] = minmax(df['fnlwgt'])
    df['education-num'] = minmax(df['education-num'])
    df['hours-per-week'] = minmax(df['hours-per-week'])
    df['capital-loss'] = minmax(df['capital-loss'])
    df['capital-gain'] = minmax(df['capital-gain'])
    df['workclass'] = minmax(convert_value_set_to_number(df.workclass))
    # df['marital-status'] = minmax(convert_value_set_to_number(df['marital-status']))
    # df['occupation'] = minmax(convert_value_set_to_number(df['occupation']))
    # df['relationship'] = minmax(convert_value_set_to_number(df['relationship']))
    # df['race'] = minmax(convert_value_set_to_number(df['race']))
    df['sex'] = minmax(convert_value_set_to_number(df['sex']))
    df['native-country'] = minmax(convert_value_set_to_number(df['native-country']))


    df = one_hot_encoding(df, 'marital-status')
    df = one_hot_encoding(df, 'occupation')
    df = one_hot_encoding(df, 'relationship')
    df = one_hot_encoding(df, 'race')
    df = df.astype(float)
    df = df.drop('fnlwgt', axis=1)
    # print(df.head())
    df.dropna()
    
    data = df.to_numpy()
    X = data[:, 0:len(df.columns)-2]
    y = data[:, len(df.columns)-1]
    
    return X, y

#TODO / Notes:
# Investigate why we're getting 100% accuracy

#Load the training data
df_dataset = pd.read_csv("dataset/income/adult.data")
X, y = preprocess_input(df_dataset)

df_test = pd.read_csv("dataset/income/adult.test")
Xt, yt = preprocess_input(df_test)

device=dml
device='cpu'
# Split into input (X) and output/label (Y)



from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import RandomOverSampler

# ros = RandomOverSampler(sampling_strategy='minority', random_state=42)
# X_resampled, y_resampled = ros.fit_resample(X, y)
X_resampled, y_resampled = X, y
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=42)

DATA_TYPE = torch.float32
X_train = torch.tensor(X_train, dtype=DATA_TYPE).to(device)
X_test = torch.tensor(X_test, dtype=DATA_TYPE).to(device)
y_train = torch.tensor(y_train, dtype=DATA_TYPE).reshape(-1,1).to(device)
y_test = torch.tensor(y_test, dtype=DATA_TYPE).reshape(-1,1).to(device)

Xt = torch.tensor(Xt, dtype=DATA_TYPE).to(device)
yt = torch.tensor(yt, dtype=DATA_TYPE).reshape(-1,1).to(device)

HIDDEN_LAYER_SIZE = 5
model = torch.nn.Sequential(
    torch.nn.Linear(X.shape[1], HIDDEN_LAYER_SIZE),
    torch.nn.Softmax(),
    torch.nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE),
    torch.nn.Softmax(),
    torch.nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE),
    torch.nn.ReLU(),
    torch.nn.Linear(HIDDEN_LAYER_SIZE, 1),
    torch.nn.Sigmoid()
).to(device, dtype=DATA_TYPE)

loss_function = torch.nn.BCELoss() #Binary cross entropy
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

# Training loop

n_epoch = 1000
batch_size = 10

for epoch in range(n_epoch):
    for i in range(0, len(X_train), batch_size):
        Xbatch = X_train[i:i+batch_size]
        y_pred = model(Xbatch) #run the input through the model to get the current prediction
        ybatch = y_train[i:i+batch_size] #extract the ground truth corresponding to this batch
        loss = loss_function(y_pred, ybatch) #calcullate the loss (mismatch/difference) between ground truth and prediction
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    y_pred = model(X_test)
    in_accuracy = (y_pred.round() == y_test).float().mean()
    # Commented this as I'm not sure if the model parameters get updated when calling model(Xt)
    # yt_pred = model(Xt)
    # t_accuracy = (yt_pred.round() == yt).float().mean()
    t_accuracy=0
    print(f"Finished epoch {epoch}, latest loss {loss}, acc(in) {in_accuracy}, acc(test) {t_accuracy}")



# Questions for Bruce
# When using minmax scaling on a training dataset, could we get an out of range value from the test dataset or in the real world input during inference

