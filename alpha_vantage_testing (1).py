# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 08:14:22 2021

@author: Morten Sahlertz
"""

# Imports
import Models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabl_model import run_tabl_config
import csv
import time
import random
import tensorflow as tf
import keras
import sklearn
from sklearn.metrics import f1_score, accuracy_score
from tensorflow.keras import optimizers
from alpha_vantage.timeseries import TimeSeries
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

np.set_printoptions(precision=2)
#%% Functions

def one_year_stock(key, symbol, interval='1min'):
    # Use TimeSeries from Alpha Vantage
    ts = TimeSeries(key, output_format='csv')
    data, meta = ts.get_intraday_extended(symbol=symbol, interval=interval, slice='year1month1')
    
    df = pd.DataFrame(data)
    total_data = df

    for i in range(2, 13):
        time.sleep(15) # So that Alpha Vantage is over burdened
        slice = 'year1month' + str(i)
        data, meta = ts.get_intraday_extended(symbol=symbol, interval='1min', slice=slice)
        df = pd.DataFrame(data)
        total_data = total_data.append(df)
    
    header_row = 0
    total_data.columns = total_data.iloc[header_row]
    total_data = total_data.drop(header_row)
    total_data['Datetime'] = pd.to_datetime(total_data['time'])
    total_data = total_data.set_index(pd.DatetimeIndex(total_data['Datetime']))
    total_data = total_data.drop(columns=['time', 'Datetime'])
    # Set data as float
    total_data = pd.DataFrame(total_data, dtype=float)
    # Reverse the data
    total_data = total_data.reindex(index=total_data.index[::-1])
    
    return total_data


def get_model_data_alpha(data, sample_size=10, target_time=10, feature_num=5, normalize_data=True):
    Y = np.zeros(len(data)-sample_size)
    for i in range(len(data)-(sample_size+target_time)):
        if (data['close'][i+sample_size+target_time]/data['close'][i+sample_size]) >= 1.002:
            Y[i] = 0
        elif (data['close'][i+sample_size+target_time]/data['close'][i+sample_size]) <= 0.998:
            Y[i] = 2
        else:
            Y[i] = 1
    if normalize_data == True:
        data = (data-data.min())/(data.max()-data.min())
    
    data = data.values
    shape = data.shape
    X = np.zeros((shape[0]-sample_size, sample_size, feature_num), dtype=np.float64)# added dtype as float 32 so it's possible to allocate
    
    for i in range(shape[0]-sample_size):
        X[i] = data[i:i+sample_size,0:feature_num]# take the first 40 columns as features
    #X = X.reshape(X.shape[0], sample_size, feature_num, 1)# add the 4th dimension: 1 channel
    
    # "Benchmark dataset for mid-price forecasting of limit order book data with machine learning"
    # labels 0: equal to or greater than 0.002
    # labels 1: -0.00199 to 0.00199
    # labels 2: smaller or equal to -0.002
    # Y=Y-1 relabels as 0,1,2
    return X, Y
    
def run_normalization(X_train, X_test) -> list:
    # Normalize each image
    # X_train = X_train / X_train.max(axis=0)
    # y_train = y_train / (X_train.max(axis=0)-y_train.max(axis=0))
    for i in range(len(X_train)):
        X_train[i] = (X_train[i] - X_train[i].min(axis=0)) / (X_train[i].max(axis=0) - X_train[i].min(axis=0))

    for i in range(len(X_test)):
        X_test[i] = (X_test[i] - X_test[i].min(axis=0)) / (X_test[i].max(axis=0) - X_test[i].min(axis=0))

    return [X_test, X_train]


def recompile_model(model, lr:float):
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=lr),
                  metrics=['accuracy'])
    return model


def run_fitting(model, X_train: np.ndarray, y_train: np.ndarray, epochs: int, class_weight: dict):
    model.fit(X_train,
              y_train,
              epochs=epochs,
              class_weight=class_weight,
              validation_data=(X_test, y_test))
    return model


def execute_training(model, X_train: np.ndarray, y_train: np.ndarray, X_test, y_test, class_weight: dict):
    model = recompile_model(model, 0.001)
    model = run_fitting(model=model,
                        X_train=X_train,
                        y_train=y_train,
                        epochs=40,
                        class_weight=class_weight)
    model = recompile_model(model, 0.0001)
    model = run_fitting(model=model,
                        X_train=X_train,
                        y_train=y_train,
                        epochs=50,
                        class_weight=class_weight)
    model = recompile_model(model, 0.00001)
    model = run_fitting(model=model,
                        X_train=X_train,
                        y_train=y_train,
                        epochs=50,
                        class_weight=class_weight)
    return model


def run_prediction(model, X_test: np.ndarray) -> list:
    pred = model.predict(X_test)
    y_pred = np.argmax(pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    return [y_pred, y_true]


def print_evaluation_metrics(y_true, y_pred) -> None:
    acc = accuracy_score(y_true, y_pred)
    print(acc)
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f1)


def get_groups(y_true: np.ndarray, y_pred: np.ndarray) -> list:
    df = pd.DataFrame(columns=["predicted", "real"])
    df["predicted"] = pd.Series(y_pred + 1)
    df["real"] = pd.Series(y_true + 1)
    real_groups = df.groupby("real")
    t1 = real_groups.get_group(1).groupby("predicted").count()
    t2 = real_groups.get_group(2).groupby("predicted").count()
    t3 = real_groups.get_group(3).groupby("predicted").count()
    return [t1,t2,t3]

def plot_groups(y_true: np.ndarray, y_pred: np.ndarray):
    [t1,t2,t3] = get_groups(y_true=y_true,
                            y_pred=y_pred)
    plt.rcParams['figure.figsize'] = [15, 10]
    labels = t1.index
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots()
    first_data_bar = [t1.real.loc[1], t2.real.loc[1], t3.real.loc[1]]
    second_data_bar = [t1.real.loc[2], t2.real.loc[2], t3.real.loc[2]]
    third_data_bar = [t1.real.loc[3], t2.real.loc[3], t3.real.loc[3]]
    rects1 = ax.bar(x - width, first_data_bar, width, label='Predicted as 1')
    rects2 = ax.bar(x, second_data_bar, width, label='Predicted as 2')
    rects3 = ax.bar(x + width, third_data_bar, width, label='Predicted as 3')
    ax.set_xlabel('Correct class')
    ax.set_ylabel('Occurences')
    ax.set_title('classified as in correct bin')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    #fig.tight_layout()
    plt.show()

# Plot confusion matrix
# From https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred) - 1]
    classes.sort()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    # Temporary fix to fix y-axis overflow: https://github.com/matplotlib/matplotlib/issues/14751
    ax.set_ylim(cm.shape[0]-0.5, -0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    ax.figure.set_size_inches(12, 12)
    plt.show()


def get_to_df_format(model, X_test, y_test) -> pd.DataFrame:
    pred = model.predict(X_test, verbose=1)
    y_pred = np.argmax(pred, axis=1) + 1
    y_true = np.argmax(y_test, axis=1) + 1
    df = pd.DataFrame(columns=["predicted", "real"])
    df["predicted"] = pd.Series(y_pred.astype(int))
    df["real"] = y_true
    return df

plt.plot(possible_positions,current_fortune)
plt.show()
   




buy_data = data[len(X_train):len(X)]

#%%
def buy_one_stock(stake, stock, brokerage, stock_price):
    if stock_price <= stake:
        stock += 1
        stake -= stock_price+stock_price*brokerage
        return stock, stake
    else:
        return stock, stake
    
def sell_one_stock(stake, stock, brokerage, stock_price):
    if stock > 0:
        stock -= 1
        stake += stock_price-stock_price*brokerage
        return stock, stake
    else:
        return stock, stake
    
def buy_stock_min(stake, stock, brokerage, stock_price, min_stock=60):
    if stake >= (min_stock+1)*stock_price:
        stock_amount = np.floor(stake/stock_price)-min_stock
        stake_amount = stock_price*stock_amount+(stock_price*stock_amount)*brokerage
        stock += stock_amount
        stake -= stake_amount
        return stock, stake, int(stock_amount)
    elif stake < (min_stock+1)*stock_price:
        stock, stake = buy_one_stock(stake, stock, brokerage, stock_price)
        return stock, stake, 1
    
def sell_stock_min(stake, stock, brokerage, stock_price, min_stock=60):
    if stock >= min_stock+1:
        stock_amount = stock-min_stock
        stake_amount = stock_amount*stock_price-(stock_amount*stock_price)*brokerage
        stock -= stock_amount
        stake += stake_amount
        return stock, stake, int(stock_amount)
    elif stock < min_stock+1:
        stock, stake = sell_one_stock(stake, stock, brokerage, stock_price)
        return stock, stake, 1
    
    
def sell_stock(stake, stock, brokerage, stock_price, max_stock=200):
    if max_stock >= stock > 0:
        stake_amount = stock*stock_price-(stock*stock_price)*brokerage
        stock -= stock
        stake += stake_amount
        return stock, stake
    elif max_stock < stock:
        stake_amount = max_stock*stock_price-(max_stock*stock_price)*brokerage
        stock -= max_stock
        stake += stake_amount
        return stock, stake
    else:
        print("Can't sell stock")
        return stock, stake

def buy_stock(stake, stock, brokerage, stock_price, max_stock=200):
    if stock_price <= stake:
        stock_amount = np.floor(stake/stock_price)
        if stock_amount <= max_stock:
            stock += stock_amount
            stake -= stock_price*stock_amount+(stock_price*stock_amount)*brokerage
            return stock, stake
        else:
            stock += max_stock
            stake -= stock_price*max_stock+(stock_price*max_stock)*brokerage
            return stock, stake
    else:
        print("Can't buy stock")
        return stock, stake
    
def sell_stock(stake, stock, brokerage, stock_price, max_stock=200):
    if max_stock >= stock > 0:
        stake_amount = stock*stock_price-(stock*stock_price)*brokerage
        stock -= stock
        stake += stake_amount
        return stock, stake
    elif max_stock < stock:
        stake_amount = max_stock*stock_price-(max_stock*stock_price)*brokerage
        stock -= max_stock
        stake += stake_amount
        return stock, stake
    else:
        print("Can't sell stock")
        return stock, stake
  
def run_stock_game(y_pred):
    stake = 10000
    stock = 0
    brokerage = 0.001
    buy_sell_delay = []
    sell_amount = 0
    buy_amount = 0

    #stock, stake = buy_stock(stake, stock, brokerage, buy_data['high'][0])

    # for _ in range(61):
    #     stock, stake = buy_one_stock(stake, stock, brokerage, buy_data['high'][0])


    # for _ in range(60):
    #     stock, stake, _ = buy_stock_min(stake, stock, brokerage, buy_data['high'][0])


    for i in range(len(y_pred)):
        if y_pred[i] == 3:
            stock, stake = buy_stock(stake, stock, brokerage, buy_data['high'][i])
            buy_sell_delay.append(str(i+60)+'s')
        # if y_pred[i] == 1:
        #     stock, stake = sell_stock(stake, stock, brokerage, buy_data['low'][i])
        #     buy_sell_delay.append(str(i+60)+'b')
        for l in range(len(buy_sell_delay)):
            if buy_sell_delay[l] == str(i)+'s':
                stock, stake = sell_stock(stake, stock, brokerage, buy_data['low'][i])
                buy_sell_delay.remove(str(i)+'s')
                break
            # elif buy_sell_delay[l] == str(i)+'b':
            #     stock, stake = buy_stock(stake, stock, brokerage, buy_data['high'][i])
            #     buy_sell_delay.remove(str(i)+'b')
            #     break
    #     elif y_pred[i] == 1 and counter <= 0:
    #         stock, stake = sell_stock(stake, stock, brokerage, buy_data['low'][i])
    #         stock, stake = buy_stock(stake, stock, brokerage, buy_data['high'][i+60])
    #         counter = 60



    # for i in range(len(y_pred)):
    #     if y_pred[i] == 3:
    #         stock, stake = buy_one_stock(stake, stock, brokerage, buy_data['close'][i])
    #         buy_sell_delay.append(str(i+60)+'s'+str(1))
    #     elif y_pred[i] == 1:
    #         stock, stake = sell_one_stock(stake, stock, brokerage, buy_data['low'][i])
    #         buy_sell_delay.append(str(i+60)+'b'+str(1))
    #     for l in range(len(buy_sell_delay)):
    #         if buy_sell_delay[l][0:len(str(i))+1] == str(i)+'s':
    #             amount = buy_sell_delay[l][len(str(i))+1:len(buy_sell_delay[l])]
    #             print(amount)
    #             stock, stake = sell_one_stock(stake, stock, brokerage, buy_data['close'][i])
    #             buy_sell_delay.remove(str(i)+'s'+str(amount))
    #             break
    #         elif buy_sell_delay[l][0:len(str(i))+1] == str(i)+'b':
    #             amount = buy_sell_delay[l][len(str(i))+1:len(buy_sell_delay[l])]
    #             print(amount)
    #             stock, stake = buy_one_stock(stake, stock, brokerage, buy_data['high'][i])
    #             buy_sell_delay.remove(str(i)+'b'+str(amount))
    #             break
     

    
    # for i in range(len(y_pred)):
    #     if y_pred[i] == 3:
    #         stock, stake, sell_amount = buy_stock_min(stake, stock, brokerage, buy_data['high'][i])
    #         buy_sell_delay.append(str(i+60)+'s'+str(sell_amount))
    #     elif y_pred[i] == 1:
    #         stock, stake, buy_amount = sell_stock_min(stake, stock, brokerage, buy_data['low'][i])
    #         buy_sell_delay.append(str(i+60)+'b'+str(buy_amount))
    #     for l in range(len(buy_sell_delay)):
    #         if buy_sell_delay[l][0:len(str(i))+1] == str(i)+'s':
    #             amount = buy_sell_delay[l][len(str(i))+1:len(buy_sell_delay[l])]
    #             print(amount)
    #             for _ in range(int(amount)):
    #                 stock, stake = sell_one_stock(stake, stock, brokerage, buy_data['close'][i])
    #             buy_sell_delay.remove(str(i)+'s'+str(amount))
    #             break
    #         elif buy_sell_delay[l][0:len(str(i))+1] == str(i)+'b':
    #             amount = buy_sell_delay[l][len(str(i))+1:len(buy_sell_delay[l])]
    #             for _ in range (int(amount)):
    #                 stock, stake = buy_one_stock(stake, stock, brokerage, buy_data['high'][i])
    #             buy_sell_delay.remove(str(i)+'b'+str(amount))
    #             break

    # for i in range(len(y_pred)):
    #     if y_pred[i] == 3:
    #         stock, stake, sell_amount = buy_stock_min(stake, stock, brokerage, buy_data['high'][i])
    #         buy_sell_delay.append(str(i+60)+'s'+str(sell_amount))
    #     for l in range(len(buy_sell_delay)):
    #         if buy_sell_delay[l][0:len(str(i))+1] == str(i)+'s':
    #             amount = buy_sell_delay[l][len(str(i))+1:len(buy_sell_delay[l])]
    #             for _ in range(int(amount)):
    #                 stock, stake = sell_one_stock(stake, stock, brokerage, buy_data['close'][i])
    #             buy_sell_delay.remove(str(i)+'s'+str(amount))
    #             break


    # for i in range(len(y_pred)):
    #     if y_pred[i] == 3:
    #         stock, stake, _ = buy_stock_min(stake, stock, brokerage, buy_data['high'][i])
    #         buy_sell_delay.append(str(i+60)+'s')
    #     for l in range(len(buy_sell_delay)):
    #         if buy_sell_delay[l][0:len(str(i))+1] == str(i)+'s':
    #             amount = buy_sell_delay[l][len(str(i))+1:len(buy_sell_delay[l])]
    #             for _ in range(int(amount)):
    #                 stock, stake = sell_one_stock(stake, stock, brokerage, buy_data['close'][i])
    #             buy_sell_delay.remove(str(i)+'s'+str(amount))
    #             break

def find_occurences(y_true, y_predicted, prediction, bottom_ceiling=0.5, top_ceiling=0.75):
    true_counter = 0
    false_counter = 0    
    for i in range(len(y_predicted)):
        if y_predicted[i] == y_true[i]:
            if(bottom_ceiling < np.max(prediction[i]) < top_ceiling):
                true_counter += 1
        else:
            if(bottom_ceiling < np.max(prediction[i]) < top_ceiling):
                false_counter += 1
                
    print("Amounts of true occurences between {}% and {}% are: {}".format(bottom_ceiling*100, top_ceiling*100, true_counter))
    print("Amounts of false occurences below {}% and {}% are: {}".format(bottom_ceiling*100, top_ceiling*100, false_counter))
    
    print("Percentage of true occurences: {}".format(true_counter/(true_counter+false_counter)))
    print("Percentage of false occurences: {}".format(false_counter/(true_counter+false_counter)))            

#%%

counter = 0
wrong_counter = 0

for i in range(len(y_pred)-10):
    if y_pred[i] == 3:
        for l in range(11):
            if y_pred[i-l] == 1:
                counter += 1
                if y_pred[i] != y_true[i]:
                    wrong_counter += 1
                    print("Wrong at index: {}".format(i))
                break
                
print(counter)
print(wrong_counter)
                
#%%

counter = 0
wrong_counter = 0

for i in range(len(y_pred)-10):
    if y_pred[i] == 1:
        for l in range(11):
            if y_pred[i-l] == 3:
                counter += 1
                if y_pred[i] != y_true[i]:
                    wrong_counter += 1
                    print("Wrong at index: {}".format(i))
                break
                
print(counter)
print(wrong_counter)
                 

#%%

for i in range(len(y_pred)):
    if y_pred[i] == 3:
        print(i)
        break
    
#%%

df1 = pd.DataFrame(pred)
df1.index = buy_data.index

if __name__ == '__main__':
    # %% Get data
    company = "MSFT" #Microsoft imported
    class_weight = {0: 1, 1: 1, 2: 1}
    key = open('alphavantage.txt').read()
    data = one_year_stock(key, symbol=company)
    data['close'].plot()
    # %% Ready data for model
    X, y = get_model_data_alpha(data, sample_size=60, normalize_data=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        shuffle=False)
    # Change y to categorical
    y_train = to_categorical(y_train, 3)
    y_test = to_categorical(y_test, 3)
    run_normalization(X_train=X_train,
                      X_test=X_test)
    model = run_tabl_config()
    trained_model = execute_training(model, X_train, y_train, X_test, y_test, class_weight)
    [y_pred, y_true] = run_prediction(model=model, X_test=X_test)
    print_evaluation_metrics(y_true, y_pred)
    plot_groups(y_true=y_true, y_pred=y_pred)
    df = get_to_df_format(model, X_test, y_test)
    find_occurences(y_true, y_pred, pred=, bottom_ceiling=0.40, top_ceiling=0.50)