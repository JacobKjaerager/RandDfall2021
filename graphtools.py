
from sklearn.metrics import classification_report, confusion_matrix

def get_heatmap_confusion_matrix(df: pd.DataFrame):
    plt.figure(figsize=(15, 10))
    df_cm = pd.DataFrame(confusion_matrix(df.real, df.predicted, normalize="true"), range(1, 4, 1), range(1, 4, 1))
    sn.heatmap(df_cm, annot=True)
    return model


def get_groups(df: pd.DataFrame) -> list:
    real_groups = df.groupby("real")
    t1 = real_groups.get_group(1).groupby("predicted").count()
    t2 = real_groups.get_group(2).groupby("predicted").count()
    t3 = real_groups.get_group(3).groupby("predicted").count()
    return [t1, t2, t3]


def print_stats(df: pd.DataFrame):
    [t1, t2, t3] = get_groups(df)
    print(classification_report(y_true=y_test, y_pred=pd.Series(y_pred.mean(axis=1).round()).astype(int),
                                labels=[1, 2, 3]))
    print("Correct classified percentage of class 1 is: {}%".format(round(((t1.real.loc[1] * 100) / t1.real.sum()), 3)))
    print("Correct classified percentage of class 2 is: {}%".format(round(((t2.real.loc[2] * 100) / t2.real.sum()), 3)))
    print("Correct classified percentage of class 3 is: {}%".format(round(((t3.real.loc[3] * 100) / t3.real.sum()), 3)))
    print("Correct percentage of testset 1 realtion between predicted 1 and 3 is: {}%".format(
        round(((t1.real.loc[1] * 100) / (t1.real.loc[1] + t1.real.loc[3])), 3)))
    print("Correct percentage of testset 3 realtion between predicted 1 and 3 is: {}%".format(
        round(((t3.real.loc[3] * 100) / (t3.real.loc[1] + t3.real.loc[3])), 3)))


def plot_groups(df: pd.DataFrame):
    [t1, t2, t3] = get_groups(df)
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

    # fig.tight_layout()

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
    return ax