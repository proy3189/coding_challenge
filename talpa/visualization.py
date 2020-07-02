import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging



def targetclass_distribution(df):
    '''
    Plot the Target class distribution
    :param df: Dataframe of shape (n_samples, n_features)
            The input samples.
    :return:
    '''
    logging.info("Count of each target class labels : {} ".format(df['activity'].value_counts()))
    df['activity'].value_counts().plot(kind='barh')
    path = os.path.join(os.getcwd(), "plots")
    filename = os.path.join(path, "label_distribution.png")
    plt.title("Target class distribution")
    plt.savefig(filename)
    plt.show()


def data_distribution(df):
    '''
    Plot the data distribution
    :param df: Dataframe of shape (n_samples, n_features)
            The input samples.
    :return:
    '''
    sns.pairplot(df, hue='activity')
    path = os.path.join(os.getcwd(), "plots")
    filename = os.path.join(path, "data_distribution.png")
    plt.title("Data Distribution")
    plt.savefig(filename)
    plt.show()


def plot_metrics(name,df):
    '''
    Plot Accuracy and F1 score of the classifier
    :param name: Classifier name
    :param df: Dataframe of shape (n_samples, n_features)
            The input samples.
    :return:
    '''
    ax = df.plot.bar(rot=0)
    path = os.path.join(os.getcwd(), "plots")
    filename = os.path.join(path, name+"_metrics.png")
    plt.title(name)
    plt.savefig(filename)
    plt.show()


def visualise_mising_values(df):
    '''
    Heat map to show the missing values in the dataset
    :param df: Dataframe of shape (n_samples, n_features)
            The input samples.
    :return:
    '''
    path = os.path.join(os.getcwd(), "plots")
    filename = os.path.join(path,"Missing_Values_Heatmap.png")

    cols = df.columns
    colours = ['g', 'r']
    f, ax = plt.subplots(figsize=(19, 30))
    sns.set_style("whitegrid")
    plt.title('Missing Values Heatmap')
    sns.heatmap(df[cols].isnull(), cmap=sns.color_palette(colours))
    plt.savefig(filename)
    plt.show()




