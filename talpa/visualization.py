import matplotlib.pyplot as plt
import seaborn as sns
import os


def label_distribution(df):
    print(df['activity'].value_counts())
    df['activity'].value_counts().plot(kind='barh')
    path = os.path.join(os.getcwd(), "plots")
    print(path)
    filename = os.path.join(path, "label_distribution.png")
    print(filename)
    plt.savefig(filename)
    plt.show()


def data_distribution(df):
    sns.pairplot(df, hue='activity')
    path = os.path.join(os.getcwd(), "plots")
    print(path)
    filename = os.path.join(path, "data_distribution.png")
    plt.savefig(filename)
    plt.show()


def plot_metrics(name,df):
    ax = df.plot.bar(rot=0)
    path = os.path.join(os.getcwd(), "plots")

    filename = os.path.join(path, name+"_metrics.png")
    plt.savefig(filename)
    plt.show()
