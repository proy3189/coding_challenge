import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np



def targetclass_distribution(df):
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

def visualise_mising_values(df):
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

def res(df):
    #df = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])
    print("index", df.index)
    print("columns", df.columns)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    df[["TrueLabel"]].value_counts().plot(ax=axes[0], kind='bar')
    df[["Predicted"]].value_counts().plot(ax=axes[1], kind='bar');
    plt.show()



