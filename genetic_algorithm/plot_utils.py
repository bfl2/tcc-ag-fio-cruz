import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import time

def sumMatrixes(matrixes):
    tp=fp=fn=tn=0
    for matrix in matrixes:
        tp+=matrix[1][1]
        fn+=matrix[1][0]
        fp+=matrix[0][1]
        tn+=matrix[0][0]
    return [[tn, fn],[fp, tp]]

def printConfusionMatrix(cms, plot_matrix=False):
    cm = sumMatrixes(cms)
    print("Accumulated Confusion Matrix:")
    print(cm[0])
    print(cm[1])
    if(plot_matrix):
        plotConfusionMatrix(cm)
    return cm

def plot_to_file():
    reports_folder = "reports"
    millis = int(round(time.time() * 1000))%1000
    time_formatted = time.strftime("%H %M %S", time.gmtime(time.time()))
    image_name = "{}/confusion-matrix-{} {}.png".format(reports_folder, time_formatted, millis)
    plt.savefig(image_name)

    return image_name

def plotConfusionMatrix(cm, save_plot_to_file=True):

    sns.set(font_scale=2)
    fig = plt.figure(figsize=(10,10))
    heatmap = sns.heatmap(cm, center=1, fmt="d", annot=True, annot_kws={"ha": 'center',"va": 'center'})
    i = 0
    # Centering numbers
    for t in heatmap.texts:
        trans = t.get_transform()
        if(i < 2):
            offs = matplotlib.transforms.ScaledTranslation(0, 0.25, matplotlib.transforms.IdentityTransform())
            t.set_transform( offs + trans )
        else:
            offs = matplotlib.transforms.ScaledTranslation(0, -0.25,matplotlib.transforms.IdentityTransform())
            t.set_transform( offs + trans )
        i += 1

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, va="center", ha='center', fontsize=14)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, va="center", ha='center', fontsize=14)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if(save_plot_to_file):
        image_name = plot_to_file()
    else:
        plt.show()
    print("Printed Confusion Matrix to file:", image_name)

    return