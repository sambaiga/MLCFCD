import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import multilabel_confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import itertools
nice_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        'axes.labelsize': 8, # fontsize for x and y labels (was 10)
        'axes.titlesize': 8,
        # Use 10pt font in plots, to match 10pt font in document
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
}
matplotlib.rcParams.update(nice_fonts)
SPINE_COLOR="gray"
import seaborn as sns
sns.set_color_codes("pastel")
import itertools

def set_figure_size(fig_width=None, fig_height=None, columns=2):
    assert(columns in [1,2])

    if fig_width is None:
        fig_width = 3.39 if columns==1 else 6.9 # width in inches

    if fig_height is None:
        golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height + 
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES
    return (fig_width, fig_height)


def format_axes(ax):
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)
    
    
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)
    return ax

def figure(fig_width=None, fig_height=None, columns=2):
    """
    Returns a figure with an appropriate size and tight layout.
    """
    fig_width, fig_height =set_figure_size(fig_width, fig_height, columns)
    fig = plt.figure(figsize=(fig_width, fig_height))
    return fig



def legend(ax, ncol=3, loc=9, pos=(0.5, -0.1)):
    leg=ax.legend(loc=loc, bbox_to_anchor=pos, ncol=ncol)
    return leg

def savefig(filename, leg=None, format='.eps', *args, **kwargs):
    """
    Save in PDF file with the given filename.
    """
    if leg:
        art=[leg]
        plt.savefig(filename + format, additional_artists=art, bbox_inches="tight", *args, **kwargs)
    else:
        plt.savefig(filename + format,  bbox_inches="tight", *args, **kwargs)
    plt.close()


def plot_learning_curve(tra_loss_list, tra_f1_list, val_loss_list, val_f1_list):
    
    def line_plot(y_train, y_val, early_stoping, y_label="Loss", y_min=None, y_max=None, best_score=None):
        iterations = range(1,len(y_train)+1)
        if y_min is None:
            y_min = min(min(y_train), min(y_val))
            y_min = max(0, (y_min - y_min*0.01))
        if y_max is None:
            y_max = max(max(y_train), max(y_val))
            y_max = min(1, (y_max + 0.1*y_max))

       
        plt.plot(iterations, y_train, label="training " )
        plt.plot(iterations, y_val, label="validation ")

        if best_score:
            
            plt.title(r"\textbf{Learning curve}"  f": best score: {best_score}",  fontsize=8)
            #plt.axvline(early_stoping, linestyle='--', color='r',label='Early Stopping')
       
        else:
            plt.title(r'\textbf{Learning curve}')
           

        plt.ylabel(y_label)
        #plt.ylim(y_min, y_max)
        plt.xlabel(r"Iterations")
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
       
        plt.legend(loc="best")
        ax = plt.gca()
        ax.patch.set_alpha(0.0)
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))  
        format_axes(ax)
    

   

    min_val_loss_poss = val_loss_list.index(min(val_loss_list))+1 
    min_val_score_poss = val_f1_list.index(max(val_f1_list))+1 
    
    

    fig = figure(fig_width=8)
    plt.subplot(1,2,1)
    line_plot(tra_loss_list, val_loss_list, min_val_loss_poss, y_label="Loss", y_min=0)
   
    
    plt.subplot(1,2,2)
    
    line_plot(tra_f1_list, val_f1_list, min_val_score_poss, y_label="Accuracy", y_min=None, y_max=1, best_score=np.max(val_f1_list))
    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout(pad=1.0)
    
    
   

def plot_confusion_matrix_multilabel(cm, class_names, classes=["OFF", "ON"], column=None, row=1, cmap=plt.cm.Blues):
    '''[summary]
    
    Arguments:
        y_true {[type]} -- [description]
        y_pred {[type]} -- [description]
        class_names {[type]} -- [description]
    
    Keyword Arguments:
        classes {list} -- [description] (default: {["Active", "OFF"]})
        row {int} -- [description] (default: {2})
    '''
    if column is None:
         column=len(class_names)
    #plt.figure(figsize=figsize)
    for k in range(len(class_names)):
        plt.subplot(row, column, k+1)
        plt.imshow(cm[k,:,:], interpolation='nearest', cmap=cmap)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=0)
        plt.yticks(tick_marks, classes)
        plt.tight_layout()
        plt.title(class_names[k])
        thresh = cm[k,:,:].max() / 2.
        fmt =   'd'
        for i, j in itertools.product(range(cm[k,:,:].shape[0]), range(cm[k,:,:].shape[1])):
            plt.text(j, i, format(cm[k,:,:][i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[k,:,:][i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()