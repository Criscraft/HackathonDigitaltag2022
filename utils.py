
import numpy as np
import matplotlib.pyplot as plt


def init_log(log_path):
    """Create a log textfile, overwrite the old one if it already exists
    
    Arguments:
        log_path {str} -- Filename of the log textfile
    """

    textline = "\t".join([
        'epoch', 
        'loss', 
        'accuracy',
        ])
    with open(log_path, 'w') as data:
        data.write("".join([textline, "\n"]))

def write_log(log_path, epoch, loss, accuracy):
    """Write data into existing logfile
    
    Arguments:
        log_path {str} -- Filename of the log textfile
        epoch {int} -- epoch to log
        loss {float} -- loss value to log
        accuracy {float} -- accuracy value to log
    """
    textline = '\t'.join([
        '{:d}'.format(epoch),
        '{:g}'.format(loss),
        '{:g}'.format(accuracy),
        ])
    with open(log_path, 'a') as data:
        data.write("".join([textline, "\n"]))


class AnimationPlotter():
    """Plot the loss curve during training for train and test set
    
    Arguments:
        train_filename {str} -- Filename of the train log textfile
        test_filename {str} -- Filename of the test log textfile
        label {str} -- Name of y axis
    """
    def __init__(self, train_filename, test_filename, label):
        self.train_filename = train_filename
        self.test_filename = test_filename
        self.label = label

        #load loss data from disk
        train_data, test_data = self.read_data(self.train_filename, self.test_filename, self.label)

        #plot data
        self.fig, self.ax, self.handle_train, self.handle_test = plot_curves([train_data, test_data], x_label='epochs', y_label=label, labels=['train', 'test'])

    def read_data(self, train_filename, test_filename, col):
        with open(train_filename, 'r') as f:
            train_data = np.genfromtxt(f, names=True, usecols=['epoch', col], delimiter='\t')
            try:
                train_data = train_data.view(float).reshape(train_data.shape + (-1,))
            except ValueError:
                train_data = np.zeros((1,2))
        with open(test_filename, 'r') as f:
            test_data = np.genfromtxt(f, names=True, usecols=['epoch', col], delimiter='\t')
            try:
                test_data = test_data.view(float).reshape(test_data.shape + (-1,))
            except ValueError:
                test_data = np.zeros((1,2))
        return train_data, test_data

    def update_values(self):
        #load loss data from disk
        train_data, test_data = self.read_data(self.train_filename, self.test_filename, self.label)
        #plot data
        self.handle_train.set_data(train_data[:,0], train_data[:,1])
        self.handle_test.set_data(test_data[:,0], test_data[:,1])

        #self.fig.canvas.draw()
        plt.show()


def plot_curves(data_list, x_label='x', y_label='y', labels=[]):
    """generic plot function for drawing lines
    
    Arguments:
        data_list {list} -- List of ndarrays to plot
    
    Keyword Arguments:
        x_label {str} -- label of x axis (default: {'x'})
        y_label {str} -- label of y axis (default: {'y'})
        labels {list} -- label of plotted lines (default: {[]})
    
    Returns:
        figure -- matplotlib figure
        ax -- matplotlib ax

    """
    if labels:
        label_iterator = iter(labels)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    handles = []
    legend_labels = []

    for i, data in enumerate(data_list):
        handle, = ax.plot(data[:,0], data[:,1], linestyle='-')
        handles.append(handle)   
        if labels:
            legend_labels.append(next(label_iterator))
        else:
            legend_labels.append(i)

    plt.legend(handles, legend_labels)
    xlabels = ax.get_xticklabels()
    plt.setp(xlabels, rotation=45, horizontalalignment='right')
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_ylim(0.,1.)
    plt.grid()
    fig.tight_layout()

    return fig, ax, handles[0], handles[1]