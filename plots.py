import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator

# save figure training log likelihood vs iteration
def figure_log_likelihood(log_lhoods, model_name):
    ax = figure().gca()
    _ = plt.plot(log_lhoods)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _ = plt.ylabel('log likelihood')
    _ = plt.xlabel('iteration')
    _ = plt.title(f'{model_name}: log likelihood on training set', fontweight = 'bold')
    _ = plt.savefig(f'{model_name}_loglhood.png')
    _ = plt.show()
    
# save figure validation AER vs iteration
def figure_AER(AERs, model_name):
    ax = figure().gca()
    _ = plt.plot(AERs)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _ = plt.ylabel('AER')
    _ = plt.xlabel('iteration')
    _ = plt.title(f'{model_name}: AER on validation set', fontweight = 'bold')
    _ = plt.savefig(f'{model_name}_AER.png')
    _ = plt.show()
