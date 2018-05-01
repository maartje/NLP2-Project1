import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator

# save figure training log likelihood vs iteration
def figure_log_likelihood(log_lhoods, model_name):
    ax = figure().gca()
    _ = plt.plot(log_lhoods)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_label_coords(0.08, 0.79)
    ax.xaxis.set_label_coords(0.85, 0.11)
    ax.yaxis.get_offset_text().set_fontsize(20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    _ = plt.ylabel('LL-hood', fontsize=20, fontstyle = 'oblique')
    _ = plt.xlabel('iteration', fontsize=20, fontstyle = 'oblique')
    _ = plt.savefig(f'{model_name}_llhood.png')
    _ = plt.show()
    
# save figure validation AER vs iteration
def figure_AER(AERs, model_name):
    ax = figure().gca()
    _ = plt.plot(AERs)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_label_coords(0.08, 0.5)
    ax.xaxis.set_label_coords(0.85, 0.16)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    _ = plt.ylabel('AER', fontsize=20, fontstyle = 'oblique')
    _ = plt.xlabel('iteration', fontsize=20, fontstyle = 'oblique')
    _ = plt.savefig(f'{model_name}_AER.png')
    _ = plt.show()

