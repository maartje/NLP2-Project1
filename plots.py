import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator

# save figure training log likelihood vs iteration
def figure_log_likelihood(log_lhoods, model_name=None):
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
    if model_name:
         _ = plt.savefig(f'{model_name}_llhood.png')
    _ = plt.show()
    
# save figure validation AER vs iteration
def figure_AER(AERs, model_name = None):
    ax = figure().gca()
    _ = plt.plot(AERs)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_label_coords(0.08, 0.5)
    ax.xaxis.set_label_coords(0.85, 0.16)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    _ = plt.ylabel('AER', fontsize=20, fontstyle = 'oblique')
    _ = plt.xlabel('iteration', fontsize=20, fontstyle = 'oblique')
    if model_name:
        _ = plt.savefig(f'{model_name}_AER.png')
    _ = plt.show()

def figure_llhood_multiple_lines(llhood_lines, line_labels, model_name=None):
    ax = figure().gca()
    for i, llhoods in enumerate(llhood_lines):
        _ = plt.plot(llhoods, label = line_labels[i])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_label_coords(0.08, 0.79)
    ax.xaxis.set_label_coords(0.5, 0.11)
    ax.yaxis.get_offset_text().set_fontsize(20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    _ = plt.ylabel('LL-hood', fontsize=20, fontstyle = 'oblique')
    _ = plt.xlabel('iteration', fontsize=20, fontstyle = 'oblique')
    plt.legend()
    if model_name:
         _ = plt.savefig(f'{model_name}_llhood.png')
    _ = plt.show()

def figure_AER_multiple_lines(AER_lines, line_labels, model_name=None):
    ax = figure().gca()
    for i, AERs in enumerate(AER_lines):
        _ = plt.plot(AERs, label = line_labels[i])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_label_coords(0.08, 0.5)
    ax.xaxis.set_label_coords(0.85, 0.16)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    _ = plt.ylabel('AER', fontsize=20, fontstyle = 'oblique')
    _ = plt.xlabel('iteration', fontsize=20, fontstyle = 'oblique')
    plt.legend()
    if model_name:
        _ = plt.savefig(f'{model_name}_AER.png')
    _ = plt.show()

