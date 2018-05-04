import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator

# save figure training log likelihood vs iteration
def figure_log_likelihood(log_lhoods, fname=None):
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
    if fname:
         _ = plt.savefig(fname)
    _ = plt.show()
    
# save figure validation AER vs iteration
def figure_AER(AERs, fname = None):
    ax = figure().gca()
    _ = plt.plot(AERs)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_label_coords(0.08, 0.5)
    ax.xaxis.set_label_coords(0.85, 0.16)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    _ = plt.ylabel('AER', fontsize=20, fontstyle = 'oblique')
    _ = plt.xlabel('iteration', fontsize=20, fontstyle = 'oblique')
    if fname:
        _ = plt.savefig(fname)
    _ = plt.show()

def figure_llhood_multiple_lines(llhood_lines, line_labels, fname = None):
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
    if fname:
         _ = plt.savefig(fname)
    _ = plt.show()


def figure_AER_multiple_lines(AER_lines, line_labels, fname = None):
    ax = figure().gca()
    for i, AERs in enumerate(AER_lines):
        _ = plt.plot(AERs, label = line_labels[i])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_label_coords(0.08, 0.5)
    ax.xaxis.set_label_coords(0.85, 0.16)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    _ = plt.ylabel('AER ', fontsize=20, fontstyle = 'oblique')
    _ = plt.xlabel('iteration', fontsize=20, fontstyle = 'oblique')
    plt.legend()
    if fname:
        _ = plt.savefig(fname)
    _ = plt.show()


def figure_AER_multiple_models(model_names, aer_scores, selected_model=None, fname=None):
    ymax = max(aer_scores) + 0.06 # create some space for label

    ax = figure().gca()
    ax.yaxis.set_label_coords(0.08, 0.9)

    barlist = plt.bar(range(len(model_names)), aer_scores, align='center')
    if selected_model:
        barlist[model_names.index(selected_model)].set_color('r')
    plt.xticks(range(len(model_names)), model_names)
    plt.xticks(fontsize=20, rotation=90)
    plt.yticks(fontsize=20)
    plt.ylabel('AER', fontsize=20, fontstyle = 'oblique')
    plt.ylim(0, ymax)
    plt.tight_layout()
    if fname:
         _ = plt.savefig(fname)
    plt.show()

def figure_LL_multiple_models(model_names, ll_scores, selected_model = None, fname=None):
    ymax = max(ll_scores)  # create some space for label

    ax = figure().gca()
    ax.yaxis.set_label_coords(0.14, 0.88)
    ax.yaxis.get_offset_text().set_fontsize(20)

    barlist = plt.bar(range(len(model_names)), ll_scores, align='center')
    if selected_model:
        barlist[model_names.index(selected_model)].set_color('r')
    plt.xticks(range(len(model_names)), model_names)
    plt.xticks(fontsize=20, rotation=90)
    plt.yticks(fontsize=20)
    h = plt.ylabel('LL-hood', fontsize=20, fontstyle = 'oblique')
    h.set_rotation(0)
    plt.ylim(-15000000, -15900000)
    plt.tight_layout()
    if fname:
         _ = plt.savefig(fname)
    plt.show()


