import random
import persistence
import math
import collections

# returns list with jump probabilities 
# (and special NULL jump probability)
# [-max_jump ....0 ...max_jump, NULL_jump]
def initialize_jump_probs_uniformly(sentence_pairs):
    max_jump = 0
    for (s_sentence, t_sentence) in sentence_pairs:
        s_length = len(s_sentence) - 1 #ignore NULL
        t_length = len(t_sentence)
        jump_1 = abs(1 - math.floor(s_length))
        jump_2 = abs(s_length - math.floor(s_length/t_length))
        max_jump = max(jump_1, jump_2, max_jump)
    init_prob = 1. / (2*max_jump + 1 + 1) # last item is special NULL jump prob
    return [init_prob] * (2*max_jump + 1 + 1)

def initialize_jump_probs_randomly(sentence_pairs):
    max_jump = 0
    for (s_sentence, t_sentence) in sentence_pairs:
        s_length = len(s_sentence) - 1 #ignore NULL
        t_length = len(t_sentence)
        jump_1 = abs(1 - math.floor(s_length))
        jump_2 = abs(s_length - math.floor(s_length/t_length))
        max_jump = max(jump_1, jump_2, max_jump)
    probs_length = 2*max_jump + 1 + 1 # last item is special NULL jump prob
    randoms = [random.random() for i in range(probs_length)]
    sum_randoms = sum(randoms)
    return [r/sum_randoms for r in randoms]

# {Hause: { book:0.25, ...}, ...}
# read: the probability of 'book' given 'Haus' is 0.25
def initialize_lprobs_uniform(s_t_pairs):
    return _initialize_lprobs(s_t_pairs, lambda: 1.)

def initialize_lprobs_randomly(s_t_pairs):
    return _initialize_lprobs(s_t_pairs, random.random)

def initialize_lprobs_staged(s_t_pairs):
    return persistence.load_ibm1_model('IBM1_output/params_AER_5.txt')

def _initialize_lprobs(s_t_pairs, fn_set_prob):

    # find all word combinations in sentence pairs
    s_t_combinations = ((s_word, t_word) 
          for (s_sentence, t_sentence) in s_t_pairs 
          for s_word in s_sentence 
          for t_word in t_sentence)

    # set random probabilities for word combinations
    lprobs = collections.defaultdict(lambda: collections.defaultdict(lambda: float('NaN')))
    for (s, t) in s_t_combinations:
        lprobs[s][t] = fn_set_prob()

    # normalize probabilities
    for s_key, t_dict in lprobs.items():
        total = sum(t_dict.values())
        for (t_key, r) in t_dict.items():
            t_dict[t_key] = t_dict[t_key] / total
    return lprobs
