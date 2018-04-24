import collections
import math

def EM(s_t_pairs, s_vocabulary, t_vocabulary, fn_after_iter = None):
    lprobs = _initialize_lexicon_probabilities(s_vocabulary, t_vocabulary)
    i = 1
    while i <= 20:
        # initialize
        log_likelihood = 0
        counts_t_given_s = collections.defaultdict(lambda: collections.defaultdict(int))
        total_s = collections.defaultdict(int)
        for (s_sentence, t_sentence) in s_t_pairs:
            for t_word in t_sentence:
                # normalization factor
                s_total_t = sum([lprobs[s_word][t_word] for s_word in s_sentence])
                log_likelihood += math.log(s_total_t)
                for s_word in s_sentence:
                    update = lprobs[s_word][t_word]/s_total_t
                    counts_t_given_s[s_word][t_word] += update
                    total_s[s_word] += update
        for s in lprobs.keys():
            for t in lprobs[s].keys():
                lprobs[s][t] = counts_t_given_s[s][t]/total_s[s]
        if fn_after_iter:
            fn_after_iter(i, lprobs, log_likelihood)
        i += 1
    return lprobs

# {Hause: { book:0.25, ...}, ...}
# read: the probability of 'book' given 'Haus' is 0.25
def _initialize_lexicon_probabilities(source_vocabulary, target_vocabulary):
    p_init = 1./len(target_vocabulary)
    lexicon_probabilities = collections.defaultdict(
        lambda: collections.defaultdict(lambda: p_init))
    return lexicon_probabilities

def _print_lexicon_probs(lprobs):
    for s in lprobs.keys():
        for t in lprobs[s].keys():
            if lprobs[s][t] > 0:
                print (s, t, lprobs[s][t])
