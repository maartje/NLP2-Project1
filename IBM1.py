import collections
import math
import aer

def EM(s_t_pairs, s_vocabulary, t_vocabulary, max_iterations = 10,
        val_sentence_pairs = None, reference_alignments = None, fn_debug = None):
    lprobs = _initialize_lexicon_probabilities(s_vocabulary, t_vocabulary)
    i = 0
    log_likelihoods = []
    AERs = []
    while i < max_iterations:
        # initialize
        log_likelihood = 0
        AER = 0
        counts_t_given_s = collections.defaultdict(lambda: collections.defaultdict(int))
        total_s = collections.defaultdict(int)

        # calculate counts and log likelihood
        for (s_sentence, t_sentence) in s_t_pairs:
            for t_word in t_sentence:
                # normalization factor
                s_total_t = _likelihood_target_word(s_sentence, t_word, lprobs)
                log_likelihood += math.log(s_total_t)
                for s_word in s_sentence:
                    update = lprobs[s_word][t_word]/(s_total_t * len(s_sentence))
                    counts_t_given_s[s_word][t_word] += update
                    total_s[s_word] += update
        
        # store log_likelihood and AER values
        log_likelihoods.append(log_likelihood)
        if val_sentence_pairs and reference_alignments:
            predicted_alignments = align(lprobs, val_sentence_pairs)
            AER = aer.calculate_AER(reference_alignments, predicted_alignments)
            AERs.append(AER)

        # print debug info
        if fn_debug:
            fn_debug(i, lprobs, log_likelihood, AER)

        # update probabilities
        for s in lprobs.keys():
            for t in lprobs[s].keys():
                lprobs[s][t] = counts_t_given_s[s][t]/total_s[s]

        # update iteration number
        i += 1
    return lprobs, log_likelihoods, AERs

def log_likelihood_data(s_t_pairs, lprobs):
    return sum([log_likelihood_sentence(s_t_pair, lprobs) for s_t_pair in s_t_pairs])

def log_likelihood_sentence(s_t_pair, lprobs):
    (s_sentence, t_sentence) = s_t_pair
    return sum([ math.log(_likelihood_target_word(s_sentence, t_word, lprobs)) for t_word in t_sentence])

def _likelihood_target_word(s_sentence, t_word, lprobs):
    return sum([lprobs[s_word][t_word] for s_word in s_sentence]) / len(s_sentence)

def align(lprobs, sentence_pairs):
    if isinstance(sentence_pairs, tuple):
        return _align_sentence_pair(lprobs, sentence_pairs)
    return [ _align_sentence_pair(lprobs, sentence_pair) for sentence_pair in sentence_pairs ]

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

def _align_sentence_pair(lprobs, sentence_pair):
    s_sentence = sentence_pair[0]
    t_sentence = sentence_pair[1]
    best_alignment = set()
    for j, t_word in enumerate(t_sentence):
        best_align_prob = -1
        best_align_pos = -1
        for i, s_word in enumerate(s_sentence):
            if s_word not in lprobs.keys() or t_word not in lprobs[s_word].keys():
                continue # ignore unseen source and target words
            align_prob = lprobs[s_word][t_word] #p(t|s)
            if align_prob >= best_align_prob:
                best_align_pos = i
                best_align_prob = align_prob
        if (best_align_pos > 0): # Leave out NULL-alignments (and alignments between unseen words)
            best_alignment.add((best_align_pos, j + 1)) # word positions start at 1
    return best_alignment

