import collections
import math
import aer
from scipy.special import digamma
from scipy.special import gammaln
import scipy.sparse as sp
import numpy as np
import progressbar
import debug_helpers as debug

import datasets

def EM(s_t_pairs, s_vocabulary, t_vocabulary, max_iterations = 10,
        val_sentence_pairs = None, reference_alignments = None,alpha=0.1, flag_ELBO=False,fn_debug = None, fn_after_E = None, mname='ibm1_var'):
    lprobs = _initialize_lexicon_probabilities(s_vocabulary, t_vocabulary) # lexical probabilities
    i = 0
    ELBOS = []
    elbo = 0
    AERs = []
    while i < max_iterations:
        AER = 0
        log_likelihood = 0
        elbo = 0
        lambda_f_e = collections.defaultdict(lambda: collections.defaultdict(lambda:alpha)) # lambda_{f|e}
        sum_lambda_f_e = collections.defaultdict(lambda:alpha) # sum f lambda_{f|e} -- basically sum over e
        for (s_sentence, t_sentence) in s_t_pairs:
            for t_word in t_sentence:
                s_total_t = _likelihood_target_word(s_sentence, t_word, lprobs)
                log_likelihood += math.log(s_total_t)
                for s_word in s_sentence:
                    update = lprobs[s_word][t_word] / s_total_t #normalize
                    lambda_f_e[s_word][t_word] += update # M step
                    sum_lambda_f_e[s_word] += update

        if val_sentence_pairs and reference_alignments:
            predicted_alignments = align(lprobs, val_sentence_pairs)
            AER = aer.calculate_AER(reference_alignments, predicted_alignments)
            AERs.append(AER)

        # compute ELBO if flag is set to true
        # ELBO is computed by the log likelihood plus the Kullback divergence
        if flag_ELBO:
            elbo = Kullback(lambda_f_e, sum_lambda_f_e, lprobs, alpha,s_vocabulary, t_vocabulary)
            elbo += log_likelihood

        if fn_debug:
            fn_debug(i, lprobs, elbo, AER)

        if fn_after_E:
            prev_llhood = None
            prev_AER = None
            if len(ELBOS) > 1:
                prev_llhood = ELBOS[-2]
            if len(AERs) > 1:
                prev_AER = AERs[-2]
            fn_after_E(i, elbo, AER, prev_llhood, prev_AER,
                    lprobs, mname)

        ELBOS.append(elbo)

        # E step
        for s in lprobs.keys():
            for t in lprobs[s].keys():
                #denum = sum([np.exp(digamma(lambda_f_e[si][t]) - digamma(sum_lambda_f_e[si])) for si in lprobs.keys()])
                lprobs[s][t] = np.exp(digamma(lambda_f_e[s][t]) - digamma(sum_lambda_f_e[s]))  #/ denum

        i += 1

    return lprobs, ELBOS , AERs

def Kullback(lambda_f_e, sum_lambda_f_e,lprobs, alpha,s_vocabulary, t_vocabulary):
    ''' Compute Kullback-divergence according to (5) of Computing the ELBO for
    Dirichlet from Schulz
    where lamda_f_e gamma, sum_lambda_f_e sum over gamma, lprobs theta and alpha '''
    print('Computing Kullback')
    KL = 0
    for s_word in s_vocabulary:
        for t_word in t_vocabulary:
            gamma = lambda_f_e[s_word][t_word]
            KL += (np.log(lprobs[s_word][t_word]) * (alpha - gamma)) + gammaln(gamma)
            KL -= gammaln(alpha)
        KL -= gammaln(sum_lambda_f_e[s_word])
    KL += gammaln((alpha * len(t_vocabulary)))
    print('Done Kullback')
    return KL

def fname_ibm1_var(i):
    return f'ibm1_var_iter_{i}.txt'

# After each iteration we print llhood and AER and store the model
# We can later load the models that meet our convergence criteria
# and apply those to the test data
def fn_after_iter_ibm1(i, lprobs, log_likelihood, AER, alpha):
    # debug_helpers.print_ELBO(i, lprobs, log_likelihood, AER)
    if i > 0:
        persistence.save_ibm1_model(lprobs, fname_ibm1_var(i))

def log_likelihood_data(s_t_pairs, lprobs):
    return sum([log_likelihood_sentence(s_t_pair, lprobs) for s_t_pair in s_t_pairs])

def log_likelihood_sentence(s_t_pair, lprobs):
    (s_sentence, t_sentence) = s_t_pair
    return sum([ math.log(_likelihood_target_word(s_sentence, t_word, lprobs)) for t_word in t_sentence])

def _likelihood_target_word(s_sentence, t_word, lprobs):
    return sum([lprobs[s_word][t_word] for s_word in s_sentence])

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
