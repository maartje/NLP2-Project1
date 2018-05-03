import math
import collections
import aer
import warnings


def source_dependencies(s_sentence, t_word, t_pos, t_length, lprobs, jump_probs):
    s_length = len(s_sentence)    
    jump_probs = [get_jump_prob(
        s_word_pos, t_pos, s_length, t_length, jump_probs
    ) for s_word_pos in range(s_length)]
    sum_jump_probs = sum(jump_probs)
    
    return [
        (jump_probs[s_pos]/ sum_jump_probs) * lprobs[s_word][t_word] 
        for s_pos, s_word in enumerate(s_sentence)
    ]

# returns the index in the jump_probabilities list
# for given source and target positions and sentence lengths
# s_pos and s_length for source sentence including the special NULL word
# sentence positions start at index 0
def get_jump_prob_index(s_pos, t_pos, s_length, t_length, jump_probs):
    if s_pos == 0:
        return len(jump_probs) - 1
    jump = int(s_pos - math.floor((t_pos + 1) * (s_length - 1) / t_length))
    max_jump = int((len(jump_probs) - 2)/2)
    jump_prob_index = jump + max_jump
    if jump_prob_index < 0:
        warnings.warn(
            f'Jump prob index {jump_prob_index} (jump:{jump}) out of range.'
        )
        return 0 #approximate with prob of largest negative jump
    if jump_prob_index >= len(jump_probs) - 1:
        warnings.warn(
            f'Jump prob index {jump_prob_index} (jump:{jump}) out of range.'
        )
        return len(jump_probs) - 2 #approximate with prob of largest positive jump

    return jump_prob_index

# returns jump probability for given source and target positions
# and lengths. Positions start at index 0, source sentence contains
# special NULL word at position 0
def get_jump_prob(s_pos, t_pos, s_length, t_length, jump_probs):
    jump_prob_index = get_jump_prob_index(s_pos, t_pos, s_length, t_length, jump_probs)
    return jump_probs[jump_prob_index]

def align(lprobs, jump_probs, sentence_pairs):
    if isinstance(sentence_pairs, tuple):
        return _align_sentence_pair(lprobs, jump_probs, sentence_pairs)
    return [ _align_sentence_pair(lprobs, jump_probs, sentence_pair) for sentence_pair in sentence_pairs ]

def _align_sentence_pair(lprobs, jump_probs, sentence_pair):
    s_sentence = sentence_pair[0]
    t_sentence = sentence_pair[1]
    s_length = len(s_sentence)
    t_length = len(t_sentence)
    best_alignment = set()
    for t_pos, t_word in enumerate(t_sentence):
        sd = source_dependencies(s_sentence, t_word, t_pos, t_length, lprobs, jump_probs)
        (best_align_pos, _) = max(enumerate(sd), key=lambda t: t[1])
        if (best_align_pos > 0): # Leave out NULL-alignments (and alignments between unseen words)
            best_alignment.add((best_align_pos, t_pos + 1)) # word positions start at 1
    return best_alignment



def EM(s_t_pairs, lprobs, jump_probs, max_iterations = 10,
        val_sentence_pairs = None, reference_alignments = None, fn_after_E = None, mname='IBM2'):
    i = 0
    log_likelihoods = []
    AERs = []
    while i < max_iterations:
        
        # initialize
        log_likelihood = 0
        AER = 0
        counts_t_given_s = collections.defaultdict(lambda: collections.defaultdict(int))
        total_s = collections.defaultdict(int)
        jump_counts = [0]*len(jump_probs)

        # calculate counts and log likelihood
        for (s_sentence, t_sentence) in s_t_pairs:
            s_length = len(s_sentence)
            t_length = len(t_sentence)
            for t_pos, t_word in enumerate(t_sentence):
                prob_counts = source_dependencies(
                    s_sentence, t_word, t_pos, t_length, lprobs, jump_probs)
                s_total_t = sum(prob_counts)
                log_likelihood += math.log(s_total_t)
                                
                for s_pos, s_word in enumerate(s_sentence):
                    update = prob_counts[s_pos]/s_total_t
                    counts_t_given_s[s_word][t_word] += update
                    total_s[s_word] += update
                    jump_count_index = get_jump_prob_index(s_pos, t_pos, s_length, t_length, jump_probs)
                    jump_counts[jump_count_index] += update 
        
        # store log_likelihood and AER values
        log_likelihoods.append(log_likelihood)
        if val_sentence_pairs and reference_alignments:
            predicted_alignments = align(lprobs, jump_probs, val_sentence_pairs)
            AER = aer.calculate_AER(reference_alignments, predicted_alignments)
            AERs.append(AER)

        # print debug info or store models on disk
        if fn_after_E:
            prev_llhood = None
            prev_AER = None
            if len(log_likelihoods) > 1:
                prev_llhood = log_likelihoods[-2]
            if len(AERs) > 1:
                prev_AER = AERs[-2]
            fn_after_E(i, log_likelihood, AER, prev_llhood, prev_AER,
                    lprobs, jump_probs, mname)

        # update probabilities
        for s in lprobs.keys():
            for t in lprobs[s].keys():
                lprobs[s][t] = counts_t_given_s[s][t]/total_s[s]
        
        jump_count_sum = sum(jump_counts)
        jump_probs = [jc/jump_count_sum for jc in jump_counts]

        # update iteration number
        i += 1

    # add AER after final update
    if val_sentence_pairs and reference_alignments:
        predicted_alignments = align(lprobs, jump_probs, val_sentence_pairs)
        AER = aer.calculate_AER(reference_alignments, predicted_alignments)
        AERs.append(AER)
    
    # add llhood after final update
    log_likelihood = 0
    for (s_sentence, t_sentence) in s_t_pairs:
        s_length = len(s_sentence)
        t_length = len(t_sentence)
        for t_pos, t_word in enumerate(t_sentence):
            prob_counts = source_dependencies(
                s_sentence, t_word, t_pos, t_length, lprobs, jump_probs)
            s_total_t = sum(prob_counts)
            log_likelihood += math.log(s_total_t)

    return lprobs, jump_probs, log_likelihoods, AERs

