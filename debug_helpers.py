import math
import IBM1 as ibm1
import aer
import datetime


# Helper function to output likelihood and AER after each
# EM iteration
def print_likelihood(i, lprobs, log_likelihood, aer):
    likelihood = math.exp(log_likelihood)
    if i == 0:
        print('iteration    log_likelihood    AER    time')
    time_hm = datetime.datetime.now().strftime("%I:%M")
    print(f'{i} {log_likelihood:.3f} {aer:.5f} {time_hm}')

def print_ELBO(i, lprobs, elbo, aer):
    if i == 0:
        print('iteration    ELBO     AER    time')
    time_hm = datetime.datetime.now().strftime("%I:%M")
    print(f'{i} {elbo:.3f} {aer:.5f} {time_hm}')

# Helper function to output lexical probabilities (after each
# EM iteration)
def print_lexicon_probs(i, lprobs, log_likelihood, aer):
    for s in lprobs.keys():
        for t in lprobs[s].keys():
            if lprobs[s][t] > 0:
                print (s, t, lprobs[s][t])
    print()

# print translations with max probability for n target words
def print_learned_translations(lprobs, n = 100):
    split = math.ceil(len(lprobs.keys())/n)
    i = 0
    for t_word in lprobs.keys():
        if i%split == 0:
            max_sword = max(lprobs[t_word], key=lprobs[t_word].get)
            print (t_word, max_sword)
        i += 1

# print predicted and reference alignment for a given sentence pair
def print_alignment(sentence_pairs, reference_alignments, i):
    sentence_pair = sentence_pairs[i]
    predicted_alignment = ibm1.align(lprobs, sentence_pair)
    reference_alignment = reference_alignments[i:i+1]
    AER = aer.calculate_AER(reference_alignment, [predicted_alignment])
    print(sentence_pair[0])
    print(sentence_pair[1])
    print()
    print('predicted: ')
    print(predicted_alignment)
    print()
    print('sure: ')
    print(reference_alignment[0][0])
    print()
    print('possible: ')
    print(reference_alignment[0][1])
    print()
    print('AER: ')
    print(AER)
