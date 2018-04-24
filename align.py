def align(lprobs, sentence_pair):
    s_sentence = sentence_pair[0]
    t_sentence = sentence_pair[1]
    best_alignment = set()
    for j, t_word in enumerate(t_sentence):
        best_align_prob = -1
        best_align_pos = -1
        for i, s_word in enumerate(s_sentence):
            align_prob = lprobs[s_word][t_word] #p(t|s)
            if align_prob >= best_align_prob:
                best_align_pos = i
                best_align_prob = align_prob
        if (best_align_pos != 0): # Leave out NULL-alignments
            best_alignment.add((j + 1, best_align_pos)) # word positions start at 1
    return best_alignment

