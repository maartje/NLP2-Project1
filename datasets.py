def training_data(s_path, t_path):
    french_sentences = list(_read_sentences(s_path))
    english_sentences = list(_read_sentences(t_path))
    return _process_sentences(french_sentences, english_sentences)

def example_data():
    german_sentences = [['das', 'Haus'], ['das', 'Buch'], ['ein', 'Buch']]
    english_sentences = [['the', 'house'], ['the', 'book'], ['a', 'book']]
    return _process_sentences(german_sentences, english_sentences, add_null_words = False)

def example_data_null_words():
    german_sentences = [['das', 'Haus'], ['das', 'Buch'], ['ein', 'Buch'], ['ein', 'Haus']]
    english_sentences = [['the', 'house'], ['the', 'book'], ['a', 'book'], ['a', 'small', 'house']]
    return _process_sentences(german_sentences, english_sentences)

def _read_sentences(fpath):
    with open(fpath, 'r') as lines:
        for line in lines:
            yield line.split()

def _process_sentences(s_sentences, t_sentences, add_null_words = True):
    if add_null_words:
        s_sentences = _add_null_words(s_sentences)
    s_vocabulary = _build_vocabulary(s_sentences)
    t_vocabulary = _build_vocabulary(t_sentences)
    s_t_pairs = list(zip(s_sentences, t_sentences))
    return s_t_pairs, s_vocabulary, t_vocabulary

def _add_null_words(sentences):
    return [['<NULL>'] + s for s in sentences]

def _build_vocabulary(sentences):
    return list({word for sentence in sentences for word in sentence})



