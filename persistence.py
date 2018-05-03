import json
import os.path

def save_ibm1_model(lprobs, fname):
    json.dump(lprobs, open(fname,'w'))

def load_ibm1_model(fname):
    lprobs = json.load(open(fname))
    return lprobs

def save_ibm2_model(lprobs, jump_probs, fname):
    ibm2_model = {
        'lprobs' : lprobs, 
        'jump_probs' : jump_probs
    }
    json.dump(ibm2_model, open(fname,'w'))

def load_ibm2_model(fname):
    ibm2_model =  json.load(open(fname))
    return (ibm2_model['lprobs'], ibm2_model['jump_probs'])

def save(data, fname):
    with open(fname, "w") as file:
        file.write(str(data))

def load(fname):
    with open(fname, "r") as file:
        data = eval(file.readline())
    return data

def get_preprocessed_sentence_pairs():
    path_to_training = 'training_pairs.txt'
    path_to_validation = 'validation_pairs.txt'
    path_to_test = 'test_pairs.txt'
    if os.path.isfile(path_to_training) and os.path.isfile(path_to_validation) and os.path.isfile(path_to_test):
        training_pairs = persistence.load(path_to_training)
        validation_pairs = persistence.load(path_to_validation)
        test_pairs = persistence.load(path_to_test)
        return (training_pairs, validation_pairs, test_pairs)
        
    training_pairs, s_vocabulary, t_vocabulary = datasets.training_data()
    validation_pairs = datasets.validation_data(s_vocabulary, t_vocabulary)
    test_pairs = datasets.test_data(s_vocabulary, t_vocabulary)
    persistence.save(training_pairs, path_to_training)
    persistence.save(validation_pairs, path_to_validation)
    persistence.save(test_pairs, path_to_test)
    return (training_pairs, validation_pairs, test_pairs)


