import json

def store_ibm1_model(lprobs, fname):
    json.dump(lprobs, open(fname,'w'))

def load_ibm1_model(fname):
    lprobs = json.load(open(fname))
    return lprobs

def store_iteration_results(scores, fname):
    with open(fname, "w") as file:
        file.write(str(scores))

def load_iteration_results(fname):
    with open(fname, "r") as file:
        scores = eval(file.readline())
    return scores
