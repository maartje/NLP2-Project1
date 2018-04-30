import json

def store_ibm1_model(lprobs, fname):
    json.dump(lprobs, open(fname,'w'))

def load_ibm1_model(fname):
    lprobs = json.load(open(fname))
    return lprobs
