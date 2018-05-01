import json

def save_ibm1_model(lprobs, fname):
    json.dump(lprobs, open(fname,'w'))

def load_ibm1_model(fname):
    lprobs = json.load(open(fname))
    return lprobs

def save(data, fname):
    with open(fname, "w") as file:
        file.write(str(scores))

def load(fname):
    with open(fname, "r") as file:
        data = eval(file.readline())
    return data
