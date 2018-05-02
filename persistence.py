import json

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
