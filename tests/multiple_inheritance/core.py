import numpy as np

mods = []

def train_model(model, X, Y):
    model.fit(X, Y)
    return model

def buga():
    print('buga')

class Trainer():
    def __init__(self):
        pass

    def say(self, msg):
        print(msg)