import core as ln

ln.mods.append(__name__)

def buga():
    print('baluga')

class Trainer(ln.Trainer):
    def teacher(self):
        print('Sensei!')

ln.buga = buga
ln.Trainer = Trainer