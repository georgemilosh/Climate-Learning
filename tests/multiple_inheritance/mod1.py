import time
print('this is mod1')
time.sleep(2)

import core as ln
ln.mods.append(__name__)

def buga():
    print('baluga')

class Trainer(ln.Trainer):
    def teacher(self):
        print('Sensei!')

ln.buga = buga
ln.Trainer = Trainer