import time
print('this is mod2')
time.sleep(2)

import core as ln

ln.mods.append(__name__)

class Trainer(ln.Trainer):
    def funk(self):
        print('YOOOOO')

    def uli(self):
        print('uli')
        self.teacher()

ln.Trainer = Trainer