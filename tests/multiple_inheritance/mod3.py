import time
print('this is mod3')
time.sleep(2)

import mod2
import mod1

ln = mod2.ln

mod_name = __name__
if __name__ == '__main__':
    mod_name += f':{__file__}'

ln.mods.append(mod_name)

print(ln.mods)

trainer = ln.Trainer()
trainer.funk()
trainer.uli()

ln.buga()