import simple_module as sm

old_func = sm.func

def new_func(x):
    print('new_func:', x)
    print('Calling old_func')
    old_func(x)

# override
sm.func = new_func

sm.func(5)