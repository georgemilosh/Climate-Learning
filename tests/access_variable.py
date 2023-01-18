def func2():
    global a
    print(f'func2: {a = }')
    a = 3
    print(f'func2: {a = }')

def func1():
    global a
    a = 1
    print(f'func1: {a = }')
    func2()
    print(f'func1: {a = }')


func1()