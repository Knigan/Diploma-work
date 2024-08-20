import numpy as np

def f(arr1, arr2):
    if len(arr1) != len(arr2):
        return False
    
    for i in range(len(arr1)):
        if arr1[i] != arr2[i]:
            return False
    
    return True


seed = 100000
np.random.seed(seed)

arr_control = [11.517513082786476,11.377023812789837,10.581328419129363,
8.930361341729288,10.26029512452561,10.182246819327965,
10.196601915788328,9.34215603659727,8.971535216541797,
8.63517513026982]

arr = [np.random.normal(10, 1) for _ in range(10)]

while not f(arr, arr_control):
    seed += 1
    np.random.seed(seed)
    arr = [np.random.normal(10, 1) for _ in range(10)]

print(seed)

