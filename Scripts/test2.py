import numpy as np



sar = list(range(0, 10, 2))
sar2 = [x for x in range(0, 10, 2)]
sar3 = np.arange(0, 10, 3)
sar3 = list(sar3) + [10]
print(sar)
print(sar2)
print(sar3)

