import numpy as np

v1 = 6
v2 = 7

a1 = 4
a2 = 5

b = 0.4
g = 0.3

out = v1*v2*(b*np.cos(a1 - a2) - g*np.sin(a1 - a2))

print(out)