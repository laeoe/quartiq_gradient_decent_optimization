import numpy as np

dic = {}
dic["a"] = (1, 3)
dic["b"] = [1, 2, 3, 4]
print(np.random.choice(dic["b"]))
print(np.random.uniform(dic["a"][0], dic["a"][1]))

