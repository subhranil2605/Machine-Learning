import numpy as np
from itertools import chain

data = [
    ["Beef", "Chicken", "Milk"],
    ["Beef", "Cheese"],
    ["Cheese", "Boots"],
    ["Beef", "Chicken", "Cheese"],
    ["Beef", "Chicken", "Clothes", "Cheese", "Milk"],
    ["Chicken", "Clothes", "Milk"],
    ["Chicken", "Milk", "Clothes"]
]

##unique_items = set(list(chain(*data)))
unique_items = set([item for sublist in data for item in sublist])
print(unique_items)

matrix = [[1 if u_item in item else 0 for u_item in unique_items] for item in data]
print(np.array(matrix))


# calculate the support

