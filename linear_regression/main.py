def func(a, b, return_losses=False, return_weights=False):
    weights = {"a": a, 'b': b}
    if return_losses:
        losses = []
    for i in range(a):
        b = b * i
        if return_losses:
            losses.append(b * i)
        weights["a"] -= b
        weights["b"] -= a
    
    if return_weights:
        return losses, weights
    return None

print(func(10, 5, return_losses=False, return_weights=True))