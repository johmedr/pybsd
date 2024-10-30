import numpy as np
from jax import numpy as jnp
from collections import OrderedDict


def vectorize(*args):
    """
    Vectorize a numeric, list, or dictionary array in Python.
    
    Parameters:
    x : numeric, list, or dictionary
        The data to be vectorized.
    *args : additional items to be added to x if needed
    
    Returns:
    v : np.ndarray
        Vectorized form of x.
    """
    x = args[0] if len(args) == 1 else args

    # Vectorize numeric arrays
    if isinstance(x, (jnp.ndarray, np.ndarray, list, tuple, set)):
        return jnp.array(x).ravel()

    # Vectorize dictionary (struct-like) by concatenating fields
    elif isinstance(x, (dict, OrderedDict)):
        v = []
        for key, value in x.items():
            v.extend(vectorize(value))
        return jnp.array(v).ravel()

    # Vectorize lists (cell-like) into numerical arrays
    elif isinstance(x, (set, list, tuple)):
        v = []
        for item in x:
            if np.isscalar(item):
                v.append(item)
            else:
                v.extend(vectorize(item))
        return jnp.array(v).ravel()

    # Return empty array if x does not match any expected type
    else:
        return jnp.array([])


def unvectorize(v, x):
    """
    Unvectorize a vectorized array in Python.
    
    Parameters:
    v : np.ndarray or list
        The vectorized form of data.
    x : numeric, list, or dictionary
        The target structure(s) to unvectorize into.
    
    Returns:
        Unvectorized data matching the structure of x.
    """
    # Handle the single input structure
    v = jnp.array(v).ravel()  # Ensure vX is a flat array
    idx = 0  # Index tracker for vX

    def _unvectorize(element):
        nonlocal idx
        if isinstance(element, (np.ndarray, jnp.ndarray, list, set, tuple)):
            n = jnp.size(element)
            result = jnp.array(v[idx:idx + n]).reshape(jnp.shape(element))
            idx += n
            return result
        elif isinstance(element, (dict, OrderedDict)):
            result = type(element)()
            for key in element:
                result[key] = _unvectorize(element[key])
            return result
        elif isinstance(element, (set, list, tuple)):
            return type(element)(_unvectorize(item) for item in element)
        elif np.isscalar(element):
            result = v[idx]
            idx += 1
            return result
        else:
            return None

    # Apply unvectorization to x and return the result
    return _unvectorize(x)
