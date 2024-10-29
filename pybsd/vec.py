import numpy as np
from jax import numpy as jnp
from collections import OrderedDict

def vectorize(X, *args):
    """
    Vectorize a numeric, list, or dictionary array in Python.
    
    Parameters:
    X : numeric, list, or dictionary
        The data to be vectorized.
    *args : additional items to be added to X if needed
    
    Returns:
    vX : np.ndarray
        Vectorized form of X.
    """
    # Initialize X and vX if there are additional arguments
    if args:
        X = [X, *args]

    # Vectorize numeric arrays
    if isinstance(X, (jnp.ndarray, np.ndarray, list, tuple, set)):
        return jnp.array(X).ravel()

    # Vectorize dictionary (struct-like) by concatenating fields
    elif isinstance(X, (dict, OrderedDict)):
        vX = []
        for key, value in X.items():
            vX.extend(vectorize(value))
        return jnp.array(vX).ravel()

    # Vectorize lists (cell-like) into numerical arrays
    elif isinstance(X, (set, list, tuple)):
        vX = []
        for item in X:
            if np.isscalar(item):
                vX.append(item)
            else:
                vX.extend(vec(item))
        return jnp.array(vX).ravel()

    # Return empty array if X does not match any expected type
    else:
        return jnp.array([])


def unvectorize(vX, *args):
    """
    Unvectorize a vectorized array in Python.
    
    Parameters:
    vX : np.ndarray or list
        The vectorized form of data.
    *args : numeric, list, or dictionary
        The target structure(s) to unvectorize into.
    
    Returns:
    varargout : list
        Unvectorized data matching the structure of each element in args.
    """
    # If multiple arguments are provided, recursively apply spm_unvec to each
    if len(args) > 1:
        return [unvectorize(vX, arg) for arg in args]

    if len(args) == 0: 
    	return vX

    # Handle the single input structure
    X = args[0]
    vX = jnp.array(vX).ravel()  # Ensure vX is a flat array
    idx = 0  # Index tracker for vX

    def _unvectorize(element):
        nonlocal idx
        if isinstance(element, (np.ndarray, jnp.ndarray, list, set, tuple)):
            n = jnp.size(element)
            result = jnp.array(vX[idx:idx + n]).reshape(jnp.shape(element))
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
            result = vX[idx]
            idx += 1
            return result
        else:
            return None

    # Apply unvectorization to X and return the result
    return unvectorize(X)