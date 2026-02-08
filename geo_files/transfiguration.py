#transfiguration file | created by Swastik Abhishek Singh
#Part of the GEOBORN library
# Licensed under the MIT license

import jax 
import jax.numpy as jnp

def stretch(state, xmult, ymult, zmult):

    x, y, z = state
    xc *= xmult
    yc *= ymult
    zc *= zmult

    shape = (xc, yc, zc)
    return shape
