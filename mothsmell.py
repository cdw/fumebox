#!/usr/bin/env python
# encoding: utf-8

""" odor.py
Created by Dave Williams on 2010-09-12

This defines functions that return the current 'smell' 
at any given location on any given timestep.
"""

import numpy as np
from numpy import random as ra

BOX_SIZE = (1,1,1)
TIME_STEP = 1 # ms, maybe?

def smell(loc):
    """ What's the smell here?"""
    x, y, z = loc
    y0 = BOX_SIZE[1] * 0.5
    z0 = BOX_SIZE[2] * 0.5
    return (1/x) * (1/np.hypot(y-y0, z-z0))
