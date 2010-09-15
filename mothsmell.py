#!/usr/bin/env python
# encoding: utf-8

""" odor.py
Created by Dave Williams on 2010-09-12

This defines functions that return the current 'smell' 
at any given location on any given timestep.
"""

import numpy as np
from numpy import random.uniform as ra
from numpy import pi

BOX_SIZE = (1,1,1)
TIME_STEP = 1 # ms, maybe?

def smell(loc):
    """ What's the smell here?"""
    x, y, z = loc
    y0 = BOX_SIZE[1] * 0.5
    z0 = BOX_SIZE[2] * 0.5
    return window((1/x), 1, 0) * window((1/np.hypot(y-y0, z-z0)), 1, 0)

def window(x, up, dn):
    """Make that number fit in a window."""
    return (x>=up)*up + (x<up)*(x>dn)*x + (x<=dn)*dn

def direction_wrap(direction):
    """Wrap (th,ph) to (+-pi, +-pi)."""
    out = []
    for val in direction:
        if val > pi:
            out.append(np.mod(val,pi))
        elif val < -pi:
            out.append(np.mod(val,-pi))
        else:
            out.append(val)
    return tuple(out)

def move(location, direction, distance):
    """Move a distance in a direction from a location"""
    # TODO: Work on it

class moth(object):
    def __init__(self, starting_loc):
        self.location = starting_loc
        self.location_history = [starting_loc]
        self.direction = (pi*ra(-1, 1), pi*ra(-1, 1))
        self.direction_history = [self.direction]
    
    def decide(self):
        """Which way do you want to go?"""
        prev = smell(self.location_history[-1])
        curr = smell(self.location)
        diff = window(curr - prev, 100, 0.01)
        dire = self.direction
        self.direction_history.append(dire)
        self.direction = direction_wrap(dire(0) + pi*ra(-1, 1)*(1/diff), 
                                        dire(1) + pi*ra(-1, 1)*(1/diff))
        return self.direction
    
    def move(self):
        """Get a move on, now."""
        loc = self.location
        self.location_history.append(loc)
        loc = 
