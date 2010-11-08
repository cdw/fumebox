#!/usr/bin/env python
# encoding: utf-8

""" odor.py
Created by Dave Williams on 2010-09-12

This defines functions that return the current 'smell' 
at any given location on any given timestep.
"""

import numpy as np
from numpy.random import uniform as ra
from scipy import interpolate as spl
from numpy import pi, sin, cos
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from animation import FuncAnimation

BOX_SIZE = (1.0, 1.0, 1.0)
BODY_LEN = 0.01
ANT_ANG = np.radians(30)

class plume(object):
    def __init__(self, source=(ra(0,BOX_SIZE[1]), ra(0,BOX_SIZE[2]))):
        """A time-evolving odorant for our little guy to flap around in"""
        self.res = 100.0      #Split into 100 straight segments
        self.X = self.steps(0, BOX_SIZE[0], self.res) #X locs of plume
        self.Y = np.zeros_like(self.X) #Y of odor slices
        self.Z = np.zeros_like(self.X) #Z of same
        self.source = source #Y/Z of the source
        self.cross = np.zeros((2, len(self.X)))
        [self.update() for i in range(self.res)]
    
    def cross_breeze(self, wind_samples=5, fade = 0.99):
        """Puff! How do things get blown around?"""
        samps = self.steps(0, BOX_SIZE[0], wind_samples-1)
        for i in [0,1]:
            new_broom = spl.splrep(samps, ra(-0.1,0.1,wind_samples))
            clean_sweep = spl.splev(self.X, new_broom)
            self.cross[i,:] = clean_sweep + fade * self.cross[i,:]
    
    def blow_downstream(self, sheer_strength=0.01):
        """Huff! Move it and sheer it!"""
        self.Y[1:] = sheer_strength * self.cross[0,1:] + self.Y[:-1]
        self.Y[0] = sheer_strength * self.cross[0,0] + self.source[0]
        self.Z[1:] = sheer_strength * self.cross[1,1:] + self.Z[:-1]
        self.Z[0] = sheer_strength * self.cross[1,0] + self.source[1]
        
    def update(self):
        """Let that wind blow"""
        self.cross_breeze()
        self.blow_downstream()

    @staticmethod
    def steps(lower, upper, steps):
            """A one, and a two; get from here to there in this many"""
            l,u,s = lower, upper, steps
            return np.arange(l,u*(1+0.5*(u-l)/s), float(u-l)/s)

def smell(loc):
    """ What's the smell here?"""
    x, y, z = loc
    y0 = BOX_SIZE[1] * 0.5
    z0 = BOX_SIZE[2] * 0.5
    return window((1/x), 10, 0) * window((1/np.hypot(y-y0, z-z0)), 10, 0)

def window(x, up, dn):
    """Make that number fit in a window."""
    if x>=up:   
        return up
    elif x<=dn: 
        return dn
    else:
        return x

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

def move_some(location, direction, distance):
    """Move a distance in a direction from a location"""
    r, th, ph = distance, direction[0], direction[1]
    x, y, z, = location
    return (x + r*sin(th)*cos(th), y + r*sin(th)*sin(ph), z + r*cos(ph))

class moth(object):
    def __init__(self, starting_loc):
        self.location = starting_loc
        self.location_history = [starting_loc]
        self.direction = (pi*ra(-1, 1), pi*ra(-1, 1))
        self.direction_history = [self.direction]
    
    def __repr__(self):
        x, y, z, = self.location
        th, ph = self.direction
        return 'Moth at (%0.3f, %0.3f, %0.3f) pointing (%3d, %3d)' % \
            (x,y,z, np.degrees(th), np.degrees(ph))
    
    def antenna_smell(self):
        """What do the ol' feelers say?"""
        dire = self.direction
        loc = self.location
        ant_left, ant_right = ((dire[0]-ANT_ANG, dire[1]), 
                               (dire[0]+ANT_ANG, dire[1]))
        smell_left = smell(move_some(loc, ant_left, 0.3*BODY_LEN)) 
        smell_right = smell(move_some(loc, ant_right, 0.3*BODY_LEN)) 
        return smell_right-smell_left
    
    def decide(self):
        """Which way do you want to go?"""
        prev = smell(self.location_history[-1])
        curr = smell(self.location)
        diff = window(curr - prev, 100, 0.01)
        dire = self.direction
        self.direction_history.append(dire)
        self.direction = direction_wrap((
            dire[0] + pi*ra(-.1, .1)*self.antenna_smell(), 
            dire[1] + pi*ra(-.1, .1)*(1/diff)))
        return self.direction
    
    def move(self):
        """Get a move on, now."""
        loc = self.location
        self.location_history.append(loc)
        self.location = move_some(loc, self.direction, 0.5*BODY_LEN)
        return self.location
    
    def fly(self, time=1):
        """Fly for as long as we like"""
        for timestep in range(time):
            self.decide()
            self.move()

if __name__ == '__main__':
    # Set up a figure
    fig = plt.figure(0, figsize=(6,6))
    ax = p3.Axes3D(fig)
    lines = []
    # Set up some moths
    #moths = [moth((ra(0,1), ra(0,1), ra(0,1))) for i in range(5)]
    #for moth in moths:
    #    x, y, z = zip(*moth.location_history)
    #    lines.append(ax.plot(x,y,z)[0])
    #def update_moth(num, moths, lines):
    #    for m,l in zip(moths, lines):
    #        m.fly()
    #        x, y, z = zip(*m.location_history)
    #        l.set_data(np.array([x,y]))
    #        l.set_3d_properties(z)
    #    return lines
    # Set up a plume
    plumes = [plume((0.5,0.5))]
    for plume in plumes:
        x, y, z = plume.X, plume.Y, plume.Z
        lines.append(ax.plot(x, y, z)[0])
    def update_plume(num, plumes, lines):
        for plume, line in zip(plumes, lines):
            plume.update()
            x, y, z = plume.X, plume.Y, plume.Z
            line.set_data(np.array([x,y]))
            line.set_3d_properties(z)
        return lines
    # Make the figure nice
    ax.set_xlim3d([0.0, BOX_SIZE[0]])
    ax.set_ylim3d([0.0, BOX_SIZE[1]])
    ax.set_zlim3d([0.0, BOX_SIZE[2]])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Bouncin' around")
    #moth_ani = FuncAnimation(fig, func=update_moth, frames=100, 
    #                         fargs=(moths,lines), interval=10)
    moth_ani = FuncAnimation(fig, func=update_plume, frames=100, 
                             fargs=(plumes,lines), interval=10)
    plt.show()