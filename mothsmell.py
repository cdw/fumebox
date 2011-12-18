#!/usr/bin/env python
# encoding: utf-8

""" odor.py
Created by Dave Williams on 2010-09-12

This defines functions that return the current 'smell' 
at any given location on any given timestep.
"""

import numpy as np
from numpy.random import uniform as ra
from numpy.random import normal as norm
from scipy import interpolate as spl
from numpy import pi, sin, cos
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from animation import FuncAnimation

BOX_SIZE = (1.0, 1.0, 1.0)
BODY_LEN = 0.01
ANT_ANG = np.radians(30)
SOURCE_WANDERS = True

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


class plume(object):
    def __init__(self, source=(ra(0,BOX_SIZE[1]), ra(0,BOX_SIZE[2]))):
        """A time-evolving odorant for our little guy to flap around in"""
        self.res = 100.0      #Split into 100 straight segments
        self.X = self.steps(0, BOX_SIZE[0], self.res) #X locs of plume
        self.Y = np.zeros_like(self.X) #Y of odor slices
        self.Z = np.zeros_like(self.X) #Z of same
        self.source = list(source) #Y/Z of the source
        self.original = source #Initial source location
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
    
    def source_wander(self, scale = BODY_LEN):
        """Where is that smell coming from?"""
        if SOURCE_WANDERS is True:
            self.source = [s+norm(scale=scale) for s in self.source]
            if any((self.source[0]<0, self.source[0]>BOX_SIZE[0],
                    self.source[0]<0, self.source[0]>BOX_SIZE[0])):
                self.source_wander(scale)
    
    def update(self):
        """Let that wind blow"""
        self.source_wander()
        self.cross_breeze()
        self.blow_downstream()
    
    def _spline_yz(self, x):
        """Return interpolated values for plume's y/z loc at x"""
        i = self.X.searchsorted(x)
        #x_samples = self.X[i-2:i+3]
        #y_samples = self.Y[i-2:i+3]
        #z_samples = self.Z[i-2:i+3]
        #y_spline = spl.splrep(x_samples, y_samples)
        #z_spline = spl.splrep(x_samples, z_samples)
        #y, z = spl.splev(x, y_spline), spl.splev(x, z_spline)
        y,z = self.Y[i], self.Z[i]
        return y,z
    
    def smell(self, loc):
        """Sniff, sniff... what's that?"""
        x, y, z = loc
        if 0>x or x>BOX_SIZE[0] or 0>y or y>BOX_SIZE[1] \
               or 0>z or z>BOX_SIZE[2]:
            print "Moth sniffing outside the box"
            return 0.0
        Y, Z = self._spline_yz(x)
        dist = np.hypot(Y-y, Z-z)
        return 1/np.sqrt(4*pi*x) * np.exp(-dist**2/(4*x))
        
    @staticmethod
    def steps(lower, upper, steps):
            """A one, and a two; get from here to there in this many"""
            l,u,s = lower, upper, steps
            return np.arange(l,u*(1+0.5*(u-l)/s), float(u-l)/s)


class moth(object):
    def __init__(self, starting_loc, plume=plume([0.5,0.5])):
        self.location = starting_loc
        self.location_history = [starting_loc]
        self.direction = (pi*ra(-1, 1), pi*ra(-1, 1))
        self.direction_history = [self.direction]
        self.plume = plume
    
    def __repr__(self):
        x, y, z, = self.location
        th, ph = self.direction
        return 'Moth at (%0.3f, %0.3f, %0.3f) pointing (%3d, %3d)' % \
            (x,y,z, np.degrees(th), np.degrees(ph))
    
    def check_loc(self, loc=None):
        if loc is None: loc = self.location
        x, y, z = loc
        if 0>x or x>BOX_SIZE[0] or 0>y or y>BOX_SIZE[1] or 0>z or z>BOX_SIZE[2]:
            print "Location is outside box, moth is pulling a phoenix"
            self.__init__((ra(0,1), ra(0,1), ra(0,1)), plume=self.plume)
    
    def sniff(self, loc):
        """This fellow goes forth and sniffs the smell"""
        #return smell(loc)
        return self.plume.smell(loc)
    
    def antenna_smell(self):
        """What do the ol' feelers say?"""
        dire = self.direction
        loc = self.location
        ant_left, ant_right = ((dire[0]-ANT_ANG, dire[1]), 
                               (dire[0]+ANT_ANG, dire[1]))
        smell_left = self.sniff(move_some(loc, ant_left, 0.3*BODY_LEN)) 
        smell_right = self.sniff(move_some(loc, ant_right, 0.3*BODY_LEN)) 
        return smell_right-smell_left
    
    def decide(self):
        """Which way do you want to go?"""
        prev = self.sniff(self.location_history[-1])
        curr = self.sniff(self.location)
        diff = window(curr - prev, 100, 0.01)
        dire = self.direction
        self.direction_history.append(dire)
        self.direction = direction_wrap((
            dire[0] + pi*ra(-.01, .01)*self.antenna_smell(), 
            dire[1] + pi*ra(-.01, .01)*(1/diff)))
        for pt in self.direction:
            assert np.isnan(pt) == False
        return self.direction
    
    def move(self):
        """Get a move on, now."""
        loc = self.location
        self.location_history.append(loc)
        self.location = move_some(loc, self.direction, 0.5*BODY_LEN)
        self.check_loc(self.location)
        for xyz in self.location:
            assert np.isnan(xyz) == False
        return self.location
    
    def fly(self, time=1):
        """Fly for as long as we like"""
        for timestep in range(time):
            self.decide()
            self.move()

def animate_moth(moth, num_of_moths=3):
    # Set up a figure
    fig = plt.figure(0, figsize=(6,6))
    ax = p3.Axes3D(fig)
    lines = []
    # Set up some moths
    moths = [moth((ra(0,1), ra(0,1), ra(0,1))) for i in range(num_of_moths)]
    for moth in moths:
        x, y, z = zip(*moth.location_history)
        lines.append(ax.plot(x,y,z)[0])
    def update_moth(num, moths, lines):
        for m,l in zip(moths, lines):
            m.fly()
            x, y, z = zip(*m.location_history)
            l.set_data(np.array([x,y]))
            l.set_3d_properties(z)
        return lines
    # Make the figure nice
    ax.set_xlim3d([0.0, BOX_SIZE[0]])
    ax.set_ylim3d([0.0, BOX_SIZE[1]])
    ax.set_zlim3d([0.0, BOX_SIZE[2]])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Bouncin' around")
    moth_ani = FuncAnimation(fig, func=update_moth, frames=100, 
                             fargs=(moths,lines), interval=10)
    plt.show()

def animate_moth_and_plume(moth, plume, num_of_moths=1):
    # Set up a figure
    fig = plt.figure(0, figsize=(6,6))
    ax = p3.Axes3D(fig)
    lines = []
    # Set up a plume
    plume = plume([0.5, 0.5])
    # Set up some moths
    moth_locs = lambda:(ra(0,1), ra(0,1), ra(0,1))
    moths = [moth(moth_locs(), plume) for i in range(num_of_moths)]
    for moth in moths:
        x, y, z = zip(*moth.location_history)
        lines.append(ax.plot(x,y,z)[0])
    x, y, z = plume.X, plume.Y, plume.Z
    lines.append(ax.plot(x, y, z)[0])
    def update(num, moths, plume, lines):
        plume.update()
        for m,l in zip(moths, lines[:-1]):
            m.fly()
            x, y, z = zip(*m.location_history)
            l.set_data(np.array([x,y]))
            l.set_3d_properties(z)
        x, y, z = plume.X, plume.Y, plume.Z
        lines[-1].set_data(np.array([x,y]))
        lines[-1].set_3d_properties(z)
        return lines
    # Make the figure nice
    ax.set_xlim3d([0.0, BOX_SIZE[0]])
    ax.set_ylim3d([0.0, BOX_SIZE[1]])
    ax.set_zlim3d([0.0, BOX_SIZE[2]])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Bouncin' around")
    moth_ani = FuncAnimation(fig, func=update, frames=100, 
                             fargs=(moths, plume, lines), interval=10)
    plt.show()

def animate_plume(plume):
    # Set up a figure
    fig = plt.figure(0, figsize=(6,6))
    ax = p3.Axes3D(fig)
    lines = []
    # Set up a plume
    plumes = [plume([0.5,0.5])]
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
    plume_ani = FuncAnimation(fig, func=update_plume, frames=100, 
                             fargs=(plumes,lines), interval=10)
    plt.show()

if __name__ == '__main__':
    animate_moth_and_plume(moth, plume, 5)
    #animate_plume(plume)
