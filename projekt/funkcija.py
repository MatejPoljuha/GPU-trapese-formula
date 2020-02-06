"""
import numpy as np
from pylab import *

def f(x):
    return abs(sin(x**x)/(2**((x**x-pi/2)/pi)))

x = linspace(0,3,10000)
plot(x, f(x), 'r-')
ylabel('y-os')
xlabel('x-os')
grid()
show()
"""

from numpy import *
from scipy import integrate
from scipy import interpolate as i
from pylab import *

def f(x):
    return abs(sin(x**x)/(2**((x**x-pi/2)/pi)))

a=0.0
b=3.0
n=500000                   #broj podsegmenata
h=(b-a)/n             #duljina podsegmenta
x=linspace(a,b,n+1)
y=f(x)


#1. nacin

#Produljena trapezna formula
iPTrap=h/2.*(y[0]+2.*sum(y[1:n])+y[n])
print('Rjesenje integrala koristeci produljenu trapeznu formulu: I*=', iPTrap)

#2. nacin

#Trapezna funkcija
iTrap=trapz(y,x)
print('Rjesenje integrala koristeci trapeznu funkciju: I*=', iTrap)


#Graf
plot(x,y,'r-')
grid()
show()
