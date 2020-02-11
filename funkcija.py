from numpy import *
from scipy import integrate
from scipy import interpolate as i
from pylab import *

def f(x):
    return sin(x)

a = 0.0
b = 3.14
n = 200048                #broj podsegmenata
h = (b-a)/n             #duljina podsegmenta
x = linspace(a,b,n+1)
y = f(x)


#1. nacin

#Produljena trapezna formula
iPTrap = h/2.*(y[0]+2.*sum(y[1:n])+y[n])
print('Rjesenje integrala koristeci produljenu trapeznu formulu: I*=', iPTrap)

#2. nacin

#Trapezna funkcija
iTrap = trapz(y,x)
print('Rjesenje integrala koristeci trapeznu funkciju: I*=', iTrap)

#Graf
plot(x,y,'r-')
grid()
show()
