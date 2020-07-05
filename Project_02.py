# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 07:10:39 2020

@author: Saumya Dholakia
"""

#Input parameters
import numpy as np
L=1
a=1
n=1
eps=10**(-30)
tend= (L/a)/4
dx= 0.025
x = np.arange(0,L+dx,dx)
zeta = x-a*tend
nx = np.size(x)

#The Lax-Werendoff without limiter algorithm
class lax_w():
    def solve(self,x,uinitial,c,a,tend):
        import numpy as np
        dx = x[1]-x[0]
        dt = c*dx/a
        nx = np.size(x) 
        uold = np.zeros(nx)
        unew = np.zeros(nx)
        t = np.arange(0,tend,dt)
        uold = np.copy(uinitial)
        for j in t:
            for i in range(1,nx-1):
                unew[i] =uold[i]-c*(uold[i+1]-uold[i-1])/2 +(c*c)*(uold[i+1]-2*uold[i]+uold[i-1])/2
            unew[0] =uold[0]-c*(uold[1]-uold[nx-2])/2 +(c*c)*(uold[1]-2*uold[0]+uold[nx-2])/2
            unew[nx-1] = unew[0]
            for i in range(nx):
                uold[i]=unew[i]
        return unew
    
#The Upwind method   
class upwind():
    def solve(self,x,uinitial,c,a,tend):
        import numpy as np
        dx = x[1]-x[0]
        dt = c*dx/a
        nx = np.size(x)
        uold = np.zeros(nx)
        unew = np.zeros(nx)
        t = np.arange(0,tend,dt)
        uold = np.copy(uinitial)
        for j in t:
            for i in range(1,nx):
                unew[i]=uold[i] - c*(uold[i]-uold[i-1])
            unew[0] = uold[0] - c*(uold[0]-uold[nx-2])
            for i in range(nx):
                uold[i]=unew[i]
        return unew
    
#The Lax-Werendoff with the limiter algorithm
class lax_limiter():
    def solve(self,x,uinitial,c,a,tend,eps):
        import numpy as np
        dx = x[1]-x[0]
        dt = c*dx/a
        nx = np.size(x) 
        uold = np.zeros(nx)
        unew = np.zeros(nx)
        t = np.arange(0,tend,dt)
        uold = np.copy(uinitial)
        for j in t:
            for i in range(2,nx-1):
                rleft = (uold[i-1]-uold[i-2])/(uold[i]-uold[i-1]+eps)
                rright = (uold[i]-uold[i-1])/(uold[i+1]-uold[i]+eps)
                phileft = (abs(rleft)+rleft)/(1+abs(rleft))
                phiright = (abs(rright)+rright)/(1+abs(rright))
                Fleft = a*uold[i-1]+a*((1-c)/2)*(uold[i]-uold[i-1])*phileft
                Fright = a*uold[i]+a*((1-c)/2)*(uold[i+1]-uold[i])*phiright
                unew[i] =uold[i]-(Fright-Fleft)*(dt/dx)
            rleft0 = (uold[nx-2]-uold[nx-3])/(uold[0]-uold[nx-2]+eps)
            rright0 = (uold[0]-uold[nx-2])/(uold[1]-uold[0]+eps)
            rleft1= (uold[0]-uold[nx-1])/(uold[1]-uold[0]+eps)
            rright1 = (uold[1]-uold[0])/(uold[2]-uold[1]+eps)
            phileft0 = (abs(rleft0)+rleft0)/(1+abs(rleft0))
            phiright0 = (abs(rright0)+rright0)/(1+abs(rright0))
            phileft1 = (abs(rleft1)+rleft1)/(1+abs(rleft1))
            phiright1 = (abs(rright1)+rright1)/(1+abs(rright1))
            Fleft0 = a*uold[nx-2]+a*((1-c)/2)*(uold[0]-uold[nx-2])*phileft0
            Fright0 = a*uold[0]+a*((1-c)/2)*(uold[1]-uold[0])*phiright0
            Fleft1= a*uold[0]+a*((1-c)/2)*(uold[1]-uold[0])*phileft1
            Fright1 = a*uold[1]+a*((1-c)/2)*(uold[2]-uold[1])*phiright1
            unew[0] =uold[0]-(Fright0-Fleft0)*(dt/dx)
            unew[1] =uold[1]-(Fright1-Fleft1)*(dt/dx)
            unew[nx-1] = unew[0]
            for i in range(nx):
                uold[i]=unew[i]
        return unew

#The Superbee algorithm with the LW method  
class lax_superbee():
    def solve(self,x,uinitial,c,a,tend,eps):
      import numpy as np
      dx = x[1]-x[0]
      dt = c*dx/a
      nx = np.size(x) 
      uold = np.zeros(nx)
      unew = np.zeros(nx)
      t = np.arange(0,tend,dt)
      uold = np.copy(uinitial)
      for j in t:
          for i in range(2,nx-1):
              rleft = (uold[i-1]-uold[i-2])/(uold[i]-uold[i-1]+eps)
              rright = (uold[i]-uold[i-1])/(uold[i+1]-uold[i]+eps)
              phileft=np.max([0,np.min([1,2*rleft]),np.min([rleft,2])])
              phiright=np.max([0,np.min([1,2*rright]),np.min([rright,2])])
              Fleft = a*uold[i-1]+a*((1-c)/2)*(uold[i]-uold[i-1])*phileft
              Fright = a*uold[i]+a*((1-c)/2)*(uold[i+1]-uold[i])*phiright
              unew[i] =uold[i]-(Fright-Fleft)*(dt/dx)
              
          #Boundary conditions
          #At node 0
          rleft0 = (uold[nx-2]-uold[nx-3])/(uold[0]-uold[nx-2]+eps)
          rright0 = (uold[0]-uold[nx-2])/(uold[1]-uold[0]+eps)
          phileft0=np.max([0,np.min([1,2*rleft0]),np.min([rleft0,2])])
          phiright0=np.max([0,np.min([1,2*rright0]),np.min([rright0,2])])
          Fleft0 = a*uold[nx-2]+a*((1-c)/2)*(uold[0]-uold[nx-2])*phileft0
          Fright0 = a*uold[0]+a*((1-c)/2)*(uold[1]-uold[0])*phiright0
          unew[0] =uold[0]-(Fright0-Fleft0)*(dt/dx)
          
          #At node 1
          rleft1= (uold[0]-uold[nx-1])/(uold[1]-uold[0]+eps)
          rright1 = (uold[1]-uold[0])/(uold[2]-uold[1]+eps)
          phileft1=np.max([0,np.min([1,2*rleft1]),np.min([rleft1,2])])
          phiright1=np.max([0,np.min([1,2*rright1]),np.min([rright1,2])])
          Fleft1= a*uold[0]+a*((1-c)/2)*(uold[1]-uold[0])*phileft1
          Fright1 = a*uold[1]+a*((1-c)/2)*(uold[2]-uold[1])*phiright1
          unew[1] =uold[1]-(Fright1-Fleft1)*(dt/dx)
          
          #At node 40
          unew[nx-1] = unew[0]
          
          for i in range(nx):
              uold[i]=unew[i]
              
      return unew
  
#PART 1
#Creating dictionaries
clist = dict()
#clist[0.1] = 0.1
clist[0.5] = 0.5
#clist[0.9] = 0.9
#clist[1.0] =1.0
colors = dict()
#colors[0.1]='red'
colors[0.5]='blue'
#colors[0.9]='green'
#colors[1.0]='yellow'
  
#Initial condition and Exact solution for the sine function
uinitial = np.sin(2.*np.pi*n*x)
uexact = np.sin(2.*np.pi*n*zeta)
  
#Solver 1 LW method (Sine and Top hat)
solver1 = lax_w()
#Solver 2 Upwind method (Sine and Top hat)
solver2 = upwind()
#Solver 3 LW with limiter (Sine and Top hat)
solver3 = lax_limiter()
#Solver 4 LW with superbee (Sine and Top hat)
solver4 = lax_superbee()

u_LW_sine = dict()
u_upwind_sine = dict()
u_LW_limiter_sine = dict()
u_LW_superbee_sine = dict()
u_LW_top_hat = dict()
u_LW_top_hat_1 = dict()
u_upwind_top_hat = dict()
u_LW_limiter_top_hat = dict()
u_LW_superbee_top_hat = dict()

for c in clist:
    u_LW_sine[c] = solver1.solve(x,uinitial,c,a,tend)
    u_upwind_sine[c] = solver2.solve(x,uinitial,c,a,tend)
    u_LW_limiter_sine[c] = solver3.solve(x,uinitial,c,a,tend,eps)
    u_LW_superbee_sine[c] = solver4.solve(x,uinitial,c,a,tend,eps)
    
#Plots - u-x plot (Sine function)
import matplotlib.pyplot as plt
f, ax = plt.subplots(1,1,figsize=(8,5))
ax.plot(x,uinitial,label='Initial condition',color='black',)
ax.plot(x,uexact,label='Exact solution',color='black', linestyle='--')
for c in clist:
    ax.plot(x,u_LW_sine[c],label='LW',color='red')
    ax.plot(x,u_upwind_sine[c],label='Upwind',color='blue') 
    ax.plot(x,u_LW_limiter_sine[c],label='LW_limiter',color='green') 
    ax.plot(x,u_LW_superbee_sine[c],label='LW_superbee',color='black') 
ax.set_xlabel('$x$',size=20)
ax.set_ylabel('$u(x,t)$',size=20)
ax.grid()
ax.legend(fontsize=16)
plt.savefig('LW_sine.png',bbox_inches='tight')
plt.savefig('Upwind_sine.png',bbox_inches='tight')
plt.savefig('LW_limiter_sine.png',bbox_inches='tight')
plt.savefig('LW_superbee_sine.png',bbox_inches='tight')

#Initial condition and Exact solution for the top hat function
uinitial = np.heaviside(x,1.)-np.heaviside(x-L/2,1.)
uexact = np.heaviside(zeta,1.)-np.heaviside(zeta-L/2,1.)

for c in clist:
    u_LW_sine[c] = solver1.solve(x,uinitial,c,a,tend)
    u_upwind_sine[c] = solver2.solve(x,uinitial,c,a,tend)
    u_LW_limiter_sine[c] = solver3.solve(x,uinitial,c,a,tend,eps)
    u_LW_superbee_sine[c] = solver4.solve(x,uinitial,c,a,tend,eps)

#Plots - u-x plot (Top hat function)
import matplotlib.pyplot as plt
f, ax = plt.subplots(1,1,figsize=(8,5))
ax.plot(x,uinitial,label='Initial condition',color='black',)
ax.plot(x,uexact,label='Exact solution',color='black', linestyle='--')
for c in clist:
    ax.plot(x,u_LW_sine[c],label='LW',color='red')
    ax.plot(x,u_upwind_sine[c],label='Upwind',color='blue') 
    ax.plot(x,u_LW_limiter_sine[c],label='LW_limiter',color='green') 
    ax.plot(x,u_LW_superbee_sine[c],label='LW_superbee',color='black') 
ax.set_xlabel('$x$',size=20)
ax.set_ylabel('$u(x,t)$',size=20)
ax.grid()
ax.legend(fontsize=16)
plt.savefig('LW_top_hat.png',bbox_inches='tight')
plt.savefig('Upwind_top_hat.png',bbox_inches='tight')
plt.savefig('LW_limiter_top_hat.png',bbox_inches='tight')
plt.savefig('LW_superbee_top_hat.png',bbox_inches='tight')

#PART 2
#Creating dictionaries
clist = dict()
clist[0.1] = 0.1
#clist[0.5] = 0.5
#clist[0.9] = 0.9
#clist[1.0] =1.0
colors = dict()
colors[0.1]='red'
#colors[0.5]='blue'
#colors[0.9]='green'
#colors[1.0]='yellow'

#Initial condition and Exact solution for the sine function
uinitial = np.sin(2.*np.pi*n*x)
uexact = np.sin(2.*np.pi*n*zeta)

for c in clist:
    u_LW_sine[c] = solver1.solve(x,uinitial,c,a,tend)
    u_upwind_sine[c] = solver2.solve(x,uinitial,c,a,tend)
    u_LW_limiter_sine[c] = solver3.solve(x,uinitial,c,a,tend,eps)
    u_LW_superbee_sine[c] = solver4.solve(x,uinitial,c,a,tend,eps)
    
#Plots - u-x plot (Sine function)
import matplotlib.pyplot as plt
f, ax = plt.subplots(1,1,figsize=(8,5))
ax.plot(x,uinitial,label='Initial condition',color='black',)
ax.plot(x,uexact,label='Exact solution',color='black', linestyle='--')
for c in clist:
    ax.plot(x,u_LW_sine[c],label='LW',color='red')
    ax.plot(x,u_upwind_sine[c],label='Upwind',color='blue') 
    ax.plot(x,u_LW_limiter_sine[c],label='LW_limiter',color='green') 
    ax.plot(x,u_LW_superbee_sine[c],label='LW_superbee',color='black')  
ax.set_xlabel('$x$',size=20)
ax.set_ylabel('$u(x,t)$',size=20)
ax.grid()
ax.legend(fontsize=16)
plt.savefig('LW_sine.png',bbox_inches='tight')
plt.savefig('Upwind_sine.png',bbox_inches='tight')
plt.savefig('LW_limiter_sine.png',bbox_inches='tight')
plt.savefig('LW_superbee_sine.png',bbox_inches='tight')

#Initial condition and Exact solution for the top hat function
uinitial = np.heaviside(x,1.)-np.heaviside(x-L/2,1.)
uexact = np.heaviside(zeta,1.)-np.heaviside(zeta-L/2,1.)

for c in clist:
    u_LW_top_hat[c] = solver1.solve(x,uinitial,c,a,tend)
    u_upwind_top_hat[c] = solver2.solve(x,uinitial,c,a,tend)
    u_LW_limiter_top_hat[c] = solver3.solve(x,uinitial,c,a,tend,eps)
    u_LW_superbee_top_hat[c] = solver4.solve(x,uinitial,c,a,tend,eps)

#Plots - u-x plot (Top hat function)
import matplotlib.pyplot as plt
f, ax = plt.subplots(1,1,figsize=(8,5))
ax.plot(x,uinitial,label='Initial condition',color='black',)
ax.plot(x,uexact,label='Exact solution',color='black', linestyle='--')
for c in clist:
    ax.plot(x,u_LW_top_hat[c],label='LW',color='red')
    ax.plot(x,u_upwind_top_hat[c],label='Upwind',color='blue') 
    ax.plot(x,u_LW_limiter_top_hat[c],label='LW_limiter',color='green') 
    ax.plot(x,u_LW_superbee_top_hat[c],label='LW_superbee',color='black') 
ax.set_xlabel('$x$',size=20)
ax.set_ylabel('$u(x,t)$',size=20)
ax.grid()
ax.legend(fontsize=16)
plt.savefig('LW_top_hat.png',bbox_inches='tight')
plt.savefig('Upwind_top_hat.png',bbox_inches='tight')
plt.savefig('LW_limiter_top_hat.png',bbox_inches='tight')
plt.savefig('LW_superbee_top_hat.png',bbox_inches='tight')

#Creating dictionaries
clist = dict()
#clist[0.1] = 0.1
#clist[0.5] = 0.5
clist[0.9] = 0.9
#clist[1.0] =1.0
colors = dict()
#colors[0.1]='red'
#colors[0.5]='blue'
colors[0.9]='green'
#colors[1.0]='yellow'

#Initial condition and Exact solution for the sine function
uinitial = np.sin(2.*np.pi*n*x)
uexact = np.sin(2.*np.pi*n*zeta)

for c in clist:
    u_LW_sine[c] = solver1.solve(x,uinitial,c,a,tend)
    u_upwind_sine[c] = solver2.solve(x,uinitial,c,a,tend)
    u_LW_limiter_sine[c] = solver3.solve(x,uinitial,c,a,tend,eps)
    u_LW_superbee_sine[c] = solver4.solve(x,uinitial,c,a,tend,eps)
    
#Plots - u-x plot (Sine function)
import matplotlib.pyplot as plt
f, ax = plt.subplots(1,1,figsize=(8,5))
ax.plot(x,uinitial,label='Initial condition',color='black',)
ax.plot(x,uexact,label='Exact solution',color='black', linestyle='--')
for c in clist:
    ax.plot(x,u_LW_sine[c],label='LW',color='red')
    ax.plot(x,u_upwind_sine[c],label='Upwind',color='blue') 
    ax.plot(x,u_LW_limiter_sine[c],label='LW_limiter',color='green') 
    ax.plot(x,u_LW_superbee_sine[c],label='LW_superbee',color='black') 
ax.set_xlabel('$x$',size=20)
ax.set_ylabel('$u(x,t)$',size=20)
ax.grid()
ax.legend(fontsize=16)
plt.savefig('LW_sine.png',bbox_inches='tight')
plt.savefig('Upwind_sine.png',bbox_inches='tight')
plt.savefig('LW_limiter_sine.png',bbox_inches='tight')
plt.savefig('LW_superbee_sine.png',bbox_inches='tight')

#Initial condition and Exact solution for the top hat function
uinitial = np.heaviside(x,1.)-np.heaviside(x-L/2,1.)
uexact = np.heaviside(zeta,1.)-np.heaviside(zeta-L/2,1.)

for c in clist:
    u_LW_top_hat[c] = solver1.solve(x,uinitial,c,a,tend)
    u_upwind_top_hat[c] = solver2.solve(x,uinitial,c,a,tend)
    u_LW_limiter_top_hat[c] = solver3.solve(x,uinitial,c,a,tend,eps)
    u_LW_superbee_top_hat[c] = solver4.solve(x,uinitial,c,a,tend,eps)

#Plots - u-x plot (Top hat function)
import matplotlib.pyplot as plt
f, ax = plt.subplots(1,1,figsize=(8,5))
ax.plot(x,uinitial,label='Initial condition',color='black',)
ax.plot(x,uexact,label='Exact solution',color='black', linestyle='--')
for c in clist:
    ax.plot(x,u_LW_top_hat[c],label='LW',color='red')
    ax.plot(x,u_upwind_top_hat[c],label='Upwind',color='blue') 
    ax.plot(x,u_LW_limiter_top_hat[c],label='LW_limiter',color='green') 
    ax.plot(x,u_LW_superbee_top_hat[c],label='LW_superbee',color='black') 
ax.set_xlabel('$x$',size=20)
ax.set_ylabel('$u(x,t)$',size=20)
ax.grid()
ax.legend(fontsize=16)
plt.savefig('LW_top_hat.png',bbox_inches='tight')
plt.savefig('Upwind_top_hat.png',bbox_inches='tight')
plt.savefig('LW_limiter_top_hat.png',bbox_inches='tight')
plt.savefig('LW_superbee_top_hat.png',bbox_inches='tight')

#PART3
dx= 0.0125
x1 = np.arange(0,L+dx,dx)
zeta1 = x1-a*tend


#Creating dictionaries
clist = dict()
#clist[0.1] = 0.1
clist[0.5] = 0.5
#clist[0.9] = 0.9
#clist[1.0] =1.0
colors = dict()
#colors[0.1]='red'
colors[0.5]='blue'
#colors[0.9]='green'
#colors[1.0]='yellow'

#Initial condition and Exact solution for the top hat function
uinitial = np.heaviside(x,1.)-np.heaviside(x-L/2,1.)
uinitial1 = np.heaviside(x1,1.)-np.heaviside(x1-L/2,1.)
uexact = np.heaviside(zeta,1.)-np.heaviside(zeta-L/2,1.)
uexact1 = np.heaviside(zeta1,1.)-np.heaviside(zeta1-L/2,1.)

for c in clist:
    u_LW_top_hat[c] = solver1.solve(x,uinitial,c,a,tend)
    u_LW_top_hat_1[c] = solver1.solve(x1,uinitial1,c,a,tend)

#Plots - u-x plot (Top hat function)
import matplotlib.pyplot as plt
f, ax = plt.subplots(1,1,figsize=(8,5))
ax.plot(x,uinitial,label='Initial condition',color='black',)
ax.plot(x,uexact,label='Exact solution',color='black', linestyle='--')

for c in clist:
    ax.plot(x,u_LW_top_hat[c],label='LW_top_hat_with_dx=0.025',color='red')
    ax.plot(x1,u_LW_top_hat_1[c],label='LW_top_hat_with_dx=0.0125',color='blue')
ax.set_xlabel('$x$',size=20)
ax.set_ylabel('$u(x,t)$',size=20)
ax.grid()
ax.legend(fontsize=16)
plt.savefig('LW_top_hat_part_3.png',bbox_inches='tight')

f, ax = plt.subplots(1,1,figsize=(8,5))
for c in clist:
    ax.plot(x,u_LW_top_hat[c]-uexact,label='Error_with_dx=0.025',color='red')
    ax.plot(x1,u_LW_top_hat_1[c]-uexact1,label='Error_with_dx=0.0125',color='blue')
ax.set_xlabel('$x$',size=20)
ax.set_ylabel('$error(x,t)$',size=20)
ax.grid()
ax.legend(fontsize=16)
plt.savefig('LW_top_hat_part_3.png',bbox_inches='tight')

#PART4
#Initial condition and Exact solution for the sine function
c =  0.5
DX = [0.1,0.05,0.025,0.0125,0.00625,0.003125,0.0015625]
nsize = np.size(DX)
eps1 = np.zeros(nsize)
eps2 = np.zeros(nsize)
eps3 = np.zeros(nsize)
eps4 = np.zeros(nsize)
LX = np.zeros(nsize)
for i in range(nsize):
    e1 = 0
    e2 = 0
    e3 = 0
    e4 = 0
    dx = DX[i]
    x = np.arange(0,L+dx,dx)
    uinitial = np.sin(2.*np.pi*n*x)
    zeta = x-a*tend
    nx = np.size(x)
    u1 = np.zeros(nx)
    u2= np.zeros(nx)
    u3 = np.zeros(nx)
    u4 = np.zeros(nx)
    uexact = np.sin(2.*np.pi*n*zeta)
    u_LW_sine = solver1.solve(x,uinitial,c,a,tend)
    u_upwind_sine = solver2.solve(x,uinitial,c,a,tend)
    u_LW_limiter_sine = solver3.solve(x,uinitial,c,a,tend,eps)
    u_LW_superbee_sine = solver4.solve(x,uinitial,c,a,tend,eps)
    for j in range(nx):
        u1[j] = u_LW_sine[j]-uexact[j]
        u2[j] = u_upwind_sine[j]-uexact[j]
        u3[j] = u_LW_limiter_sine[j]-uexact[j] 
        u4[j]= u_LW_superbee_sine[j]-uexact[j]
        e1 = e1 + u1[j]**2
        e2 = e2 + u2[j]**2
        e3 = e3 + u3[j]**2
        e4 = e4 + u4[j]**2
    eps1[i] = (e1**0.5)/nx
    eps2[i] = (e2**0.5)/nx
    eps3[i] = (e3**0.5)/nx
    eps4[i] = (e4**0.5)/nx
    LX[i] = np.log10(1/dx)
    
f, ax = plt.subplots(1,1,figsize=(8,5))
ax.plot(LX,eps1,label='LW_sine',color='black')
ax.plot(LX,eps2,label='LW_upwind_sine',color='red')
ax.plot(LX,eps3,label='LW_limiter_sine',color='blue')
ax.plot(LX,eps4,label='LW_superbee_sine',color='green')
ax.set_xlabel('$log(1/dx)$',size=20)
ax.set_ylabel('$log(L2_Norm)$',size=20)
ax.grid()
ax.legend(fontsize=16)
plt.savefig('PART4.png',bbox_inches='tight')

#For the top hat function
c =  0.5
DX = [0.1,0.05,0.025,0.0125,0.00625,0.003125,0.0015625]
nsize = np.size(DX)
eps1 = np.zeros(nsize)
eps2 = np.zeros(nsize)
eps3 = np.zeros(nsize)
eps4 = np.zeros(nsize)
LX = np.zeros(nsize)
for i in range(nsize):
    e1 = 0
    e2 = 0
    e3 = 0
    e4 = 0
    dx = DX[i]
    x = np.arange(0,L+dx,dx)
    uinitial = np.heaviside(x,1.)-np.heaviside(x-L/2,1.)
    zeta = x-a*tend
    nx = np.size(x)
    u1 = np.zeros(nx)
    u2= np.zeros(nx)
    u3 = np.zeros(nx)
    u4 = np.zeros(nx)
    uexact = np.heaviside(zeta,1.)-np.heaviside(zeta-L/2,1.)
    u_LW_top_hat = solver1.solve(x,uinitial,c,a,tend)
    u_upwind_top_hat = solver2.solve(x,uinitial,c,a,tend)
    u_LW_limiter_top_hat = solver3.solve(x,uinitial,c,a,tend,eps)
    u_LW_superbee_top_hat = solver4.solve(x,uinitial,c,a,tend,eps)
    for j in range(nx):
        u1[j] = u_LW_top_hat[j]-uexact[j]
        u2[j] = u_upwind_top_hat[j]-uexact[j]
        u3[j] = u_LW_limiter_top_hat[j]-uexact[j] 
        u4[j]= u_LW_superbee_top_hat[j]-uexact[j]
        e1 = e1 + u1[j]**2
        e2 = e2 + u2[j]**2
        e3 = e3 + u3[j]**2
        e4 = e4 + u4[j]**2
    eps1[i] = (e1**0.5)/nx
    eps2[i] = (e2**0.5)/nx
    eps3[i] = (e3**0.5)/nx
    eps4[i] = (e4**0.5)/nx
    LX[i] = np.log10(1/dx)
    
f, ax = plt.subplots(1,1,figsize=(8,5))
ax.plot(LX,eps1,label='LW_top_hat',color='black')
ax.plot(LX,eps2,label='LW_upwind_top_hat',color='red')
ax.plot(LX,eps3,label='LW_limiter_top_hat',color='blue')
ax.plot(LX,eps4,label='LW_superbee_top_hat',color='green')
ax.set_xlabel('$log(1/dx)$',size=20)
ax.set_ylabel('$log(L2_Norm)$',size=20)
ax.grid()
ax.legend(fontsize=16)
plt.savefig('PART4.png',bbox_inches='tight')    




