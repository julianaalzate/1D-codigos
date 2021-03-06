# 1D-codigos
Serie de còdigos 1D por diferentes aproximaciones
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 11:02:00 2018
1D Transport equation solver for FDM with FTCS scheme.
Explicit Method
@author: Juliana Alzate
Marzo 2018
"""
# =============================================================================
# Importing libraries and external functions
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Declaration of physical variables for the problem. 
# Datos de entrada
# =============================================================================
M = 10.           # Mass injected (g)
A = 10.0          # Domain area (m2)
Dx = 0.3          # Diff coeff (m2/s)
x0 = -4.          # x min coordinate
xf = 4.           # x min coordinate
n = 500            # Nodes in the domain
Tf = 5.0          # Final time (s)
T0 = 1.           # Initial time (s)
# =============================================================================
#Soluciòn analitica de la ecuaciòn de transporte 1D
# =============================================================================

def difuana(M,A,Dx,x,t):
    
    Aux1 = A * np.sqrt(4 * np.pi * Dx * t)
    
    C = (M/Aux1) * np.exp((- x ** 2)/ (4 * Dx * t))
    
    return C

# MAIN
# =============================================================================
# Generando vector con puntos de la discretización (equiespaciados) 
# linspace Return evenly spaced numbers over a specified interval
# =============================================================================
x = np.linspace(x0,xf,n)                    # 1D array style 
#print (x)
# =============================================================================
# Informacion numerica
# =============================================================================
Sx= 0.3
L = xf - x0                                 # Domain length, distancia entre ptos
dx = x[1] - x[0]                            # Calculating spacing, can be L      
dT = Sx * ( dx ** 2) / (Dx)                 # Calculating timestep size
npt = 100                                   # Cuando disminuyo el paso de tiempo, el error baja
#int(np.ceil((Tf - T0) / (dT)))             # Number of timesteps, 
# La funciòn np.ceil redondea los resultados
#The ceil of the scalar x is the smallest integer i, such that i >= x.

# Generating vector that stores error in time
ert = np.zeros(int(npt))
time= np.zeros(int(npt))

# Generando Condiciòn Inicial
C = difuana(M,A,Dx,x,T0)
C1 = np.zeros(int(n))     
Cmax = np.max(C)
#print (C)
#print (C1)
#print (Cmax)
"""
# Plotting initial condition
style.use('ggplot')
plt.plot(x, C)
plt.title('Initial condition')
plt.xlabel(r'Distance $(m)$')
plt.ylabel(r'Concentration $ \frac{kg}{m} $')
plt.show ()
"""

# ==============================================================================
# Entering the time loop (Bucle temporal)
# ==============================================================================
for t in range(1, npt + 1):
      
   # Generating analytical solution
    Ca = difuana(M, A, Dx, x,T0 + t * dT)
           
     # Imposing boundary conditions
    C1[0] = Ca[0]
    C1[n - 1] = Ca[n - 1]
        
    # Explicit internal part
    for i in range(1, n - 1):
        
        C1[i] = Sx * C[i - 1] + (1 - 2 * Sx) * C[i] + Sx * C[i + 1]
        
    # Preparing for next timestep   
    C = C1
    
    # Estimating error
    err = np.abs(Ca - C1)
    #print (err)
    #ert[t] = t
    ert[t] = np.linalg.norm(err)
    plt.plot(x,err)
    plt.semilogy(x,err)
    plt.title('Abs Error')
    plt.semilogy(np.linspace(T0, Tf, npt), ert)
    #plt.show()
    #plt.plot (t,ert)
    #plt.semilogy()
    #plt.title('Time error')
    
    """                                            
    # Plotting numerical solution and comparison with analytical
    plt.clf()
    #plt.plot(x, C1, 'b')
    plt.plot(x, C1, '-', marker="o", markersize=2.5, Color=(1,0.2,0), label='Num ')
    plt.xlim([x0, xf])
    plt.ylim([0, Cmax])
    plt.xlabel(r'Distance $(m)$')
    plt.ylabel(r'Concentration $ \frac{kg}{m} $')
    plt.legend (loc='upper left')
    plt.title('Numerical solution')
    plt.plot(x, Ca,'b', label='Anal ')
    plt.xlim([x0, xf])
    plt.ylim([0, Cmax])
    plt.legend (loc='upper left')
    plt.title('Analytical solution vs Numerical')
    plt.show()
    """
        
