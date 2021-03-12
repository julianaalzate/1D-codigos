#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 11:02:00 2018
1D Transport equation solver for FDM with FTCS scheme.
@author: Juliana Alzate
Marzo 2018
"""
# =============================================================================
# Importing libraries and external functions
# =============================================================================

import numpy as np  
import matplotlib.pyplot as plt
from matplotlib import style

# =============================================================================
# Declaration of physical variables for the problem. 
# Datos de entrada
# =============================================================================
M = 10.           # Mass injected (g)
A = 10.0          # Domain area (m2)
Dx = 0.3          # Diff coeff (m2/s)
x0 = -4.          # x min coordinate
xf = 4.           # x min coordinate
n = 50            # Nodes in the domain
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
npt = int(np.ceil((Tf - T0) / (dT)))        # Number of timesteps, 
# La funciòn np.ceil redondea los resultados
#The ceil of the scalar x is the smallest integer i, such that i >= x.
# Generating vector that stores error in time
ert = np.zeros(int(npt))

# Generando Condiciòn Inicial
C = difuana(M,A,Dx,x,T0)
C1 = np.zeros(int(n))     
Cmax = np.max(C)
#print (C)
#print (C1)
#print (Cmax)

# Plotting initial condition
style.use('ggplot')
plt.plot(x, C)
plt.title('Initial condition')
plt.xlabel(r'Distance $(m)$')
plt.ylabel(r'Concentration $ \frac{kg}{m} $')

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
        
   #Imposing boundary conditions
    C[0] = Ca[0]
    C[n - 1] = Ca[n - 1]
                   
    # Estimating error
    err = np.abs(Ca - C1)
    ert[t] = np.linalg.norm(err)
    
    # Plotting numerical solution and comparison with analytical
    plt.clf()
    plt.plot(x, C1, 'b')
    plt.xlim([x0, xf])
    plt.ylim([0, Cmax])
    plt.ylabel(r'Concentration $ \frac{kg}{m} $')
    plt.title('Numerical solution')
    plt.plot(x, Ca)
    plt.xlim([x0, xf])
    plt.ylim([0, Cmax])
    plt.title('Analytical solution')
    plt.show()
    
# Preparing for next timestep   
C = C1
