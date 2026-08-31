# -*- coding: utf-8 -*-
"""
Heleen Fehervary, Julie Vastmans 
Copyright 2020, Soft Tissue Mechanics group, KU Leuven

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

This function is the NeoHookean material model. 
"""

import numpy as np

def get_parameter_boundaries():
    C10 = np.array([0, 10])
    
    lower_boundaries = np.array([C10[0]])
    upper_boundaries = np.array([C10[1]])
    return(lower_boundaries, upper_boundaries)

def get_initial_starting_points(num_starting_points, lower_boundaries, upper_boundaries):
    pars_range = upper_boundaries - lower_boundaries
    for n in range(num_starting_points):
        x0 = [1/num_starting_points*n+pars_range[0]]
        
    x0 = [1/nSP*n+lbbounds[0],1/nSP*n+lbbounds[0],1/nSP*n+lbbounds[0],1/nSP*n+lbbounds[0],1/nSP*n+lbbounds[0]]

    return(x0)
    
def get_Cauchy_stress(parameterset,FF):
    # Input:
    #       - parameterset: Array containing parameters of the model. Those are
    #       the ones searched for in the fitting. [C10 k1 k2 alpha1 alpha2 kappa] 
    #       - FF: Deformation gradient tensor [-]
    # Output:
    #       - sigma_mod: First Piola-Kirchhoff stress of the Holzapfel model [MPa]
    #
    # October 2016 Heleen Fehervary (heleen.fehervary@kuleuven.be)

    # The strain energy function W of the NH model:
    # W = C10*(I_1 - 3) 
    # Disclaimer: C10 = c1/2
    # C10 > 0: a stress-like parameter (also called mu, c) [MPa]
    
    # Material parameters
    c10 = parameterset[0]
    
    n = len(FF)
    sigma_mod = np.zeros([n,3,3])
    for t in range(n):
        F = FF[t,:,:]
        J = np.linalg.det(F)
        
        I = np.identity(3)
        C = np.matmul(np.transpose(F),F)
        B = np.matmul(F,np.transpose(F))
        Bbar = J**(-2.0/3.0)*B
        I1 = np.trace(C)
        I1bar = J**(-2.0/3.0)*I1
                
        # Matrix contribution
        sigma_mat = np.zeros([3,3])
        for i in range(3):
            for j in range(3):
                sigma_mat[i,j]=2.0*c10*J**(-1)*(Bbar[i,j]-1/3*I1bar*I[i,j])        
        
        # Hydrostatic pressure
        p = np.zeros([3,3])
        p = sigma_mat[2,2]*I 
    
        # Total Cauchy stress
        sigma_total = np.zeros([3,3])
        for i in range(3):
            for j in range(3):
                sigma_total[i,j]= -p[i,j] + sigma_mat[i,j]
                
        sigma_mod[t,:,:] = sigma_total
    
    return(sigma_mod)

def get_1PK_stress(parameterset,FF):
    # Get Cauchy stress
    sigma_modd = get_Cauchy_stress(parameterset, FF)

    # Convert to 1PK
    n = len(FF)
    P_mod = np.zeros([n,3,3])
    for t in range(n):
        F = FF[t,:,:]
        J = np.linalg.det(F)
        sigma_mod = sigma_modd[t,:,:]
        P_mod[t,:,:]=### TO COMPLETE
    
    return(P_mod)
        

def get_RF_mod(parameterset,FF,A0):
    # Get 1PK stress
    P_modd = get_1PK_stress(parameterset, FF)

    # Convert to RF
    n = len(FF)
    RF_mod = np.zeros([n,3,3])
    for t in range(n):
        P_mod = P_modd[t,:,:]
        RF_mod[t,:,:]=P_mod*A0
    
    return(RF_mod)
    