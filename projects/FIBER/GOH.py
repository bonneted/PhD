# -*- coding: utf-8 -*-
"""
Heleen Fehervary, Julie Vastmans 
Copyright 2020, Soft Tissue Mechanics group, KU Leuven

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

This function is the Gasser-Ogden-Holzapfel material model (Gasser et al. 2006). 
"""

import numpy as np

def get_parameter_boundaries():
    C10 = ### TO COMPLETE
    k1 = ### TO COMPLETE
    k2 = ### TO COMPLETE
    kappa = ### TO COMPLETE
    alpha = ### TO COMPLETE

    lower_boundaries = np.array([C10[0], k1[0], k2[0], kappa[0], alpha[0]])
    upper_boundaries = np.array([C10[1], k1[1], k2[1], kappa[1], alpha[1]])
    return(lower_boundaries, upper_boundaries)
        

def get_Cauchy_stress(parameterset,FF):
    # According to Gasser et al. 2006 (see below)
    #
    # Input:
    #       - parameterset: Array containing parameters of the model. Those are
    #       the ones searched for in the fitting. [C10 k1 k2 alpha1 alpha2 kappa] 
    #       - FF: Deformation gradient tensor [-]
    # Output:
    #       - sigma_mod: First Piola-Kirchhoff stress of the Holzapfel model [MPa]
    #
    # October 2016 Heleen Fehervary (heleen.fehervary@kuleuven.be)

    # The strain energy function W of the GOH model (Gasser et al., 2006):
    # W = C10*(I_1 - 3) + k_1/(2*k_2)*(exp[k_2*(kappa*I_1 + (1 - 3*kappa)*I_4 -1)^2] - 1);
    # Disclaimer: C10 = c1/2
    # C10 > 0: a stress-like parameter (also called mu, c) [MPa]
    # k_1 > 0: a stress-like parameter [MPa]
    # k_2 > 0: dimensionless parameter [-]
    # kappa [0,1/3]: a parameter related to the dispersion of the fibers [-]
    # alpha: the angle between the mean fiber direction and the circumferential direction [rad]
    # I_1: first invariant of the right Cauchy-Green tensor
    # I_4: invariant related to the fiber direction of a fiber family. 
    
    # Material parameters
    c10 = parameterset[0]
    k1 = [parameterset[1], parameterset[1]]
    k2 = [parameterset[2], parameterset[2]]
    kappa = [parameterset[3], parameterset[3]]
    alpha = [parameterset[4], parameterset[4]]
    
    num_fib_fam = len(k1)
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
        
        a0v = np.zeros([3,num_fib_fam])
        A0v = np.zeros([3,num_fib_fam])
        I4 = np.zeros(num_fib_fam)
        I4bar = np.zeros(num_fib_fam)
        a0m = np.zeros([num_fib_fam,3,3])
        A0m = np.zeros([num_fib_fam,3,3])
        A0mbar = np.zeros([num_fib_fam,3,3])
        A = np.zeros(num_fib_fam)
        
        for ii in range(num_fib_fam):
            a0v[:,ii]= [ np.cos(alpha[ii]), np.sin(alpha[ii]), 0] 
            # Alpha is measured wrt to the 11-direction (circumferential). 
            # Caveat: It always depends on how sample is cut and mounted in the device and what direction that is in the artery!

            I4[ii] = np.dot(a0v[:,ii],np.matmul(C,a0v[:,ii]))
            I4bar[ii]=J**(-2.0/3.0)*I4[ii]
            for i in range(3):
                for j in range(3):
                    a0m[ii,i,j]=a0v[i,ii]*a0v[j,ii]
        
            A0v[:,ii]=np.matmul(F,a0v[:,ii])
            
            for i in range(3):
                for j in range(3):
                    A0m[ii,i,j]=A0v[i,ii]*A0v[j,ii]
                    
            A0mbar[ii,:,:]=J**(-2.0/3.0)*A0m[ii,:,:]
            A[ii] = kappa[ii]*I1bar + (1.0-3.0*kappa[ii])*I4bar[ii] - 1.0
       
        
        # Matrix contribution
        sigma_mat = np.zeros([3,3])
        for i in range(3):
            for j in range(3):
                sigma_mat[i,j]=2.0*c10*J**(-1)*(Bbar[i,j]-1/3*I1bar*I[i,j])
        
        # Fiber contribution
        sigma_fiber = np.zeros([num_fib_fam,3,3])
        for ii in range(num_fib_fam):
            if(k1[ii]!=0):
                if(A[ii]>0):
                    for i in range(3):
                        for j in range(3):
                            sigma_fiber[ii,i,j]=2.0*k1[ii]*A[ii]*J**(-1.0)*np.exp(k2[ii]*A[ii]**2.0)*(kappa[ii]*(Bbar[i,j]-1.0/3.0*I1bar*I[i,j])+(1.0-3.0*kappa[ii])*(A0mbar[ii,i,j]-1.0/3.0*I4bar[ii]*I[i,j]))
        
        
        # Hydrostatic pressure
        p = np.zeros([3,3])
        tmp = 0
        for ii in range(num_fib_fam):
            tmp = tmp + sigma_fiber[ii,2,2]
        
        p = (sigma_mat[2,2] + tmp)*I 
    
       
        # Total Cauchy stress
        sigma_total = np.zeros([3,3])
        for i in range(3):
            for j in range(3):
                tmp = 0
                for ii in range(num_fib_fam):
                    tmp = tmp + sigma_fiber[ii,i,j]
                    
                sigma_total[i,j]=-p[i,j]+sigma_mat[i,j]+tmp
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
    

