"""Various helpful functions"""

import numpy as np
from numba import njit, objmode

# Function that approximates the integral of the normalized Planck
# function over the range x = h nu/kT. Clark, JCP vol 70. no 2, p 311
#@njit
def normalizedPlanckIntegral(x1, x2, T = 1.0 ):
    x1 = x1/T
    x2 = x2/T
    if( x2 < 0.01 ):
    #    use small x approximation: Clark, JCP 70, no. 2 1987, p 316, eq 32
    #    Integral[x1,x2] = Integral[0,x2] - Integral[0,x1]
        a1 = 0.05132991127342032; ## 1/3/(pi**4/15)
        a2 = 0.01924871672753262; ## 1/8/(pi**4/15)

        return x2*x2*x2*(a1 - a2*x2) - x1*x1*x1*(a1 - a2*x1)

    else:
    #   Clark, JCP 70, no. 2 1987, p 316, eq 48
        a1 = 1.2807339766120354
        a2 = 0.8578722513311724
        a3 = 0.33288908614428098
        a4 = 0.079984931563508915
        a5 = 0.011878558806454416

        b1 = 0.2807339758744
        b2 = 0.07713864107538

    #  Integral[x1,x2] = Integral[x1,Infinity] - Integral[x2,Infinity]
        Int_x1_Inf = \
            np.exp(-x1)*(1.0+x1*(a1+x1*(a2+x1*(a3+x1*(a4+x1*a5)))))/(1.+x1*(b1+x1*b2))

        Int_x2_Inf = \
            np.exp(-x2)*(1.0+x2*(a1+x2*(a2+x2*(a3+x2*(a4+x2*a5)))))/(1.+x2*(b1+x2*b2))

    #  Integral[x1,x2] = Integral[x1,Infinity] - Integral[x2,Infinity]
        return Int_x1_Inf - Int_x2_Inf
    

#  spectral energy density of Planckian at given temperature and
#  frequency, but integrated over the group to get a group average
#  energy density. The function returns B(nu, T) integrated over 4pi
#  and divided by c to get energy density, then integrated over the
#  group. and divided by the group width. 
#  assumes T and nu in keV; result is erg/cm^3/keV.
#@njit
def PlanckianEnergyDensityAverage( T, nu1, nu2 ):
   
    if T <= 0.0:
        print("T = ", T)
    if nu1 <= 0.0:
        print("nu1 = ", nu1)
    if nu1 >= nu2:
        print("nu1 = ", nu1)
        print("nu2 = ", nu2)
 
    # a in ergs/cm**3/keV^4
    a = 1.3720165e14
   
    baverage = normalizedPlanckIntegral( nu1, nu2, T )
 
# the extra T is because the integral is over x = nu/T, not nu
# so we are dividing the integral by dx not dnu to get the average
    dx = (nu2 - nu1)/T
    baverage /= dx
   
    return a*T*T*T*baverage

# function Fnu used in solution of Graziani's analytic problem
# r = t = point you want solution at
# sigma = opacity of outer sphere at the desired frequency
# R = radius of inner sphere
# c = speed of light
 
def FnuSlab( d, t, sigma, c,
             useOldE1 = False ):
 
    from math import exp
    if useOldE1:
        from E1 import E1
    else:
        # exponential integral function E1
        import scipy.special as sc
 
    if d <= 0:
        print("Error: d = ", d)
    if t <= 0:
        print("Error: t = ", t)
    if c <= 0:
        print("Error: c = ", c)
    if sigma <= 0:
        print("Error: sigma = ", sigma)
   
    ct = c*t
 
    if ct < d:
        return 0.0
 
#    light has had time to reach point at distance d from slab
    term1 = exp( -sigma*d )
 
    term2 = -d/ct * exp( -sigma*ct )
 
    if useOldE1:
        term3 = -sigma*d*( E1(sigma*d) - E1(sigma*ct) )
    else:
        term3 = -sigma*d*( sc.exp1(sigma*d) - sc.exp1(sigma*ct) )
 
##    print "  "
##    print "term1 = ", term1
##    print "term2 = ", term2
##    print "E1(sigma*d) = ", E1(sigma*d)
##    print "E1(sigma*ct) = ", E1(sigma*ct)
##    print "term3 = ", term3
##    print "  "
 
    return 0.5*( term1 + term2 + term3 )

# average energy density in jrk/cm^3/keV over the given frequency
# range at position d in medium at T0 with opacity sigma outside of
# plane at x = 0 with temperature Tsource at time t
 
def energyDensityAverage( d, t, nu1, nu2, sigma, Tsource, T0, c,
                          useOldE1 = False ):
 
    eInitial = PlanckianEnergyDensityAverage(T0, nu1, nu2)
 
    delta = PlanckianEnergyDensityAverage(Tsource, nu1, nu2) - eInitial
 
    return eInitial + delta*FnuSlab( d, t, sigma, c,
                                     useOldE1 )
 

from sympy import symbols, solve

def get_equilibrium_temperature(Tr, Tm, a, b):
    """This function returns the equilibrium temperature if the radiation and material start at
      different temperatures. Applicable for an infinite medium"""
    
    if a == 0 and b == 0:
        raise ValueError("Both coefficients 'a' and 'b' cannot be zero.")
    
    E0 = a * Tr ** 4
    U0 = b * Tm

    # Define the variable
    T = symbols('T')

    # Define the quartic equation
    equation = a * T ** 4 + b * T - (E0 + U0)

    # Solve the equation
    solutions = solve(equation, T)
    
    # Filter for real, positive solutions and return the smallest
    real_positive_solutions = [sol.evalf() for sol in solutions if sol.is_real and sol > 0]

    return min(real_positive_solutions, default=None)  # Return None if no valid solution exists