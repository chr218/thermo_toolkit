# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:12:47 2020

This program calculates the entropy, enthalpy, and Gibb's free energy of 
adsorbate & transition states within their ideal-gas/adsorbed states.

This code is based off of "Peng's" code.

The necessary input files are VASP
'OUTCAR' and 'CONTCAR'.

The user is given the choice of approximations for calculating the entropy
of adsorbed molecules/transition states. 

1) The harmonic oscillator-[https://aiche.onlinelibrary.wiley.com/doi/10.1002/aic.16838]
2) 2D free translator-[https://doi.org/10.1021/jp106536m, https://doi.org/10.1016/j.micromeso.2019.04.058],

@author: cvr5246@gmail.com
"""

#import matplotlib.pyplot as plt
import numpy as np
import sys

from tamkin import *
from molmod import centimeter, lightspeed
from ase.io  import vasp
from scipy import constants,optimize




'''
constants
'''
Hartree_2_Joule = (constants.physical_constants['Hartree energy'])
R = constants.physical_constants['molar gas constant']
kb = constants.physical_constants['Boltzmann constant']
pi = constants.pi
Na = constants.physical_constants['Avogadro constant']
h = constants.physical_constants['Planck constant']
ev = constants.physical_constants['electron volt']
c = constants.physical_constants['speed of light in vacuum']
amu  = constants.physical_constants['atomic mass constant']
invcm = lightspeed/centimeter


# =============================================================================
# Subroutines & Functions
# =============================================================================

'''
'get_U0' pulls the ground state energy (ZPE corrected and non ZPEcorrected) 
from the VASP relaxation and returns the value in kJ/mol.

CAUTION!
There are subtle differences in the energy from the freq calc (IBRION= 5) OUTCAR
and a standard relaxation (IBRION = 2) OUTCAR.

Ensure that you are parsing the correct OUTCAR!

'''
def get_U0(freqs):
    
    with open('OSZICAR_relax','r') as f:
        last_line = f.readlines()[-1]
    U0 = float(last_line.split()[4])*ev[0]*Na[0]/1000    

    ZPE = 0
    for freq in freqs:
        ZPE += 0.5*h[0]*c[0]*100*freq/invcm
        
    ZPE = ZPE/1.602E-19*ev[0]*Na[0]/1000    

    return U0, ZPE, (U0+ZPE)


'''
"sort_atoms" sorts the atoms into unconstrained and constrained lists as indices.
This function returns the constrained atom indices, unconstrained atom indices
and the total mass of the unconstrained atoms.
'''
def sort_atoms(atms_obj):
    atms = atms_obj.copy()
    
    #Sort atom indices into constrained and unconstrained. 
    if len(atms._get_constraints())>0:
        const = (atms._get_constraints())[0].get_indices()
    else: const = []

    unconst = [int(x) for x in range(len(atms)) if x not in const]

    mass_unconst = 0
    for atom in atms:
        if atom.index in unconst:
            mass_unconst += atom.mass
     
    #Get Moment of inertia of unconstrained atoms. (Find better way to do this!)
    del atms[[atom.index for atom in atms if atom.index in const]]
    I = atms.get_moments_of_inertia(vectors=False)#Units are [amu*ang**2]    
    I *= amu[0]*((1.0e-10)**2)#Convert to [m2*kg]
    
    return const,unconst,mass_unconst, I
        

'''
"replace_freq_cutoff" replaces the frequencies (obtained by the nma object) 
below a certain cutoff. 

The input frequencies should not be in wavenumbers!

The input cutoff should be in wavenumbers!
'''
def replace_freq_cutoff(nma_obj,cutoff):
    for i,fr in enumerate(nma_obj.freqs):
        if fr == 0:
            continue
        elif fr < cutoff*invcm:
            nma_obj.freqs[i] = cutoff*invcm
            
    return nma_obj


'''
"Thermo_2D_ZSA" calculates the entropy of a 2D translator constrained within
the zeolite's surface area. The "Modes_list" labels each mode as 'trans',
'rot' or 'vib'. Frequencies below the cutoff are replaced. 

The ensemble is NPT 'Gibbs'

This function returns the enthalpy, heat capacity, Gibb's free energy, chemical potential
and 2D translator entropy.
'''

def S_2D_ZSA(Modes_list,nma_obj,T,mass,cutoff,sno,I):
    
    #Calculate 2D translational entropy constrained to adsorption site surface area.
    CsaZeo = 1/(2E-10*6E-10)#Surface area of zeolite-De Moor et al. 'Ads of C2-C8 n-alkanes in zeolites'
    S2d_t = R[0]*((np.log((2*pi*mass*kb[0]*T/(h[0]**2))/CsaZeo))+2) # 2D Translational contribution, mol constrained to SA of zeolite.

    
    #Calculate 1D & 2D & 3D rotational contribution. NOTE: must correctly identify which principal moments apply! (i.e cartwheel vs. helicopter rotations)
    if sno == None:#Symmetry number.
        print('Warning: Unspecified symmetry number.')
        sno = 1   
    S3d_r = R[0]*(np.log(((8*(pi**2)*kb[0]*T/(h[0]**2))**(3/2))*np.sqrt(pi*I[0]*I[1]*I[2])/sno)+(3/2))
    S2d_r = R[0]*((np.log((8*(pi**2)*kb[0]*T/(h[0]**2))*np.sqrt(pi*I[0]*I[1])/sno))+1)
    S1d_r = R[0]*(np.log(np.sqrt(8*(pi**2)*kb[0]*T/(h[0]**2))*np.sqrt(pi*I[0])/sno)+(1/2))


    #Remove translational & rotational frequencies.    
    for i,mode in enumerate(Modes_list):
        if mode == 'trans' or mode =='TST':#Set modes labelled as 'trans' to 0. They will be omitted from Svib.
            nma_obj.freqs[i] = 0
        elif mode == 'rot':#To do
            nma_obj.freqs[i] = 0
        elif mode == 'vib':
            continue
        else:
            print('Error: Unknown mode type within Modes_list object.') 
        
    pf_vib = PartFun(nma_obj, [])#Construct Harmonic Oscillator partition function.
    
    Svib = pf_vib.entropy(T)*Hartree_2_Joule[0]*Na[0] # Vibrational Entropy [J/mol/K]
    St = Svib + S2d_t#Total entropy
    
    return St
 

def Shomate_S(param,T,S_data):
    t = T/1000.0
    S = param[0]*np.log(t) + param[1]*t + param[2]*t*t/2 + param[3]*t*t*t/3 - param[4]/(2*t*t) + param[5]
    Sdiff = S - S_data
    return Sdiff


def Shomate_H(param, T):
    t = T/1000.0
    H = param[0]*t + param[1]*t**2/2 + param[2]*t**3/3 + param[3]*t**4/4 - param[4]/t + param[5]
    return H

'''
'get_T_corrections' calculates the temperature corrections necessary for calculating
the enthalpy.

H = U0 + ZPE + T_corrections

The temperature corrections are calculated by computing a list of entropies
across a temperature range, which is fit to Shomate parameters.-function 'Shomate_S'.

The fit allows us to etimate the magnitude of each parameter. A seperate
Shomate parameter function allows us to estimate the enthalpy.-function 'Shomate_H'.

The enthalpy is returned in [kJ/mol]

'''    
def get_T_corrections(T,T_arr,d_points,S_dat,t0,H0):
    
    #initial guess for Shomate_S parameters.
    initparam = (1,1,1,1,1,1)
    
    #output from non-linear scipy least squares optimization.
    optimize_output = optimize.least_squares(Shomate_S,initparam,args=(T_arr,S_dat))
    
    #Solution to scipy fit for S(T) vs. T.
    param = optimize_output.x
    RMSE = np.std(optimize_output.fun)
        
    #append (U0+ZPE) to Shomate parameters.
    ShomatePar = [param[0],param[1],param[2],param[3],param[4],0,param[5],H0]
    
    #Solve for Shomate parameter[5]
    ShomatePar[5] = H0-param[0]*t0-param[1]*t0**2/2-param[2]*t0**3/3-param[3]*t0**4/4+param[4]/t0
    
    #Shomate: A,B,C,D,E,F,G
    param_ordered = [param[0],param[1],param[2],param[3],param[4],H0-ShomatePar[5],param[5]]
        
    #Calculate the enthalpy.
    H = Shomate_H(ShomatePar,T)
    
    return param_ordered, RMSE, H


# =============================================================================
# Program Begins Here
# =============================================================================

action = {'gas'}
more_info = 1
sno = 4 #Adsorbate Symmetry number.
Temperature = 298.15 #Temperature [K]

if 'ads' in action:
    cutoff = 100#Frequency cutoff [cm^-1]
       
    #Construct ASE molecule object to pull freqs.
    mol = io.vasp.load_molecule_vasp('CONTCAR','OUTCAR')
    
    #ASE get constrained atoms and masses.
    #Construct ASE Atom object.
    atoms = vasp.read_vasp('CONTCAR')
    constrained_atom_list ,unconstrained_atom_list, unconst_mass_sum, unconst_inertia_tensor  = sort_atoms(atoms)
    
    #Construct TAMkin nma object
    nma = NMA(mol,PHVA(constrained_atom_list))
    nma_HO_cutoff = replace_freq_cutoff(nma,cutoff)#Replace freq w/ cutoff.
    
    #Collect Ground State DFT energy & ZPE energy.
    U0, ZPE, U0_ZPE = get_U0(nma_HO_cutoff.freqs) 
    
    #Organize mode types. Ensure you have visualized modes first!
    Modes_list = len(unconstrained_atom_list)*['vib']
    
    if 'TST' in action:
        Modes_list[0] = 'TST'
        Modes_list[1] = 'trans'
        Modes_list[2] = 'trans'
    else:
        Modes_list[0] = 'trans'
        Modes_list[1] = 'trans'
    
    S = S_2D_ZSA(Modes_list,nma_HO_cutoff,Temperature,(unconst_mass_sum/1000/Na[0]),cutoff,sno,unconst_inertia_tensor)#Obtain entropy.
    
    '''
    Fitting Shomate parameters to calculate enthalpy.
    '''
    T_range = [10.0,350.00]#Peng's code used this range.
    d_points = 100#also used in Peng's code. 100 data points is sufficient for adequate enthalpy estimation.
    T_arr = np.linspace(T_range[0],T_range[-1],num=d_points)#Create Temperature range.

    
    #Calculate entropy across first temperature range.
    S_dat = []
    for i,T in enumerate(T_arr):
        S_dat.append(S_2D_ZSA(Modes_list,nma_HO_cutoff,T,(unconst_mass_sum/1000/Na[0]),cutoff,sno,unconst_inertia_tensor)) #[J/mol/K]    
    
    H0 = U0_ZPE
    param_ordered_298, RMSE_298, H_298 = get_T_corrections(298.15,T_arr,d_points,S_dat,10.0/1000.0,H0)
    
    #Now we solve for NIST's standard enthalpy of formation (from reference state of 298.15 K) to T.
    #We now expand our temperature range to [300,1200] and use T = 298.156 [K] as our reference state.
    Temperature_range = [300.0,1200.0]
    
    #Calculate entropy across second temperature range.
    S_dat = []
    for i,T in enumerate(T_arr):
        S_dat.append(S_2D_ZSA(Modes_list,nma_HO_cutoff,T,(unconst_mass_sum/1000/Na[0]),cutoff,sno,unconst_inertia_tensor)) #[J/mol/K]
    
    param_ordered, RMSE, H = get_T_corrections(Temperature,T_arr,d_points,S_dat,298.15/1000.0,H_298)    


if 'gas' in action:
    
    #Construst ASE Atom object. For periodic systems (i.e VASP), must manually disable periodicity.
    mol_gas = io.vasp.load_molecule_vasp("CONTCAR", "OUTCAR").copy_with(periodic=False)
    
    #Construct TAMkin NMA object. "im_threshold" determines if the molecule is linear or not. If one of the moments of inertia drops below this number, the molecule is considered to be linear.
    nma_gas = NMA(mol_gas, Full(im_threshold=10.0))
    
    #Remove freq corresponding to bond formation. (We already account for this in TST frequency factor "kBT/h")
    if 'TST' in action:
        nma_gas.freqs[0] = 0 #Visually inspect that this mode corresponds to the TST. i.e bond formation/breakage!!!

    #Collect Ground State DFT energy & ZPE energy.
    U0, ZPE, U0_ZPE = get_U0(nma_gas.freqs) 
    
    #Construct Partition function object. May also manually specify symmetry number. (ie.ExtRot(symmetry_number = 1, im_threshold=1.0))
    pf_gas = PartFun(nma_gas, [ExtTrans(cp=True), ExtRot(symmetry_number = sno ,im_threshold=1.0)])
    S  = pf_gas.entropy(Temperature)*Hartree_2_Joule[0]*Na[0] #[J/mol/K]
      	 
      
    '''
    Fitting Shomate parameters to calculate enthalpy.
    '''
    #Here, we solve for the enthalpy of formation from zero [K] to the NIST standard state of 298.15 [K]
    #We approximate the 0 [K] enthalpy to be at 10 [K]; at this temperature H(10 K) ~ (U0 +ZPE)  
    T_range = [10.0,350.00]#Peng's code used this range.
    d_points = 100#also used in Peng's code. 100 data points is sufficient for adequate enthalpy estimation.
    T_arr = np.linspace(T_range[0],T_range[-1],num=d_points)#Create Temperature range.
    
    #Calculate entropy across first temperature range.
    S_dat = []
    for i,T in enumerate(T_arr):
        S_dat.append(pf_gas.entropy(T)*Hartree_2_Joule[0]*Na[0]) #[J/mol/K]
    
    H0 = U0_ZPE
    param_ordered_298, RMSE_298, H_298 = get_T_corrections(298.15,T_arr,d_points,S_dat,10.0/1000.0,H0)

    #Now we solve for NIST's standard enthalpy of formation (from reference state of 298.15 K) to T.
    #We now expand our temperature range to [300,1200] and use T = 298.156 [K] as our reference state.
    T_range = [300.0,1200.0]
    T_arr = np.linspace(T_range[0],T_range[-1],num=d_points)#Create Temperature range.
    
    #Calculate entropy across second temperature range.
    S_dat = []
    for i,T in enumerate(T_arr):
        S_dat.append(pf_gas.entropy(T)*Hartree_2_Joule[0]*Na[0]) #[J/mol/K]

    param_ordered, RMSE, H = get_T_corrections(Temperature,T_arr,d_points,S_dat,298.15/1000.0,H_298)
    
    #for x in range(300,1050,50):    
        #param_ordered, RMSE, H = get_T_corrections(x,T_arr,d_points,S_dat,298.15/1000.0,H_298) 
        #S = pf_gas.entropy(x)*Hartree_2_Joule[0]*Na[0]
        #print(x,U0,ZPE,H,S,H-Temperature*(S/1000.0))

#Print Thermo Values
print('E [kJ/mol], ZPE [kJ/mol], H@%s [kJ/mol], S@%s [J/mol/K], G@%s [kJ/mol]' % (Temperature,Temperature,Temperature))
print('%s %s %s %s %s\n' % (U0,ZPE,H,S,H-Temperature*(S/1000.0)))    
    
    
# =============================================================================
# More Info for Debugging.
# ============================================================================= 

if more_info:
    
    #Print Vibrational Frequencies
    print('Vibrational Frequecnies [cm^-1]\n')
    for i,freq in enumerate(nma_gas.freqs):
        print(i,freq/invcm)
    print('\n')
    
    #Print Shomate Parameters
    param_letter = ['A','B','C','D','E','H-F','G','H']
    print('Shomate Parameters [%s-%s]:\n' % (T_arr[0],T_arr[-1]))
    for i,value in enumerate(param_ordered):
        print(param_letter[i],value)
    print('\n')



