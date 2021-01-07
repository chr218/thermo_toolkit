# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:12:47 2020

This program calculates the entropy, enthalpy, and Gibb's free energy of 
adsorbates, and transition states within zeolites. It is also capable
of calculating the ideal gas-phase entropy, enthalpy, and Gibb's free energy
of molecules and transition states. The necessary input files are VASP
'OUTCAR' and 'CONTCAR'.

The user is given the choice of two approximations for calculating the entropy
of adsorbed molecules/transition states. 

1) The harmonic oscillator-[https://aiche.onlinelibrary.wiley.com/doi/10.1002/aic.16838]
2) 2D free translator-[https://doi.org/10.1021/jp106536m, https://doi.org/10.1016/j.micromeso.2019.04.058],

@author: cvr5246@gmail.com
"""

import sys
from tamkin import *
from molmod import centimeter, lightspeed
from ase.io  import vasp
from scipy import constants,optimize
import numpy as np


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


'''
Subroutines and functions
'''

'''
'get_U0' pulls the ground state energy (ZPE corrected and non ZPEcorrected) 
from the VASP relaxation and returns the value in kJ/mol.

CAUTION!
There are subtle differences in the energy from the freq calc (IBRION= 5) OUTCAR
and a standard relaxation (IBRION = 2) OUTCAR.

Ensure that you are parsing the correct OUTCAR!
'''
def get_U0(freqs):
    atoms_U0 = vasp.read_vasp_out('OUTCAR_relax')
    U0 = atoms_U0.get_potential_energy()*ev[0]*Na[0]/1000
    
    ZPE = 0
    for freq in freqs:
        ZPE += 0.5*h[0]*c[0]*100*freq/invcm
        
    ZPE = ZPE/1.602E-19*ev[0]*Na[0]/1000    

    return U0, (U0+ZPE), ZPE


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
"replace_freq_cutoff" replaces the frequencies (obtain by the nma object) 
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

def S_HO(Modes_list,nma_obj,T,cutoff):
    

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
    St = Svib#Total entropy
    
    '''
    Testing and Debug
    '''
    #H = pf_vib.internal_heat(T)*Hartree_2_Joule[0]*Na[0]#Enthalpy [J/mol]
    
    #Cp = pf_vib.heat_capacity(T)*Hartree_2_Joule[0]*Na[0]#Constant Pressure Heat Capacity [J/mol/K]
    
    #G = pf_vib.free_energy(T)*Hartree_2_Joule[0]*Na[0]#Gibb's Free Energy [J/mol]
    
    #mu = pf_vib.chemical_potential(T)*Hartree_2_Joule[0]*Na[0]#Chemical Potential [J/mol]
    
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
def get_T_corrections(T_range, freqs,Modes_list,nma_obj,T,cutoff,d_points,t0,H0):
    T_arr = np.linspace(T_range[0],T_range[-1],num=d_points)#Create Temperature range.
    
    S_dat = []
    #Calculate entropy at each temperature.
    for i,T in enumerate(T_arr):
        S_dat.append(S_HO(Modes_list,nma_obj,T,cutoff))
    
    #initial guess for Shomate_S parameters.
    initparam = (1,1,1,1,1,1)
    
    #output from non-linear scipy least squares optimization.
    optimize_output = optimize.least_squares(Shomate_S,initparam,args=(T_arr,S_dat))
    
    #solution to scipy optimization.
    param = optimize_output.x
    RMSE = np.std(optimize_output.fun)
    
    #'H0' is (U0+ZPE)
    #H0 = get_U0(freqs)[1]
    
    #append (U0+ZPE) to Shomate parameters.
    ShomatePar = [param[0],param[1],param[2],param[3],param[4],0,param[5],H0]

    #if True:#Print Shomate params for comparison
        #print(param)
    
    #Adjust Shomate parameter[5]
    #t0 = 298.15/1000.0#Shomate reuqires this.
    ShomatePar[5] = H0-param[0]*t0-param[1]*t0**2/2-param[2]*t0**3/3-param[3]*t0**4/4+param[4]/t0
    
    #Calculate the enthalpy.
    H = Shomate_H(ShomatePar,T)
    
    return RMSE, H
   

'''
main program begins here
'''

action = {'ads'}

if 'ads' in action:
    cutoff = 100
    T = 300.00 #Temperature
    
    #Construct molecule object TAMkin
    mol = io.vasp.load_molecule_vasp('CONTCAR','OUTCAR')   
    
    #ASE get constrained atoms and masses.
    atoms = vasp.read_vasp('CONTCAR')
    constrained_atom_list ,unconstrained_atom_list, unconst_mass_sum, unconst_inertia_tensor  = sort_atoms(atoms)
    
    #Construct nma object TAMkin
    nma = NMA(mol,PHVA(constrained_atom_list))

    U0, U0_ZPE, ZPE = get_U0(nma.freqs)#Obtain ground state energy and ZPE corrected energy.
 
    nma_HO_cutoff = replace_freq_cutoff(nma,cutoff)#Apply cutoff. 'trans' freqs are skipped.

        
    #organize mode types based on mode visualization.
    Modes = len(unconstrained_atom_list)*['vib']
    
    if 'TST' in action:
        Modes[0] = 'TST'
    
    S = S_HO(Modes,nma_HO_cutoff,T,cutoff)#Obtain entropy.
    
   
    
    '''
    Fitting Shomate parameters to calculate enthalpy.
    '''
    #Here, we solve for the enthalpy of formation from zero [K] to the NIST standard state of 298.15 [K]
    #We approximate the 0 [K] enthalpy to be at 10 [K]; at this temperature H(10 K) ~ (U0 +ZPE)  
    Temperature_range = [10.0,300.0]#Peng's code used this range.
    data_points = 100#also used in Peng's code. 100 data points is sufficient for adequate enthalpy estimation.
    
    H0 = U0_ZPE
    RMSE_298, H_298 = get_T_corrections(Temperature_range, nma_HO_cutoff.freqs,Modes,nma_HO_cutoff,298.15,cutoff,data_points,10.0/1000.0,U0_ZPE)

    #Now we solve for NIST's standard enthalpy of formation (from reference state of 298.15 K) to T.
    Temperature_range = [300.0,1200.0]
    
    #We now expand our temperature range to [300,1200] and use T = 298.156 [K] as our reference state.
    RMSE, H = get_T_corrections(Temperature_range, nma_HO_cutoff.freqs,Modes,nma_HO_cutoff,T,cutoff,data_points,298.15/1000.0,H_298)    

    print('E [kJ/mol], ZPE [kJ/mol], H@%s [kJ/mol], S@%s [J/mol/K], G@%s [kJ/mol]' % (T,T,T))
    print('%s %s %s %s %s' % (U0,ZPE,H,S,H-T*(S/1000.00)))
    
    
    '''
    This block of commented code was used to optimize number of data points by inspecting RMSE.
    Consider 'Shomate_Sensitivty tab within Excel workbook'
    '''
    #print('data-points upper-temperature H RMSE')
    #for data_points in range(10,1000,10):
    #    Temperature_range = [10.0,350.00]#Peng's code used this range.
        #Obtained enthalpy in [kJ/mol]
    #    RMSE, H = get_T_corrections(Temperature_range, nma_HO_cutoff.freqs,Modes,nma_HO_cutoff,T,unconst_mass_sum/1000/Na[0],cutoff,data_points)
    #    print('%s, %s, %s %s' % (data_points, Temperature_range[-1],H,RMSE))
        

