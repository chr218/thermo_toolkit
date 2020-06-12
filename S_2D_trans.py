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

    return U0, (U0+ZPE)


'''
"sort_atoms" sorts the atoms into unconstrained and constrained lists as indices.
This function returns the constrained atom indices, unconstrained atom indices
and the total mass of the unconstrained atoms.
'''
def sort_atoms(atms):
    #Sort atom indices into constrained and unconstrained. 
    if len(atms._get_constraints())>0:
        const = (atms._get_constraints())[0].get_indices()
    else: const = []

    unconst = [int(x) for x in range(len(atms)) if x not in const]

    mass_unconst = 0
    for atom in atms:
        if atom.index in unconst:
            mass_unconst += atom.mass
            
    return const,unconst,mass_unconst
        

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

def S_2D_ZSA(Modes_list,nma_obj,T,mass,cutoff):
    CsaZeo = 1/(2E-10*6E-10)#Surface area of zeolite-De Moor et al. 'Ads of C2-C8 n-alkanes in zeolites'
    S2d = R[0]*((np.log((2*pi*mass*kb[0]*T/(h[0]**2))/CsaZeo))+2) # 2D Translational contribution, mol constrained to SA of zeolite.
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
    St = Svib + S2d#Total entropy
    
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
def get_T_corrections(T_range, freqs,Modes_list,nma_obj,T,mass,cutoff,d_points):
    T_arr = np.linspace(T_range[0],T_range[-1],num=d_points)#Create Temperature range.
    
    S_dat = []
    #Calculate entropy at each temperature.
    for i,T in enumerate(T_arr):
        S_dat.append(S_2D_ZSA(Modes_list,nma_obj,T,mass,cutoff))
    
    #initial guess for Shomate_S parameters.
    initparam = (1,1,1,1,1,1)
    
    #output from non-linear scipy least squares optimization.
    optimize_output = optimize.least_squares(Shomate_S,initparam,args=(T_arr,S_dat))
    
    #solution to scipy optimization.
    param = optimize_output.x
    RMSE = np.std(optimize_output.fun)
    
    #'H0' is (U0+ZPE)
    H0 = get_U0(freqs)[-1]
    
    #append (U0+ZPE) to Shomate parameters.
    ShomatePar = [param[0],param[1],param[2],param[3],param[4],0,param[5],H0]
    
    #Adjust Shomate parameter[5]
    t0 = 298.15/1000.0#Shomate reuqires this.
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
    
    #Construst molecule object
    mol = io.vasp.load_molecule_vasp('CONTCAR','OUTCAR')
    
    #ASE get constrained atoms and masses.
    atoms = vasp.read_vasp('CONTCAR')
    
    constrained_atom_list ,unconstrained_atom_list, unconst_mass_sum = sort_atoms(atoms)
    
    #Construct nma object
    nma = NMA(mol,PHVA(constrained_atom_list))
    nma_HO_cutoff = replace_freq_cutoff(nma,100)#Apply cutoff. 'trans' freqs are skipped.
    
    #organize mode types based on mode visualization.
    Modes = len(unconstrained_atom_list)*['vib']
    
    if 'TST' in action:
        Modes[0] = 'TST'
        Modes[1] = 'trans'
        Modes[2] = 'trans'
    else:
        Modes[0] = 'trans'
        Modes[1] = 'trans'
    
    S = S_2D_ZSA(Modes,nma_HO_cutoff,T,unconst_mass_sum/1000/Na[0],cutoff)#Obtain entropy.
    
    U0, U0_ZPE = get_U0(nma_HO_cutoff.freqs)#Obtain ground state energy and ZPE corrected energy.
    
    
    '''
    Fitting Shomate parameters to calculate enthalpy.
    '''
    Temperature_range = [10.0,350.00]#Peng's code used this range.
    data_points = 100#also used in Peng's code. 100 data points is sufficient for adequate enthalpy estimation.
    
    RMSE, H = get_T_corrections(Temperature_range, nma_HO_cutoff.freqs,Modes,nma_HO_cutoff,T,unconst_mass_sum/1000/Na[0],cutoff,data_points)
    
    print('H@300 [kJ/mol] \t S@300 [J/mol/K] \t G@300 [kJ/mol]')
    print('%s %s %s' % (H,S,H-300.00*(S/1000.00)))
    
    
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
        

if 'gas' in action:
    T = 300.00 #Temperature
    
    #Construst molecule object. For periodic systems (i.e VASP), must manually disable periodicity.
    mol_gas = io.vasp.load_molecule_vasp("CONTCAR", "OUTCAR").copy_with(periodic=False)
    
    #Construct NMA object. "im_threshold" determines if the molecule is linear or not. If one of the moments of inertia drops below this number, the molecule is considered to be linear.
    nma_gas = NMA(mol_gas, Full(im_threshold=10.0))
    
    #Construct Partition function object. May also manually specify symmetry number. (ie.ExtRot(symmetry_number = 1im_threshold=1.0))
    pf_gas = PartFun(nma_gas, [ExtTrans(cp=False), ExtRot(im_threshold=1.0)])
    S_gas  = pf_gas.entropy(T)*Hartree_2_Joule[0]*Na[0] #[J/mol/K]
   	 
    H = pf_gas.internal_heat(T)*Hartree_2_Joule[0]*Na[0]#Enthalpy [J/mol]
    
    #Cp = pf_gas.heat_capacity(T)*Hartree_2_Joule[0]*Na[0]#Constant Pressure Heat Capacity [J/mol/K]
    
    G = pf_gas.free_energy(T)*Hartree_2_Joule[0]*Na[0]#Gibb's Free Energy [J/mol]
    
    #mu = pf_vib.chemical_potential(T)*Hartree_2_Joule[0]*Na[0]#Chemical Potential [J/mol]
   
    print('%s %s %s' % (H/1000,S_gas, G/1000))


'''
testing/debugging
'''
#nma_HO_cutoff = replace_freq_cutoff(nma,100)
#Construct Partiion function object.
#pf = PartFun(nma, [ExtTrans(dim=2, mobile=unconst)]) #2D translator surface
#pf = PartFun(nma_HO_cutoff, []) #Harmonic Oscialltor
#pf = PartFun(nma, []) #Harmonic Oscialltor
#S = pf.entropy(300)*Hartree_2_Joule[0]*Na[0] #[J/mol/K]
#print(S)
