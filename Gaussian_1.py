from ase import Atoms
from ase.io import read, write
from scipy import constants
import re
import numpy as np
#from ase.visualize import view
from ase.constraints import FixAtoms

#Read Hessian from OUTCAR and obtain masses of unconstrained atoms---------------------------------------------------------------

atoms=read('POSCAR') #Give file to read

#Determining unconstrained "free" atom indices.
atoms_constrained = (atoms._get_constraints())[0].get_indices()
atom_all=[]
for atom in atoms:
    atom_all.append(atom.index)
atoms_free = set(atom_all)-set(atoms_constrained)#number of unconstrained atoms

#extract mass weighted Hessian (mass weighted force constants)
Hessian_raw=[]

file_OUTCAR = open("OUTCAR","r")
PATTERN_start = re.compile('SECOND DERIVATIVES')
PATTERN_end = re.compile('Eigenvectors and eigenvalues')
switch = 0
for line in file_OUTCAR:
    if switch:
        Hessian_raw.append(line)     
    if PATTERN_start.search(line):
       switch = 1
    if PATTERN_end.search(line):
        break
	
Hessian_raw.pop(0) #temperorary-removing unecessary strings
column_indices = (Hessian_raw.pop(0)).split() #This list provides the column order 'indices' of atomic masses for weighing the Hessian
Hessian_raw.pop(-1)
Hessian_raw.pop(-1)
Hessian_raw.pop(-1)

row_indices=[]
Hessian = []
for row in range(len(Hessian_raw)):
    row_split = Hessian_raw[row].split()
    row_indices.append(row_split.pop(0)) #This list provides the row order 'indices' of the atomic masses for weighing the Hessian
    for word in row_split:
        Hessian.append(float(word))

#Reshape Hessian (non mass weighted) into 3N by 3N
Hessian_array = np.asarray(Hessian)
Hessian_array = np.reshape(Hessian_array,(int(np.sqrt(len(Hessian_array))),int(np.sqrt(len(Hessian_array)))))
#make copy of Hessian array for mass weighted array.
Hessian_mwc = np.copy(Hessian_array)
#----------------------------------------------------------------------------------------------------------------------------------

#Mass weight the Hessian and diagonalize-------------------------------------------------------------------------------------------

#Mass weight the Hessian
for row in range(len(Hessian_array[:,0])):
    for col in range(len(Hessian_array[0,:])):
        mass_col_index = int(column_indices[col][:-1])-1
        mass_row_index = int(row_indices[row][:-1])-1
        mass_col = [atom.mass for atom in atoms if atom.index == mass_col_index]
        mass_row = [atom.mass for atom in atoms if atom.index == mass_row_index]
        Hessian_mwc[row,col] = Hessian_array[row,col]/(np.sqrt(float(mass_col[0])*float(mass_row[0])))

#Generate Mass list (necessary for generating translating frame vector)
mass_list = []
for col in range(len(Hessian_array[0,:])):
        mass_col_index = int(column_indices[col][:-1])-1
        mass_col = [atom.mass for atom in atoms if atom.index == mass_col_index]
        mass_list.append(float(mass_col[0]))

#Make copy of mass weighted Hessian and take negative. (VASP Hessian is first derivative, hence for V we should multiply by "-1"
Hessian_mwc_cp = -np.copy(Hessian_mwc)


#Diagnolize copy of mass weighted Hessian. 'eigh' assumes symmetric matrix and is faster than 'eig', 'walue' are eigenvalues whos roots are findamental frequencies.
walue,vec = np.linalg.eig(Hessian_mwc_cp)

#Obtain wavenumbers & energies.
Joule_to_eV, unit, uncertainty = constants.physical_constants["joule-electron volt relationship"]
frequencies = []
energies = []
factor = np.sqrt(constants.elementary_charge/(constants.angstrom**2)/constants.m_u)
for eig in walue:
    freq = factor*np.sqrt(eig)/(constants.c*100*constants.pi*2)
    frequencies.append(freq)#in wave-number cm^-1
    energies.append(freq*constants.c*constants.h*100*Joule_to_eV*1000) #in milli electron-volt

print("Frequencies in cm^-1: ",frequencies)
print("Energies in meV: ",energies)

#Determine Principal axes of inertia----------------------------------------------------------------------------------------------

#Remove constrained atoms-necessary for moving coordinate system to center of mass of unconstrained atoms.
#NOTE: The code by Guowen Peng takes the center of mass of the entire structure.
del atoms[atoms_constrained]
#Obtain center of mass
com = atoms.get_center_of_mass()
#Translate molecule's com to origin.
for atom in atoms:
    atom.position = (atom.position-com)

#Obtain the three 'principle moments of interia-computed from the eigenvalues of the symmetric inertial tensor.
I_principal, I_vec = atoms.get_moments_of_inertia(vectors=True)
#---------------------------------------------------------------------------------------------------------------------------------

#Generate coordinates in the rotating and translating frame
#---------------------------------------------------------------------------------------------------------------------------------
#Generating the three vectors 'D1, D2, D2' corresponding to the translating frame.
D = np.zeros((3,len(Hessian_mwc)))
for row in range(3):
    for col in range(len(D[0,:])):
        if not col%(3):
            D[row,col+row] = mass_list[col]

#Generating rotating frame vectors.


#Generating atomic coordinate array (necessary to find P- which is the dot product of the shifted atoms and the eigenvector of the moment of inertia tensor)
coord = np.zeros((len(atoms_free),3))
coord_mass = np.zeros((len(atoms_free),3))
mass_list_cut = []

row = 0
for atom in atoms:
    for col in range(3):
        coord[row,col] = atom.position[col]
        coord_mass[row,col] = atom.position[col]*atom.mass
    mass_list_cut.append(atom.mass)
    row += 1

#Generating 'P', the dot product of 'R_com' (the coordinates of the atoms with respect to the center of mass) and the corresponding row of the matrix used to diagnolize the moment of inertia tensor..


P = np.dot(coord_mass,I_vec)

#Generate "D_4"
D_4 = np.zeros((len(atoms_free),3))

for i in range(len(D_4[:,0])):
    for j in range(len(D_4[0,:])):
        D_4[i,j] = (P[i,1]*I_vec[2,j]-P[i,2]*I_vec[1,j])/np.sqrt(mass_list_cut[i])

#Generate "D_5"
D_5 = np.zeros((len(atoms_free),3))

for i in range(len(D_5[:,0])):
    for j in range(len(D_5[0,:])):
        D_5[i,j] = (P[i,2]*I_vec[0,j]-P[i,0]*I_vec[2,j])/np.sqrt(mass_list_cut[i])


#Generate "D_6"
D_6 = np.zeros((len(atoms_free),3))

for i in range(len(D_6[:,0])):
    for j in range(len(D_6[0,:])):
        D_6[i,j] = (P[i,0]*I_vec[1,j]-P[i,1]*I_vec[0,j])/np.sqrt(mass_list_cut[i])

#Determine the rotational and translational vectors and Normalize each Vector
#Generate main matrix with 'D vectors'
D_4_5_6 = np.vstack((D_4,D_5,D_6))
D_main = np.vstack((np.transpose(D),D_4_5_6))


print(D.shape)
#for row in range(len(D_main[:,0])):
    #print(np.dot(D_main[row,:],np.transpose(D_main[row,:])))

#---------------------------------------------------------------------------------------------------------------------------------
