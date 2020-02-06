import psi4
import pdft
import matplotlib.pyplot as plt
import libcubeprop
import numpy as np

bondlength = 4.522

psi4.set_output_file("Be2")

Full_Molec =  psi4.geometry("""
nocom
noreorient
Be %f 0.0 0.00
@Be  0 1 0
@Be  0 -1 0
Be -%f 0.0 0.00
units bohr
symmetry c1
""" % (bondlength / 2, bondlength / 2))

Monomer_1 =  psi4.geometry("""
nocom
noreorient
Be %f 0.0 0.00
@Be  0 1 0
@Be  0 -1 0
@Be -%f 0.0 0.00
units bohr
symmetry c1
""" % (bondlength / 2, bondlength / 2))

Monomer_2 =  psi4.geometry("""
nocom
noreorient
@Be %f 0.0 0.00
@Be  0 1 0
@Be  0 -1 0
Be -%f 0.0 0.00
units bohr
symmetry c1
""" % (bondlength / 2, bondlength / 2))

Full_Molec.set_name("Be2")

#Psi4 Options:
psi4.set_options({
    # 'DFT_SPHERICAL_POINTS': 110,
    # 'DFT_RADIAL_POINTS':    5,
    'REFERENCE' : 'UKS'
})

#Make fragment calculations:
f1  = pdft.U_Molecule(Monomer_2,  "cc-pvdz", "SVWN")
f2  = pdft.U_Molecule(Monomer_1,  "cc-pvdz", "SVWN")
mol = pdft.U_Molecule(Full_Molec, "cc-pvdz", "SVWN")

#Start a pdft systemm, and perform calculation to find vp
pdfter = pdft.U_Embedding([f1, f2], mol)
# vp_solution = pdfter.find_vp_optimizing(maxiter=29)

pdfter.find_vp_response2(21, regul_const=1e-5, beta=0.1)
