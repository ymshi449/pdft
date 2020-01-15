import psi4
import pdft
import matplotlib.pyplot as plt
import libcubeprop
import numpy as np

bondlength = 20.0

Full_Molec =  psi4.geometry( """
nocom
noreorient
He %f 0.0 0.00
@H  0 2.5 0
@H  0 -2.5 0
He -%f 0.0 0.00
units bohr
symmetry c1
""" % (bondlength / 2, bondlength / 2))

Monomer_1 =  psi4.geometry("""
nocom
noreorient
He %f 0.0 0.00
@H  0 2.5 0
@H  0 -2.5 0
@He -%f 0.0 0.00
units bohr
symmetry c1
""" % (bondlength / 2, bondlength / 2))

Monomer_2 =  psi4.geometry("""
nocom
noreorient
@He %f 0.0 0.00
@H  0 2.5 0
@H  0 -2.5 0
He -%f 0.0 0.00
units bohr
symmetry c1
""" % (bondlength / 2, bondlength / 2))

Full_Molec.set_name("He2")

#Psi4 Options:
psi4.set_options({
    # 'DFT_SPHERICAL_POINTS': 110,
    # 'DFT_RADIAL_POINTS':    5,
    'REFERENCE' : 'UKS'
})

# psi4.set_options({'cubeprop_tasks' : ['density'],
#                  'cubic_grid_spacing': [0.1, 0.1, 0.1]})

# energy_3, wfn_3 = psi4.energy("SVWN/cc-pvdz", molecule=mol_geometry, return_wfn=True)

#Make fragment calculations:
mol = pdft.U_Molecule(Full_Molec, "cc-pvdz", "pbe")
f1  = pdft.U_Molecule(Monomer_2,  "cc-pvdz", "pbe", jk=mol.jk)
f2  = pdft.U_Molecule(Monomer_1,  "cc-pvdz", "pbe", jk=mol.jk)

#Start a pdft systemm, and perform calculation to find vp
pdfter = pdft.U_Embedding([f1, f2], mol)
rho_conv, ep_conv = pdfter.find_vp(maxiter=21, beta=3, atol=1e-5)
#%%
vp, vp_fock = pdfter.vp_all96()
#%%
pdft.plot1d_x(np.log(np.abs(vp + 1e-34)), mol.Vpot, title="vp bond:" + str(bondlength), fignum=4, dimmer_length=bondlength)
