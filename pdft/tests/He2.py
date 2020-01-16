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

# energy_3, wfn_3 = psi4.energy("SVWN/sto-3g", molecule=mol_geometry, return_wfn=True)

#Make fragment calculations:
mol = pdft.U_Molecule(Full_Molec, "sto-3g", "pbe")
f1  = pdft.U_Molecule(Monomer_2,  "sto-3g", "pbe", jk=mol.jk)
f2  = pdft.U_Molecule(Monomer_1,  "sto-3g", "pbe", jk=mol.jk)

#Start a pdft systemm, and perform calculation to find vp
pdfter = pdft.U_Embedding([f1, f2], mol)
rho_conv, ep_conv = pdfter.find_vp(maxiter=21, beta=1, atol=1e-5)
pdfter.get_density_sum()
n_pbe = mol.to_grid(pdfter.fragments_Da + pdfter.fragments_Db)
#%%
vp, vp_fock = pdfter.vp_all96()
vp_fock_psi4 = psi4.core.Matrix.from_array(vp_fock)
#%% Measure how much does vp_all96 change the density compared with vp_pbe
pdft.plot1d_x(vp + 1e-34, mol.Vpot, title="vp bond:" + str(bondlength), fignum=4, dimmer_length=bondlength)
