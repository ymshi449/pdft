import psi4
import pdft
import matplotlib.pyplot as plt
import libcubeprop
import numpy as np

bondlength = 30.0

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
pdft.plot1d_x(n_pbe, mol.Vpot, title="density with vp density_difference" + str(bondlength), fignum=0, dimmer_length=bondlength)
vp_pbe = mol.to_grid((pdfter.vp[0] + pdfter.vp[1])*0.5)
pdft.plot1d_x(vp_pbe, mol.Vpot, title="vp density_difference" + str(bondlength), fignum=1, dimmer_length=bondlength)
#%%
vp_all96, vp_fock_all96 = pdfter.vp_all96()
print("vp all96 norm", np.linalg.norm(vp_fock_all96))
pdft.plot1d_x(vp_all96, mol.Vpot, title="vp all96:" + str(bondlength), fignum=2, dimmer_length=bondlength)
#%% Measure how much does vp_all96 change the density compared with vp_pbe
#%%
vp_fock_psi4 = psi4.core.Matrix.from_array(vp_fock_all96)
pdfter.fragments_scf(max_iter=1000, vp=True, vp_fock=[vp_fock_psi4, vp_fock_psi4])
n_all96 = mol.to_grid(pdfter.fragments_Da + pdfter.fragments_Db)
pdft.plot1d_x(n_all96, mol.Vpot, title="density all96" + str(bondlength), fignum=3, dimmer_length=bondlength)

