import psi4
import pdft
from cubeprop import Cube
import matplotlib.pyplot as plt

Monomer_1 =  psi4.geometry("""
nocom
noreorient
@H  -1 0 0
@H  0 0 0
H  1 0 0
units bohr
symmetry c1
""")

Monomer_2 =  psi4.geometry("""
nocom
noreorient
H  -1 0 0
@H  0 0 0
@H  1 0 0
units bohr
symmetry c1
""")
Full_Molec =  psi4.geometry("""
nocom
noreorient
1 2
H  -1 0 0
@H  0 0 0
H  1 0 0
units bohr
symmetry c1""")
Full_Molec.set_name("H2P")

#Psi4 Options:
psi4.set_options({'DFT_SPHERICAL_POINTS': 110,
                  'DFT_RADIAL_POINTS':    5,
                  'REFERENCE' : 'RKS'})

psi4.set_options({'cubeprop_tasks' : ['density'],
                 'cubic_grid_spacing': [0.1, 0.1, 0.1]})

# energy_3, wfn_3 = psi4.energy("SVWN/cc-pVDZ", molecule=mol_geometry, return_wfn=True)

#Make fragment calculations:
f1  = pdft.U_Molecule(Monomer_2,  "CC-PVDZ", "SVWN")
f2  = pdft.U_Molecule(Monomer_1,  "CC-PVDZ", "SVWN")
mol = pdft.U_Molecule(Full_Molec, "CC-PVDZ", "SVWN")


#Start a pdft systemm, and perform calculation to find vp
pdfter = pdft.U_Embedding([f1, f2], mol)
vp,vpa,vpb,rho_conv, ep_conv = pdfter.find_vp(maxiter=49, beta=1, atol=1e-5)
#%%
# pdfter.get_energies()
#%%
# vp_plot = Cube(mol.wfn)
#%%
# vp_plot.plot_matrix(vp,2,60)
# plt.plot(rho_conv)
# plt.xlabel(r"iteration")
# plt.ylabel(r"$\int |\rho_{whole} - \sum_{fragment} \rho|$")
# plt.title(r"$H2^+$ w/ response method ")
# plt.show()
# plt.plot(ep_conv)
# plt.xlabel(r"iteration")
# plt.ylabel(r"Ep")
# plt.title(r"$H2^+$ w/ response method ")
# plt.show()