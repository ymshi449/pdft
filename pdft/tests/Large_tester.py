import psi4
import pdft
from cubeprop import Cube
import matplotlib.pyplot as plt

Full_Molec = psi4.geometry("""
C    0.0000000    0.1929272   -1.9035340
O    0.0000000    1.1595219   -1.1616236
O    0.0000000   -1.0680669   -1.5349870
H    0.0000000    0.2949802   -2.9949776
H    0.0000000   -1.1409414   -0.5399614
C    0.0000000   -0.1929272    1.9035340
O    0.0000000   -1.1595219    1.1616236
O    0.0000000    1.0680669    1.5349870
H    0.0000000   -0.2949802    2.9949776
H    0.0000000    1.1409414    0.5399614
units bohr
symmetry c1
""")

Monomer_1 = psi4.geometry("""
@C    0.0000000    0.1929272   -1.9035340
@O    0.0000000    1.1595219   -1.1616236
@O    0.0000000   -1.0680669   -1.5349870
@H    0.0000000    0.2949802   -2.9949776
@H    0.0000000   -1.1409414   -0.5399614
C    0.0000000   -0.1929272    1.9035340
O    0.0000000   -1.1595219    1.1616236
O    0.0000000    1.0680669    1.5349870
H    0.0000000   -0.2949802    2.9949776
H    0.0000000    1.1409414    0.5399614
units bohr
symmetry c1
""")

Monomer_2 = psi4.geometry("""
C    0.0000000    0.1929272   -1.9035340
O    0.0000000    1.1595219   -1.1616236
O    0.0000000   -1.0680669   -1.5349870
H    0.0000000    0.2949802   -2.9949776
H    0.0000000   -1.1409414   -0.5399614
@C    0.0000000   -0.1929272    1.9035340
@O    0.0000000   -1.1595219    1.1616236
@O    0.0000000    1.0680669    1.5349870
@H    0.0000000   -0.2949802    2.9949776
@H    0.0000000    1.1409414    0.5399614
units bohr
symmetry c1
""")

Full_Molec.set_name("Large")

#Psi4 Options:
psi4.set_options({'DFT_SPHERICAL_POINTS': 434,
                  'DFT_RADIAL_POINTS': 99,
                  'REFERENCE' : 'RKS'})

psi4.set_options({'cubeprop_tasks' : ['density'],
                 'cubic_grid_spacing': [0.1, 0.1, 0.1]})

# energy_3, wfn_3 = psi4.energy("SVWN/cc-pVDZ", molecule=mol_geometry, return_wfn=True)

#Make fragment calculations:
f1  = pdft.U_Molecule(Monomer_2,  "6-311G", "SVWN")
f2  = pdft.U_Molecule(Monomer_1,  "6-311G", "SVWN")
mol = pdft.U_Molecule(Full_Molec, "6-311G", "SVWN")


#Start a pdft systemm, and perform calculation to find vp
pdfter = pdft.U_Embedding([f1, f2], mol)
vp,vpa,vpb,rho_conv,ep_conv = pdfter.find_vp(maxiter=140, beta=1, atol=1e-5)
#%%
# pdfter.get_energies()
#%%
# vp_plot = Cube(mol.wfn)
#%%
# vp_plot.plot_matrix(vp, 2,60)
plt.plot(rho_conv)
plt.xlabel(r"iteration")
plt.ylabel(r"$\int |\rho_{whole} - \sum_{fragment} \rho|$")
plt.title(r"Large Molecule (48 electrons) w/ density difference method ")
plt.show()
plt.plot(ep_conv)
plt.xlabel(r"iteration")
plt.ylabel(r"Ep")
plt.title(r"Large w/ density difference method ")
plt.show()