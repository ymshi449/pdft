import psi4
import pdft
import matplotlib.pyplot as plt
psi4.set_output_file("H2")
bondlength = 1.446

Full_Molec =  psi4.geometry( """
nocom
noreorient
H %f 0.0 0.00
@H 0.0 0.4 0.0
@H 0.0 -0.4 0.0
H -%f 0.0 0.00
units bohr
symmetry c1
""" % (bondlength / 2, bondlength / 2))

Monomer_1 =  psi4.geometry("""
nocom
noreorient
H %f 0.0 0.00
@H 0.0 0.4 0.0
@H 0.0 -0.4 0.0
@H -%f 0.0 0.00
units bohr
symmetry c1
""" % (bondlength / 2, bondlength / 2))

Monomer_2 =  psi4.geometry("""
nocom
noreorient
@H %f 0.0 0.00
@H 0.0 0.4 0.0
@H 0.0 -0.4 0.0
H -%f 0.0 0.00
units bohr
symmetry c1
""" % (bondlength / 2, bondlength / 2))

#Psi4 Options:
psi4.set_options({
    # 'DFT_SPHERICAL_POINTS': 110,
    # 'DFT_RADIAL_POINTS':    5,
    'REFERENCE' : 'UKS'
})

#Make fragment calculations:
f1  = pdft.U_Molecule(Monomer_2,  "cc-pvdz", "SVWN")
f2  = pdft.U_Molecule(Monomer_1,  "cc-pvdz", "SVWN")
mol = pdft.U_Molecule(Full_Molec, "CC-PVDZ", "SVWN")

#Start a pdft systemm, and perform calculation to find vp
pdfter = pdft.U_Embedding([f1, f2], mol)
rho_conv, ep_conv = pdfter.find_vp(maxiter=100, beta=1, atol=1e-5)
#%%
# pdfter.get_energies()
#%%
# vp_plot = Cube(mol.wfn)
#%%
# vp_plot.plot_matrix(vp,2,60)
vp_grid = mol.to_grid(pdfter.vp[0])
fig3 = plt.figure(num=3, figsize=(16, 12), dpi=160)
pdft.plot1d_x(vp_grid, mol.Vpot, title="vp", figure=fig3)

fig1 = plt.figure(num=1, figsize=(16, 12), dpi=160)
plt.plot(rho_conv, figure=fig1)
plt.xlabel(r"iteration")
plt.ylabel(r"$\int |\rho_{whole} - \sum_{fragment} \rho|$")
plt.title(r"$H2$ w/ response method ")
fig1.savefig("rho")
plt.show()
fig2 = plt.figure(num=2, figsize=(16, 12), dpi=160)
plt.plot(ep_conv, figure=fig2)
plt.xlabel(r"iteration")
plt.ylabel(r"Ep")
plt.title(r"$H2$ w/ response method ")
fig2.savefig("Ep")
plt.show()