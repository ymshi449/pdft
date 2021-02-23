import psi4
import XC_Inversion
import matplotlib.pyplot as plt
import libcubeprop
import numpy as np
import pickle


if __name__ == "__main__":
    psi4.set_num_threads(2)
    psi4.set_memory('5 GB')
spherical_points = 230
radial_points = 35

input_density_wfn_method = "CCSD"
reference = "RHF"

functional = 'svwn'
basis = "cc-pcvdz"
vxc_basis = "cc-pcvqz"

ortho_basis = False
svd = "input_once"
opt_method="trust-krylov"
method = "WuYangScipy"
v0 = "FermiAmaldi"

title = method +"\n"+ \
        basis + "/" + str(vxc_basis) + str(ortho_basis) + "\n" + \
        input_density_wfn_method + "\n" +\
        reference + "\n" + \
        "grid"+str(radial_points)+"/"+str(spherical_points)+"\n"+\
        v0 + "\n"\
        + opt_method + "_" + str(svd)
print(title)

psi4.set_output_file("Ethanol.psi4")

Full_Molec = psi4.geometry("""
nocom
noreorient
units angstrom
C 1.1879  -0.3829  0.0000
C 0.0000  0.5526 0.0000
O -1.1867 -0.2472 0.0000
H -1.9237 0.3850  0.0000
H 2.0985 0.2306 0.0000
H 1.1184 -1.0093 0.8869
H 1.1184 -1.0093 -0.8869
H -0.0227 1.1812 0.8852
H -0.0227 1.1812 -0.8852
symmetry c1
""")


Full_Molec.set_name("Ethanol")

#Psi4 Options:
psi4.set_options({
    'DFT_SPHERICAL_POINTS': spherical_points,
    'DFT_RADIAL_POINTS': radial_points,
    "opdm": True,
    "tpdm": True,
    "maxiter": 1000,
    "D_CONVERGENCE": 1e-3,
    'REFERENCE': reference,
    "CUBIC_BlOCK_MAX_POINTS": 10000,
    "DFT_BlOCK_MAX_POINTS": 300,
    "SAVE_JK": True
})

print("Target Density Calculation Started.")
#  Get wfn for target density
# E_input, input_density_wfn = psi4.energy("CCSD"+"/"+basis, molecule=Full_Molec, return_wfn=True)
# _, input_density_wfn = psi4.gradient("CCSD"+"/"+basis, molecule=Full_Molec, return_wfn=True)
# _,input_density_wfn = psi4.properties("CISD/"+basis, molecule=Full_Molec,
#                                             return_wfn=True, properties=['DIPOLE'])
if input_density_wfn_method.upper() == "DETCI":
    E_input,input_density_wfn = psi4.energy("DETCI/"+basis, molecule=Full_Molec,
                                            return_wfn=True)
elif input_density_wfn_method.upper() == "SVWN":
    E_input,input_density_wfn = psi4.energy("SVWN/"+basis, molecule=Full_Molec,
                                            return_wfn=True)
elif input_density_wfn_method.upper() == "SCF":
    E_HF, input_density_wfn = psi4.energy("SCF"+"/"+basis, molecule=Full_Molec, return_wfn=True)
elif input_density_wfn_method.upper() == "CCSD":
    _,input_density_wfn = psi4.properties("CCSD/"+basis, molecule=Full_Molec, properties=['dipole'], return_wfn=True)


print("Target Density Calculation Finished. Size the of basis set: %i" % input_density_wfn.nso())

mol = XC_Inversion.Molecule(Full_Molec, basis, functional)
mol.scf(100, print_energies=True)
print("The inverter molecule is constructed.")
if vxc_basis is not None:
    vxc_basis = XC_Inversion.Molecule(Full_Molec, vxc_basis, functional, jk="JK not needed")
    # vxc_basis.scf(100, print_energies=True)
else:
    vxc_basis = mol
print("The potential basis set constructed.")
print("Number of Basis: ", mol.nbf, vxc_basis.nbf)

inverser = XC_Inversion.Inverser(mol, input_density_wfn,
                                 ortho_basis=ortho_basis,
                                 vxc_basis=vxc_basis,
                                 v0=v0,
                                 # eHOMO=-0.5792,
                                 # v0_wfn=v0_wfn
                                 )

# v = inverser.mRKS(init="LDA", maxiter=25, frac_old=0.3)

# inverser.find_vxc_scipy_WuYang(opt_method="trust-krylov")
# inverser.find_vxc_scipy_constrainedoptimization(opt_method="L-BFGS-B");

# # %% Plotter
# %% Plotter
# L = [3, 0, 0]
# D = [0.02, 0.02, 0.02]
# O = [-4, -4, 0]
# N = [401, 401, 1]
# inverser.v_output_a = inverser.v_output[:vxc_basis.nbf]
# vout_cube_a, xyzw = libcubeprop.basis_to_cubic_grid(inverser.v_output_a,
#                                                     inverser.vp_basis.wfn, L, D, O, N)
# vout_cube_a = np.squeeze(vout_cube_a)
# xyzw[0].shape = xyzw[0].shape[0] * xyzw[0].shape[1] * xyzw[0].shape[2]
# xyzw[1].shape = xyzw[0].shape
# xyzw[2].shape = xyzw[0].shape
# xyzw[3].shape = xyzw[0].shape
# grid = np.array([xyzw[0][:], xyzw[1][:], xyzw[2][:]])
# if v0 == "FermiAmaldi":
#     if (inverser.vH4v0 is None) or  (inverser.vH4v0.shape != vout_cube_a.shape):
#         grid = grid.T
#         inverser.get_esp4v0(grid=grid)
#         inverser.get_vH_vext(grid)
#         inverser.vH4v0.shape = vout_cube_a.shape
#         grid = grid.T
#     nocc = mol.ndocc
#     inverser.vxc_a_grid = vout_cube_a - 1 / nocc * inverser.vH4v0
# elif v0 == "Hartree":
#     inverser.vxc_a_grid = vout_cube_a
# fig, ax = plt.subplots(dpi=200)
# im = ax.imshow(inverser.vxc_a_grid, interpolation="bicubic", vmin=-4, vmax=0)
# fig.colorbar(im)
# fig.show()
#
# vxc_temp = np.copy(inverser.vxc_a_grid)
# vout_temp = np.copy(inverser.v_output_a)