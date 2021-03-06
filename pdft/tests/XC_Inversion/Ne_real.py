import psi4
import XC_Inversion
import matplotlib.pyplot as plt
import libcubeprop
import numpy as np
import pickle


if __name__ == "__main__":
    psi4.set_num_threads(4)
    psi4.set_memory('2 GB')
spherical_points = 110
radial_points = 210

input_density_wfn_method = "DETCI"
reference = "RHF"

functional = 'svwn'
basis = "cc-pcvdz"
vxc_basis = None

ortho_basis = False
svd = "input_once"
opt_method="trust-krylov"
method = "WuYangScipy"
v0 = "FermiAmaldi"

psi4.set_output_file("Ne.psi4")

Full_Molec = psi4.geometry("""
nocom
noreorient
Ne
units bohr
symmetry c1
""")

Full_Molec.set_name("Ne")

title = method +"\n"+ \
        basis + "/" + str(vxc_basis) + str(ortho_basis) + "\n" + \
        input_density_wfn_method + "\n" +\
        reference + "\n" + \
        "grid"+str(radial_points)+"/"+str(spherical_points)+"\n"+\
        v0 + "\n"\
        + opt_method + "_" + str(svd) + '\n'\
        + "Z: " + str(Full_Molec.Z(0))
print(title)

# Exact
Ne = np.genfromtxt('/home/yuming/PDFT/pdft/pdft/data/Atom0/ne.new8/Data')
Ne_xyz = np.concatenate((-np.flip(Ne[:, 1]), Ne[:, 1]))
Ne_vxc = np.concatenate((np.flip(Ne[:, 3]), Ne[:, 3]))
Ne_n = np.concatenate((np.flip(Ne[:, 2]), Ne[:, 2]))
#Psi4 Options:
psi4.set_options({
    'DFT_SPHERICAL_POINTS': spherical_points,
    'DFT_RADIAL_POINTS': radial_points,
    "opdm": True,
    "tpdm": True,
    "maxiter": 3000,
    "D_CONVERGENCE": 1e-3,
    'REFERENCE': reference
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
elif input_density_wfn_method.upper() == "FCI":
    E_input,input_density_wfn = psi4.energy("FCI/"+basis, molecule=Full_Molec,
                                            return_wfn=True)
print("Target Density Calculation Finished.")

mol = XC_Inversion.Molecule(Full_Molec, basis, functional)
mol.scf(100)
if vxc_basis is not None:
    vxc_basis = XC_Inversion.Molecule(Full_Molec, vxc_basis, functional, jk=None)
    vxc_basis.scf(100)
else:
    vxc_basis = mol

print("Number of Basis: ", mol.nbf, vxc_basis.nbf)

inverser = XC_Inversion.Inverser(mol, input_density_wfn,
                                 ortho_basis=ortho_basis,
                                 vxc_basis=vxc_basis,
                                 v0=v0,
                                 # eHOMO=-0.5792,
                                 # v0_wfn=v0_wfn
                                 )

# if method == "WuYangScipy":
#     inverser.find_vxc_scipy_WuYang(opt_method=opt_method)
# elif method == "WuYangMN":
#     hess, jac = inverser.find_vxc_manualNewton(svd_rcond=svd, line_search_method="StrongWolfe")
# elif method == "COScipy":
#     inverser.find_vxc_scipy_constrainedoptimization(opt_method="L-BFGS-B");

# L = None
# D = [0.02, 0.02, 0.02]
# O = [-4, 0, 0]
# N = [801, 1, 1]
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
#
# vxc_temp = np.copy(inverser.vxc_a_grid)
# vout_temp = np.copy(inverser.v_output_a)
#
# fig,ax = plt.subplots(1,1,dpi=200)
# # ax.plot(Ne_xyz, Ne_vxc, label="Exact")
# XC_Inversion.pdft.plot1d_x(inverser.vxc_a_grid, xyz=grid, ax=ax, ls="--")
#
# ax.set_xlim(-2, 8)
# # ax.legend()
# fig.show()

#%%
# rcond = 12  # DD
# rcond = 24  # TT
# GL_rcond = [36, -3, -3] # CDCT
# GL_rcond = [36, -7, -7] # CDCQ
#
# inverser.find_vxc_manualNewton(svd_rcond=GL_rcond[0]*2, line_search_method="StrongWolfe", find_vxc_grid=False)
# L = [3, 0, 0]
# D = [0.01, 0.5, 0.2]
# O = [-3, 0, 0]
# N = [1001, 1, 1]
# inverser.v_output_a = inverser.v_output[:vxc_basis.nbf]
# vout_cube_a, xyzw = libcubeprop.basis_to_cubic_grid(inverser.v_output_a,
#                                                     inverser.vp_basis.wfn, L, D, O, N)
# vout_cube_a.shape = 1001
# xyzw[0].shape = 1001
# xyzw[1].shape = 1001
# xyzw[2].shape = 1001
# xyzw[3].shape = 1001
# mark_y = np.isclose(xyzw[1], 0)
# mark_z = np.isclose(xyzw[2], 0)
# grid = np.array([xyzw[0][mark_y&mark_z], xyzw[1][mark_y&mark_z], xyzw[2][mark_y&mark_z]])
# grid = grid.T
# inverser.get_esp4v0(grid=grid)
# inverser.get_vH_vext(grid)
# nocc = mol.ndocc
# if v0 == "FermiAmaldi":
#     inverser.vxc_a_grid = vout_cube_a[mark_z&mark_y] -1 / nocc * inverser.vH4v0
# elif v0 == "Hartree":
#     inverser.vxc_a_grid = vout_cube_a[mark_z&mark_y]
# grid = grid.T
#
# f,ax = plt.subplots(1,1,dpi=200)
# ax.plot(Ne_xyz, Ne_vxc, label="Exact")
# XC_Inversion.pdft.plot1d_x(inverser.vxc_a_grid, xyz=grid, ax=ax, label="TSVD", ls="--")
# ax.set_xlim(-2.1, 8.1)
#
# vxc_TSVD = np.copy(inverser.vxc_a_grid)
# vout_TSVD = np.copy(inverser.v_output_a)
#
# inverser.find_vxc_manualNewton(svd_rcond="GL_mod", line_search_method="StrongWolfe", find_vxc_grid=False, svd_parameter=GL_rcond)
#
# inverser.v_output_a = inverser.v_output[:vxc_basis.nbf]
# v0_a = inverser.v0_output[:vxc_basis.nbf]
# vbar_a = inverser.vbara_output[:vxc_basis.nbf]
# vbar_b = inverser.vbarb_output[:vxc_basis.nbf]
#
# vout_cube_a, _ = libcubeprop.basis_to_cubic_grid(inverser.v_output_a, inverser.vp_basis.wfn, L, D, O, N)
# v0_cube_a, _ = libcubeprop.basis_to_cubic_grid(v0_a, inverser.vp_basis.wfn, L, D, O, N)
# vbar_cube_a, _ = libcubeprop.basis_to_cubic_grid(vbar_a, inverser.vp_basis.wfn, L, D, O, N)
# vbar_cube_b, _ = libcubeprop.basis_to_cubic_grid(vbar_b, inverser.vp_basis.wfn, L, D, O, N)
# vout_cube_a.shape = 1001
# v0_cube_a.shape = 1001
# vbar_cube_a.shape = 1001
# vbar_cube_b.shape = 1001
#
# nocc = mol.ndocc
# if v0 == "FermiAmaldi":
#     inverser.vxc_a_grid = vout_cube_a[mark_z&mark_y] -1 / nocc * inverser.vH4v0
#     v0_a_grid = v0_cube_a[mark_z&mark_y] -1 / nocc * inverser.vH4v0
#     vbar_a_grid = vbar_cube_a[mark_z&mark_y]
#     vbar_b_grid = vbar_cube_b[mark_z&mark_y]
# elif v0 == "Hartree":
#     inverser.vxc_a_grid = vout_cube_a[mark_z&mark_y]
#     v0_a_grid = v0_cube_a[mark_z&mark_y]
#     vbar_a_grid = vbar_cube_a[mark_z&mark_y]
#     vbar_b_grid = vbar_cube_b[mark_z & mark_y]
# XC_Inversion.pdft.plot1d_x(inverser.vxc_a_grid, xyz=grid, ax=ax, label="TSVD+GL", ls="--")
# XC_Inversion.pdft.plot1d_x(v0_a_grid, xyz=grid, ax=ax, label="v0", ls=":")
# XC_Inversion.pdft.plot1d_x(vbar_a_grid+vbar_b_grid, xyz=grid, ax=ax, label="vbara", ls=":")
# # XC_Inversion.pdft.plot1d_x(vbar_b_grid, xyz=grid, ax=ax, label="vbarb", ls=":")
# ax.set_xlim(-1, 2)
# ax.set_ylim(-10, 4)
# ax.legend()

#%% vxc inverpolation
# Ne_density = np.concatenate((np.flip(Ne[:, 2]), Ne[:, 2]))
#
# from scipy import interpolate
# x,y,z,w = mol.Vpot.get_np_xyzw()
# R = np.sqrt(x**2 + y**2 + z**2)
# fn = interpolate.interp1d(Ne_xyz, Ne_vxc, kind="cubic", bounds_error=False)
# vxc = fn(R)
#
# f,ax = plt.subplots(dpi=200)
# ax.plot(Ne_xyz, Ne_vxc, label="Exact")
# XC_Inversion.pdft.plot1d_x(vxc, Vpot=mol.Vpot, ax=ax, label="Interpolate")
# ax.set_xlim(1e-6,3)
# ax.set_xscale("log")
# ax.legend()
# f.savefig("vxc_interpolation")
#
# vxc_Fock = mol.grid_to_fock(vxc)
# inverser.change_v0("Hartree")
# Vks_a = psi4.core.Matrix.from_array(vxc_Fock + inverser.v0_Fock)
# mol.KS_solver(100, [Vks_a, Vks_a], add_vext=True)
# n_exact = mol.to_grid(mol.Da.np+mol.Db.np)
# n_input = mol.to_grid(input_density_wfn.Da().np+input_density_wfn.Db().np)
# print("exact error", np.sum(np.abs(n_exact-n_input)*w))
#
# f,ax = plt.subplots(dpi=200)
# ax.plot(Ne_xyz, Ne_density, label="Exact")
# XC_Inversion.pdft.plot1d_x(n_exact, Vpot=mol.Vpot, ax=ax, label="n from exact vxc", ls="--")
# XC_Inversion.pdft.plot1d_x(n_input, Vpot=mol.Vpot, ax=ax, label="n input", ls=":")
# ax.set_xlim(1e-7,1)
# ax.set_xscale("log")
# ax.legend()
# f.show()
# f.savefig("compare_density")