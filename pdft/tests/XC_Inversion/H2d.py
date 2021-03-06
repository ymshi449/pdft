import psi4
import XC_Inversion
import matplotlib.pyplot as plt
import libcubeprop
import numpy as np

if __name__ == "__main__":
    psi4.set_num_threads(3)

functional = 'svwn'
basis = 'cc-pcvdz'

vp_basis = 'cc-pcvQz'

ortho_basis = False
svd = "segment_cycle_cutoff"
opt_method="trust-krylov"
method = "WuYangScipy"
v0 = "FermiAmaldi"


psi4.set_output_file("H2d.psi4")

Full_Molec = psi4.geometry("""
nocom
noreorient
H -3.7795 0.0 0.0
H  3.7795 0.0 0.0
units bohr
symmetry c1
""")

Full_Molec.set_name("He")

#Psi4 Options:
psi4.set_options({
    'DFT_SPHERICAL_POINTS': 194,
    'DFT_RADIAL_POINTS': 44,
    'REFERENCE' : 'UHF'
})
E, input_density_wfn = psi4.energy("CCSD(T)"+"/"+basis, molecule=Full_Molec, return_wfn=True)
# E, input_density_wfn = psi4.energy(functional+"/"+basis, molecule=Full_Molec, return_wfn=True)
#Psi4 Options:
psi4.set_options({
    'REFERENCE' : 'UHF'
})
mol = XC_Inversion.Molecule(Full_Molec, basis, functional)
mol.scf(100)
if vp_basis is not None:
    vp_basis = XC_Inversion.Molecule(Full_Molec, vp_basis, functional, jk="No Need for JK")
    print("Number of Basis: ", mol.nbf, vp_basis.nbf)
    # assert vp_basis.nbf < 230
    vp_basis.scf(10)
else:
    vp_basis = mol
    print("Number of Basis: ", mol.nbf, vp_basis.nbf)

inverser = XC_Inversion.Inverser(mol, input_density_wfn,
                                 ortho_basis=ortho_basis,
                                 vxc_basis=vp_basis,
                                 v0=v0
                                 )

# grad, grad_app = inverser.check_gradient_constrainedoptimization()
# hess, hess_app = inverser.check_hess_constrainedoptimization()

if method == "WuYangScipy":
    inverser.find_vxc_scipy_WuYang(opt_method=opt_method)
elif method == "WuYangMN":
    # rcondlist, dnlist, Llist = inverser.find_vxc_manualNewton(svd_rcond=svd, line_search_method="LD")
    inverser.find_vxc_manualNewton(svd_rcond=svd, line_search_method="StrongWolfeD")
elif method == "COScipy":
    inverser.find_vxc_scipy_constrainedoptimization(opt_method=opt_method)


npt = 300
L = [3, 0, 0]
D = [0.1, 0.5, 0.2]
O = [-15, 0, 0]
N = [npt, 1, 1]
inverser.v_output_a = inverser.v_output[:vp_basis.nbf]
vout_cube_a, xyzw = libcubeprop.basis_to_cubic_grid(inverser.v_output_a, inverser.vp_basis.wfn, L, D, O, N)
vout_cube_a.shape = npt
xyzw[0].shape = npt
xyzw[1].shape = npt
xyzw[2].shape = npt
xyzw[3].shape = npt
mark_y = np.isclose(xyzw[1], 0)
mark_z = np.isclose(xyzw[2], 0)
grid = np.array([xyzw[0][mark_y&mark_z], xyzw[1][mark_y&mark_z], xyzw[2][mark_y&mark_z]])
grid = grid.T
inverser.get_esp4v0(grid=grid)
inverser.get_vH_vext(grid)
nocc = mol.ndocc
if v0 == "FermiAmaldi":
    inverser.vxc_a_grid = vout_cube_a[mark_z&mark_y] -1 / nocc * inverser.vH4v0
elif v0 == "Hartree":
    inverser.vxc_a_grid = vout_cube_a[mark_z&mark_y]
grid = grid.T

# # LDA
# x_ax = xyzw[0][mark_y&mark_z]
# y_ax = xyzw[1][mark_y&mark_z]
# z_ax = xyzw[2][mark_y&mark_z]
# w_ax = xyzw[3][mark_y&mark_z]
# npoints = x_ax.shape[0]
# phi = np.empty((npoints, mol.nbf))
# for i in range(mol.nbf):
#     v_temp = np.zeros(mol.nbf)
#     v_temp[i] = 1.0
#     phi_temp, _ = libcubeprop.basis_to_cubic_grid(v_temp, mol.wfn, L, D, O, N)
#     phi_temp.shape = 100
#     phi[:,i] = np.copy(phi_temp[mark_y&mark_z])
# superfunc = mol.Vpot.functional()
# Da = mol.Da.np
# rho = np.einsum('pm,mn,pn->p', phi, Da, phi, optimize=True)
# inp = {}
# inp["RHO_A"] = psi4.core.Vector.from_array(rho)
# inp["RHO_B"] = psi4.core.Vector.from_array(rho)
# ret = superfunc.compute_functional(inp, -1)
# v_rho_a = np.array(ret["V_RHO_A"])[:100]
f,ax = plt.subplots(1,1,dpi=200)

# XC_Inversion.pdft.plot1d_x(v_rho_a, xyz=grid, ax=ax,label="LDA")

XC_Inversion.pdft.plot1d_x(inverser.vxc_a_grid, xyz=grid, ax=ax, label="$\lambda=0$", ls="--")

rgl_list, L_list, dT_list, P_list = inverser.my_L_curve_regularization4WuYang();
inverser.find_vxc_scipy_WuYang(opt_method=opt_method)

inverser.v_output_a = inverser.v_output[:vp_basis.nbf]
vout_cube_a, _ = libcubeprop.basis_to_cubic_grid(inverser.v_output_a, inverser.vp_basis.wfn, L, D, O, N)
vout_cube_a.shape = npt
nocc = mol.ndocc
if v0 == "FermiAmaldi":
    inverser.vxc_a_grid = vout_cube_a[mark_z&mark_y] -1 / nocc * inverser.vH4v0
elif v0 == "Hartree":
    inverser.vxc_a_grid = vout_cube_a[mark_z&mark_y]
XC_Inversion.pdft.plot1d_x(inverser.vxc_a_grid, xyz=grid, ax=ax, label="$\lambda=%.2e$"%inverser.regularization_constant, ls="--")
ax.legend()
f.show()
