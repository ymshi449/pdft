import psi4
import XC_Inversion
import matplotlib.pyplot as plt
import libcubeprop
import numpy as np

if __name__ == "__main__":
    psi4.set_num_threads(2)

spherical_points = 350
radial_points = 140

input_density_wfn_method = "SCF"
reference = "RHF"

functional = 'svwn'
basis = "cc-pvdz"
vxc_basis = None

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

psi4.set_output_file("formic_lda.psi4")

Full_Molec = psi4.geometry("""
nocom
noreorient
C     1.91331873  0.          0.        
O     1.27264403  1.03649906  0.        
O     1.41949079 -1.21742279  0.        
H     3.00950881  0.          0.        
H     0.42217118 -1.18959318  0.        
C    -1.91331873  0.          0.        
O    -1.27264403 -1.03649906  0.        
O    -1.41949079  1.21742279  0.        
H    -3.00950881  0.          0.        
H    -0.42217118  1.18959318  0.        
units bohr
symmetry c1
""")

Full_Molec.set_name("formic")

#Psi4 Options:
psi4.set_options({
    'DFT_SPHERICAL_POINTS': spherical_points,
    'DFT_RADIAL_POINTS': radial_points,
    "opdm": True,
    "tpdm": True,
    "maxiter": 1000,
    'REFERENCE': reference
})

print("Target Density Calculation Started.")
if input_density_wfn_method.upper() == "DETCI":
    E_input,input_density_wfn = psi4.energy("DETCI/"+basis, molecule=Full_Molec,
                                            return_wfn=True)
elif input_density_wfn_method.upper() == "SVWN":
    E_input,input_density_wfn = psi4.energy("SVWN/"+basis, molecule=Full_Molec,
                                            return_wfn=True)
elif input_density_wfn_method.upper() == "SCF":
    E_HF, input_density_wfn = psi4.energy("SCF"+"/"+basis, molecule=Full_Molec, return_wfn=True)
print("Target Density Calculation Finished.")

mol = XC_Inversion.Molecule(Full_Molec, basis, functional)
mol.scf_inversion(100)
if vxc_basis is not None:
    vxc_basis = XC_Inversion.Molecule(Full_Molec, vxc_basis, functional, jk="No Need for JK")
    print("Number of Basis: ", mol.nbf, vxc_basis.nbf)
    # assert vxc_basis.nbf < 230
    vxc_basis.scf_inversion(10)
else:
    vxc_basis = mol
    print("Number of Basis: ", mol.nbf, vxc_basis.nbf)

inverser = XC_Inversion.Inverser(mol, input_density_wfn,
                                 ortho_basis=ortho_basis,
                                 vxc_basis=vxc_basis,
                                 v0=v0
                                 )
#%%
# grad, grad_app = inverser.check_gradient_constrainedoptimization()
# hess, hess_app = inverser.check_hess_constrainedoptimization()

# if method == "WuYangScipy":
#     inverser.find_vxc_scipy_WuYang(opt_method=opt_method)
# elif method == "WuYangMN":
#     rcondlist, dnlist, Llist = inverser.find_vxc_manualNewton(svd_rcond=svd, line_search_method="LD")
    # inverser.find_vxc_manualNewton(svd_rcond=svd, line_search_method="StrongWolfeD")
# elif method == "COScipy":
#     inverser.find_vxc_scipy_constrainedoptimization(opt_method=opt_method)

# L = [7, 0.2, 0.2]
# D = [0.1, 0.5, 0.2]
# inverser.v_output_a = inverser.v_output[:vxc_basis.nbf]
# vout_cube_a, xyzw = libcubeprop.basis_to_cubic_grid(inverser.v_output_a, inverser.vxc_basis.wfn,L,D)
# vout_cube_a.shape = 201*6*2
# xyzw[0].shape = 201*6*2
# xyzw[1].shape = 201*6*2
# xyzw[2].shape = 201*6*2
# xyzw[3].shape = 201*6*2
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
# x_ax = xyzw[0][mark_y&mark_z]
# y_ax = xyzw[1][mark_y&mark_z]
# z_ax = xyzw[2][mark_y&mark_z]
# w_ax = xyzw[3][mark_y&mark_z]
# npoints = x_ax.shape[0]
# phi = np.empty((npoints, 128))
# for i in range(128):
#     v_temp = np.zeros(128)
#     v_temp[i] = 1.0
#     phi_temp, _ = libcubeprop.basis_to_cubic_grid(v_temp, mol.wfn, L,D)
#     phi_temp.shape = 201 * 6 * 2
#     phi[:,i] = np.copy(phi_temp[mark_y&mark_z])
#
# superfunc = mol.Vpot.functional()
# Da = mol.Da.np
# rho = np.einsum('pm,mn,pn->p', phi, Da, phi, optimize=True)
# inp = {}
# inp["RHO_A"] = psi4.core.Vector.from_array(rho)
# inp["RHO_B"] = psi4.core.Vector.from_array(rho)
# ret = superfunc.compute_functional(inp, -1)
# v_rho_a = np.array(ret["V_RHO_A"])[:201]
# f,ax = plt.subplots(1,1,dpi=200)
# XC_Inversion.pdft.plot1d_x(v_rho_a, xyz=grid, ax=ax,label="LDA")
# XC_Inversion.pdft.plot1d_x(inverser.vxc_a_grid, xyz=grid, ax=ax, label="$\lambda=0$", ls="--")
#
#
# rgl_list, L_list, dT_list, P_list = inverser.my_L_curve_regularization4WuYang();
# # inverser.find_vxc_scipy_WuYang(opt_method=opt_method)
# inverser.find_vxc_manualNewton(svd_rcond=svd, line_search_method="StrongWolfeD")
#
# inverser.v_output_a = inverser.v_output[:vxc_basis.nbf]
# vout_cube_a, _ = libcubeprop.basis_to_cubic_grid(inverser.v_output_a, inverser.vxc_basis.wfn, L,D)
# vout_cube_a.shape = 201*6*2
# nocc = mol.ndocc
# if v0 == "FermiAmaldi":
#     inverser.vxc_a_grid = vout_cube_a[mark_z&mark_y] -1 / nocc * inverser.vH4v0
# elif v0 == "Hartree":
#     inverser.vxc_a_grid = vout_cube_a[mark_z&mark_y]
# XC_Inversion.pdft.plot1d_x(inverser.vxc_a_grid, xyz=grid, ax=ax, label="$\lambda=%.2e$"%inverser.regularization_constant, ls="--")
# ax.legend()
# f.show()