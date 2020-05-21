import psi4
import XC_Inversion
import matplotlib.pyplot as plt
import libcubeprop
import numpy as np

if __name__ == "__main__":
    psi4.set_num_threads(2)

separation = 5.122
functional = 'svwn'
basis = 'sto-3g'
basis = 'aug-pcsseg-3'
basis = 'aug-cc-pvqz'
basis = 'cc-pvdz'
basis = 'aug-cc-pvtz'

vp_basis = None

ortho_basis = False
svd = "search_segment_cycle"
svd = "input_every"
opt_method="BFGS"
method = "WuYangScipy"
v0 = "FermiAmaldi"
v0 = "Hartree"

title = method +"_"+ opt_method +"_"+v0+ "_" + basis+"_"+ \
        str(vp_basis) + "_"\
        + str(ortho_basis) + "_" + str(svd)
print(title)

psi4.set_output_file("Li2.psi4")

Full_Molec = psi4.geometry("""
nocom
noreorient
Li %f 0.0 0.00
Li -%f 0.0 0.00
units bohr
symmetry c1
""" % (separation / 2, separation / 2))


Full_Molec.set_name("Li2")

#Psi4 Options:
psi4.set_options({
    'DFT_SPHERICAL_POINTS': 302,
    'DFT_RADIAL_POINTS': 77,
    'REFERENCE' : 'UKS'
})
E, input_wfn = psi4.energy(functional+"/"+basis, molecule=Full_Molec, return_wfn=True)

mol = XC_Inversion.Molecule(Full_Molec, basis, functional)
mol.scf_inversion(100)
if vp_basis is not None:
    vp_basis = XC_Inversion.Molecule(Full_Molec, vp_basis, functional)
    vp_basis.scf_inversion(100)
else:
    vp_basis = mol

print("Number of Basis: ", mol.nbf, vp_basis.nbf)

inverser = XC_Inversion.Inverser(mol, input_wfn,
                                 ortho_basis=ortho_basis,
                                 vp_basis=vp_basis,
                                 v0=v0
                                 )

# grad, grad_app = inverser.check_gradient_constrainedoptimization()
# hess, hess_app = inverser.check_hess_constrainedoptimization()

if method == "WuYangScipy":
    inverser.find_vxc_scipy_WuYang(14000, opt_method=opt_method)
elif method == "WuYangMN":
    # rcondlist, dnlist, Llist = inverser.find_vxc_manualNewton(svd_rcond=svd, back_tracking_method="LD")
    inverser.find_vxc_manualNewton(svd_rcond=svd, back_tracking_method="L")
elif method == "COScipy":
    inverser.find_vxc_scipy_constrainedoptimization(opt_method=opt_method)

# dDa = input_wfn.Da().np - mol.Da.np
# dDb = input_wfn.Db().np - mol.Db.np
# dn = mol.to_grid(dDa + dDb)

f,ax = plt.subplots(1,1,dpi=200)
XC_Inversion.pdft.plot1d_x(inverser.input_vxc_a, input_wfn.V_potential(), ax=ax,
                           dimmer_length=separation, label="input_xc_a", title=title)
XC_Inversion.pdft.plot1d_x(inverser.vxc_a_grid, vp_basis.Vpot, ax=ax, label="WuYang_xc_a", ls='--')
# XC_Inversion.pdft.plot1d_x(np.log10(np.abs(dn)), mol.Vpot, ax=ax, label="logdn", ls='dotted')
ax.legend()
ax.set_xlim(-14,14)
ax.set_ylim(-3,0.1)
f.show()
plt.close(f)