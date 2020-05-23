"""
An inverser for vxc. Inherit from PDFT.
"""

import psi4
import numpy as np
import matplotlib.pyplot as plt
import pdft
import scipy.optimize as optimizer


if __name__ == "__main__":
    psi4.set_num_threads(2)


class Molecule(pdft.U_Molecule):
    def __init__(self, geometry, basis, method):
        super().__init__(geometry, basis, method)

    def scf_inversion(self, maxiter, V=None, print_energies=False, vp_matrix=None):
        """
        Performs scf calculation to find energy and density with a given V
        Parameters
        ----------
        V_a: fock matrix of v_output_a + v_FA_a
        V_b: fock matrix of v_output_b + v_FA_b
        Returns
        -------
        """
        assert vp_matrix is None

        if V is None:
            V_a = psi4.core.Matrix.from_array(np.zeros_like(self.T.np))
            V_b = psi4.core.Matrix.from_array(np.zeros_like(self.T.np))
        else:
            V_a = V[0]
            V_b = V[1]

        if self.Da is None:
            C_a, Cocc_a, D_a, eigs_a = pdft.build_orbitals(self.H, self.A, self.nalpha)
            C_b, Cocc_b, D_b, eigs_b = pdft.build_orbitals(self.H, self.A, self.nbeta)
        else: # Use the calculation from last v as initial.
            nbf = self.A.shape[0]
            Cocc_a = psi4.core.Matrix(nbf, self.nalpha)
            Cocc_a.np[:] = self.Ca.np[:, :self.nalpha]
            Cocc_b = psi4.core.Matrix(nbf, self.nbeta)
            Cocc_b.np[:] = self.Cb.np[:, :self.nbeta]
            D_a = self.Da
            D_b = self.Db

        diisa_obj = psi4.p4util.solvers.DIIS(max_vec=3, removal_policy="largest")
        diisb_obj = psi4.p4util.solvers.DIIS(max_vec=3, removal_policy="largest")

        Eold = 0.0
        E = 0.0
        E_conv = psi4.core.get_option("SCF", "E_CONVERGENCE")
        E_conv = 1e-9
        D_conv = psi4.core.get_option("SCF", "D_CONVERGENCE")

        for SCF_ITER in range(maxiter+1):
            #Bring core matrix
            F_a = self.H.clone()
            F_b = self.H.clone()
            F_a.axpy(1.0, V_a)
            F_b.axpy(1.0, V_b)

            #DIIS
            diisa_e = psi4.core.triplet(F_a, D_a, self.S, False, False, False)
            diisa_e.subtract(psi4.core.triplet(self.S, D_a, F_a, False, False, False))
            diisa_e = psi4.core.triplet(self.A, diisa_e, self.A, False, False, False)
            diisa_obj.add(F_a, diisa_e)

            diisb_e = psi4.core.triplet(F_b, D_b, self.S, False, False, False)
            diisb_e.subtract(psi4.core.triplet(self.S, D_b, F_b, False, False, False))
            diisb_e = psi4.core.triplet(self.A, diisb_e, self.A, False, False, False)
            diisb_obj.add(F_b, diisb_e)

            Core = 1.0 * self.H.vector_dot(D_a) + 1.0 * self.H.vector_dot(D_b)
            # This is not the correct Eks but Eks - Eext
            fake_Eks = V_a.vector_dot(D_a) + V_b.vector_dot(D_b)

            SCF_E = Core
            SCF_E += fake_Eks
            SCF_E += self.Enuc

            dRMS = 0.5 * (np.mean(diisa_e.np**2)**0.5 + np.mean(diisb_e.np**2)**0.5)

            if (abs(SCF_E - Eold) < E_conv) and (dRMS < D_conv):
                if print_energies is True:
                    print(F'SCF Convergence: NUM_ITER = {SCF_ITER} dE = {abs(SCF_E - Eold)} dDIIS = {dRMS}')
                break

            Eold = SCF_E

            #DIIS extrapolate
            F_a = diisa_obj.extrapolate()
            F_b = diisb_obj.extrapolate()

            #Diagonalize Fock matrix
            C_a, Cocc_a, D_a, eigs_a = pdft.build_orbitals(F_a, self.A, self.nalpha)
            C_b, Cocc_b, D_b, eigs_b = pdft.build_orbitals(F_b, self.A, self.nbeta)
            #Exchange correlation energy/matrix
            self.Vpot.set_D([D_a,D_b])
            self.Vpot.properties()[0].set_pointers(D_a, D_b)

            if SCF_ITER == maxiter:
                # raise Exception("Maximum number of SCF cycles exceeded.")
                print("Maximum number of SCF cycles exceeded.")
                if print_energies is True:
                    print(F'SCF Convergence: NUM_ITER = {SCF_ITER} dE = {abs(SCF_E - Eold)} dDIIS = {dRMS}')

        # Diagonalize Fock matrix
        C_a, Cocc_a, D_a, eigs_a = pdft.build_orbitals(F_a, self.A, self.nalpha)
        C_b, Cocc_b, D_b, eigs_b = pdft.build_orbitals(F_b, self.A, self.nbeta)
        # Exchange correlation energy/matrix
        self.Vpot.set_D([D_a, D_b])
        self.Vpot.properties()[0].set_pointers(D_a, D_b)

        Vks_a = V_a
        Vks_b = V_b
        Vks_a.axpy(1.0, self.V)
        Vks_b.axpy(1.0, self.V)
        self.Da             = D_a
        self.Db             = D_b
        self.energy         = SCF_E
        self.eig_a          = eigs_a
        self.eig_b          = eigs_b
        self.Fa             = F_a
        self.Fb             = F_b
        self.vks_a          = Vks_a
        self.vks_b          = Vks_b
        self.Ca             = C_a
        self.Cb             = C_b
        self.Cocca          = Cocc_a
        self.Coccb          = Cocc_b

        return

class Inverser(pdft.U_Embedding):
    def __init__(self, molecule, input_wfn, vp_basis=None, ortho_basis=False, v0="FermiAmaldi"):
        super().__init__([], molecule, vp_basis=vp_basis)

        self.input_wfn = input_wfn

        # Get reference vxc
        self.input_vxc_a = None
        self.input_vxc_b = None
        self.input_vxc_cube = None # To be implemented.
        try:
            self.get_input_vxc()
        except:
            print("There is no Vpotential or vxc for input wfn.")

        # v_output = [v_output_a, v_output_b]
        self.v_output = np.zeros(int(self.vp_basis.nbf)*2)
        self.v_output_a = None
        self.v_output_b = None
        # From WuYang (25) vxc = v_output + vH[n_input-n] - 1/N*vH[n_input]
        self.vxc_a_grid = None
        self.vxc_b_grid = None

        # v_esp_input = - esp of input
        self.esp_input = None
        self.get_input_esp()

        # v_ext
        self.vext = None
        self.vH_input = None
        self.v0_input_Fock = None
        self.get_vH_vext()

        # v0
        self.v0 = v0
        if self.v0 == "FermiAmaldi":
            self.get_FermiAmaldi_v0()
        elif self.v0 == "Hartree":
            self.get_Hartree_v0()

        # vH_mol
        self.vH_mol = None

        # Orthogonal basis set
        self.ortho_basis = ortho_basis

        # TSVD. The index of svd array from where to be cut.
        self.svd_index = None

        # Counter
        self.L_counter = 0
        self.grad_counter = 0
        self.hess_counter = 0

    def get_input_vxc(self):
        self.input_vxc_a, self.input_vxc_b = pdft.U_xc(self.input_wfn.Da().np, self.input_wfn.Db().np,
                                                       # self.molecule.Vpot)[-1]
                                                       self.input_wfn.V_potential())[-1]

    def get_input_esp(self):
        # assert self.esp_input is None

        nthreads = psi4.get_num_threads()
        psi4.set_num_threads(1)
        print("ESP fitting starts. This might take a while.")
        x, y, z, _ = self.molecule.Vpot.get_np_xyzw()
        grid = np.array([x, y, z])
        grid = psi4.core.Matrix.from_array(grid.T)
        assert grid.shape[1] == 3, "Grid should be N*3 np.array"

        esp_calculator = psi4.core.ESPPropCalc(self.input_wfn)
        self.esp_input = - esp_calculator.compute_esp_over_grid_in_memory(grid).np
        print("ESP fitting done")
        psi4.set_num_threads(nthreads)
        return

    def get_mol_vH(self):
        self.molecule.update_wfn_info()
        nthreads = psi4.get_num_threads()
        psi4.set_num_threads(1)
        print("ESP fitting starts. This might take a while.")

        x, y, z, _ = self.molecule.Vpot.get_np_xyzw()
        grid = np.array([x, y, z])
        grid = psi4.core.Matrix.from_array(grid.T)
        assert grid.shape[1] == 3, "Grid should be N*3 np.array"

        esp_calculator = psi4.core.ESPPropCalc(self.molecule.wfn)
        self.vH_mol = - esp_calculator.compute_esp_over_grid_in_memory(grid).np - self.vext

        print("ESP fitting done")
        psi4.set_num_threads(nthreads)
        return

    def get_vH_vext(self, grid=None, Vpot=None):

        # Get vext ---------------------------------------------------
        natom = self.molecule.wfn.molecule().natom()
        nuclear_xyz = self.molecule.wfn.molecule().full_geometry().np

        Z = np.zeros(natom)
        # index list of real atoms. To filter out ghosts.
        zidx = []
        for i in range(natom):
            Z[i] = self.molecule.wfn.molecule().Z(i)
            if Z[i] != 0:
                zidx.append(i)

        if grid is None:
            if Vpot is None:
                grid = np.array(self.molecule.Vpot.get_np_xyzw()[:-1]).T
                grid = psi4.core.Matrix.from_array(grid)
                assert grid.shape[1] == 3
            else:
                grid = np.array(Vpot.get_np_xyzw()[:-1]).T
                grid = psi4.core.Matrix.from_array(grid)
                assert grid.shape[1] == 3
        vext = np.zeros(grid.shape[0])
        # Go through all real atoms
        for i in range(len(zidx)):
            R = np.sqrt(np.sum((grid - nuclear_xyz[zidx[i], :])**2, axis=1))
            vext += Z[zidx[i]]/R
            vext[R < 1e-15] = 0

        self.vext = -vext

        # Get vH_input and vFermiAmaldi_fock ---------------------------------------------------
        self.vH_input = self.esp_input - self.vext
        # self.v0_input_Fock = self.molecule.grid_to_fock((nocc-1)/nocc*self.vH_input)
        return

    def change_v0(self, v0: str):
        self.v0 = v0
        if self.v0 == "FermiAmaldi":
            self.get_FermiAmaldi_v0()
        elif self.v0 == "Hartree":
            self.get_Hartree_v0()

    def change_orthogonality(self, ortho: bool):
        self.ortho_basis = ortho
        self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap(self.molecule.wfn.basisset(),
                                                                         self.molecule.wfn.basisset(),
                                                                         self.vp_basis.wfn.basisset()))
        if self.ortho_basis:
            self.three_overlap = np.einsum("ijk,kl->ijl", self.three_overlap, self.vp_basis.A.np)
    

    def get_FermiAmaldi_v0(self):
        """
        Transfer v0 from v_FermiAmaldi to v_H_exact.
        :return:
        """
        nocc = self.molecule.ndocc
        self.v0_input_Fock = (nocc-1)/nocc*self.molecule.grid_to_fock(self.vH_input)
        return

    def get_Hartree_v0(self):
        """
        Transfer v0 from v_FermiAmaldi to v_H_exact.
        :return:
        """
        self.v0_input_Fock = self.molecule.grid_to_fock(self.vH_input)
        return


    def get_vxc(self):
        """
        WuYang (25)
        """
        nbf = int(self.v_output.shape[0] / 2)
        self.v_output_a = self.v_output[:nbf]
        self.v_output_b = self.v_output[nbf:]

        # Get vH_mol
        self.get_mol_vH()

        nocc = self.molecule.ndocc
        if self.ortho_basis:
            self.vxc_a_grid = self.vp_basis.to_grid(np.dot(self.vp_basis.A.np, self.v_output_a))

            self.vxc_b_grid = self.vp_basis.to_grid(np.dot(self.vp_basis.A.np, self.v_output_b))


        else:
            self.vxc_a_grid = self.vp_basis.to_grid(self.v_output_a)

            self.vxc_b_grid = self.vp_basis.to_grid(self.v_output_b)

        if self.v0 == "FermiAmaldi":
            self.vxc_a_grid += (nocc-1)/nocc*self.vH_input - self.vH_mol
            self.vxc_b_grid += (nocc-1)/nocc*self.vH_input - self.vH_mol

        elif self.v0 == "Hartree":
            self.vxc_a_grid += self.vH_input - self.vH_mol
            self.vxc_b_grid += self.vH_input - self.vH_mol
        return


    def Lagrangian_WuYang(self, v=None):
        """
        L = - <T> - \int (vks_a*(n_a-n_a_input)+vks_b*(n_b-n_b_input))
        :return: L
        """
        self.L_counter += 1

        if self.three_overlap is None:
            self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap(self.molecule.wfn.basisset(),
                                                                             self.molecule.wfn.basisset(),
                                                                             self.vp_basis.wfn.basisset()))
            if self.ortho_basis:
                
                self.three_overlap = np.einsum("ijk,kl->ijl", self.three_overlap, self.vp_basis.A.np)

        if v is not None:
            nbf = int(v.shape[0]/2)
            v_output_a = v[:nbf]
            v_output_b = v[nbf:]

            Vks_a = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_a) + self.v0_input_Fock)
            Vks_b = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_b) + self.v0_input_Fock)
            self.molecule.scf_inversion(1000, [Vks_a, Vks_b])

        L = - self.molecule.T.vector_dot(self.molecule.Da) - self.molecule.T.vector_dot(self.molecule.Db)
        L += - self.molecule.vks_a.vector_dot(self.molecule.Da) - self.molecule.vks_b.vector_dot(self.molecule.Db)
        L += self.molecule.vks_a.vector_dot(self.input_wfn.Da()) + self.molecule.vks_b.vector_dot(self.input_wfn.Db())

        return L

    def grad_WuYang(self, v=None):
        """
        grad_a = dL/dvxc_a = - (n_a-n_a_input)
        grad_b = dL/dvxc_b = - (n_b-n_b_input)
        :return:
        """
        self.grad_counter += 1

        if v is not None:
            nbf = int(v.shape[0] / 2)
            v_output_a = v[:nbf]
            v_output_b = v[nbf:]

            Vks_a = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_a) + self.v0_input_Fock)
            Vks_b = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_b) + self.v0_input_Fock)

            self.molecule.scf_inversion(100, [Vks_a, Vks_b])

        dDa = self.input_wfn.Da().np - self.molecule.Da.np
        dDb = self.input_wfn.Db().np - self.molecule.Db.np

        grad_a = np.einsum("uv,uvi->i", dDa, self.three_overlap, optimize=True)
        grad_b = np.einsum("uv,uvi->i", dDb, self.three_overlap, optimize=True)

        grad = np.concatenate((grad_a, grad_b))
        return grad

    def hess_WuYang(self, v=None):
        """
        hess: WuYang (21)
        """
        self.hess_counter += 1
        if v is not None:
            nbf = int(v.shape[0] / 2)
            v_output_a = v[:nbf]
            v_output_b = v[nbf:]

            Vks_a = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_a) + self.v0_input_Fock)
            Vks_b = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_b) + self.v0_input_Fock)

            self.molecule.scf_inversion(100, [Vks_a, Vks_b])

        epsilon_occ_a = self.molecule.eig_a.np[:self.molecule.nalpha, None]
        epsilon_occ_b = self.molecule.eig_b.np[:self.molecule.nbeta, None]
        epsilon_unocc_a = self.molecule.eig_a.np[self.molecule.nalpha:]
        epsilon_unocc_b = self.molecule.eig_b.np[self.molecule.nbeta:]
        epsilon_a = epsilon_occ_a - epsilon_unocc_a
        epsilon_b = epsilon_occ_b - epsilon_unocc_b

        hess = np.zeros((self.vp_basis.nbf*2, self.vp_basis.nbf*2))
        # Alpha electrons
        hess[0:self.vp_basis.nbf, 0:self.vp_basis.nbf] = - self.molecule.omega * np.einsum('ai,bj,ci,dj,ij,abm,cdn -> mn',
                                                                                             self.molecule.Ca.np[:, :self.molecule.nalpha],
                                                                                             self.molecule.Ca.np[:, self.molecule.nalpha:],
                                                                                             self.molecule.Ca.np[:, :self.molecule.nalpha],
                                                                                             self.molecule.Ca.np[:, self.molecule.nalpha:],
                                                                                             np.reciprocal(epsilon_a), self.three_overlap,
                                                                                             self.three_overlap, optimize=True)
        # Beta electrons
        hess[self.vp_basis.nbf:, self.vp_basis.nbf:] = - self.molecule.omega * np.einsum('ai,bj,ci,dj,ij,abm,cdn -> mn',
                                                                                           self.molecule.Cb.np[:, :self.molecule.nbeta],
                                                                                           self.molecule.Cb.np[:, self.molecule.nbeta:],
                                                                                           self.molecule.Cb.np[:, :self.molecule.nbeta],
                                                                                           self.molecule.Cb.np[:, self.molecule.nbeta:],
                                                                                           np.reciprocal(epsilon_b),self.three_overlap,
                                                                                           self.three_overlap, optimize=True)
        hess = (hess + hess.T)

        # Ugly TSVD preparation
        # hess = np.linalg.inv(np.linalg.pinv(hess, rcond=1e-3))

        return hess

    def check_gradient_WuYang(self, dv=None):
        if self.three_overlap is None:
            self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap(self.molecule.wfn.basisset(),
                                                                             self.molecule.wfn.basisset(),
                                                                             self.vp_basis.wfn.basisset()))
            if self.ortho_basis:
                
                self.three_overlap = np.einsum("ijk,kl->ijl", self.three_overlap, self.vp_basis.A.np)

        Vks_a = psi4.core.Matrix.from_array(self.v0_input_Fock)
        Vks_b = psi4.core.Matrix.from_array(self.v0_input_Fock)
        self.molecule.scf_inversion(100, [Vks_a, Vks_b])

        L = self.Lagrangian_WuYang()
        grad = self.grad_WuYang()

        if dv is None:
            dv = 1e-7*np.ones_like(self.v_output)

        grad_app = np.zeros_like(dv)

        for i in range(dv.shape[0]):
            dvi = np.zeros_like(dv)
            dvi[i] = dv[i]

            L_new = self.Lagrangian_WuYang(dvi)

            grad_app[i] = (L_new-L) / dvi[i]

            # print(L_new, L, i + 1, "out of ", dv.shape[0])

        print(np.sum(grad*grad_app)/np.linalg.norm(grad)/np.linalg.norm(grad_app))
        print(np.linalg.norm(grad_app-grad))

        return grad, grad_app

    def check_hess_WuYang(self, dv=None):
        if self.three_overlap is None:
            self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap(self.molecule.wfn.basisset(),
                                                                             self.molecule.wfn.basisset(),
                                                                             self.vp_basis.wfn.basisset()))
            if self.ortho_basis:
                
                self.three_overlap = np.einsum("ijk,kl->ijl", self.three_overlap, self.vp_basis.A.np)

        Vks_a = psi4.core.Matrix.from_array(self.v0_input_Fock)
        Vks_b = psi4.core.Matrix.from_array(self.v0_input_Fock)
        self.molecule.scf_inversion(100, [Vks_a, Vks_b])

        hess = self.hess_WuYang()
        grad = self.grad_WuYang()

        if dv is None:
            dv = 1e-7*np.ones_like(self.v_output)

        hess_app = np.zeros_like(hess)

        for i in range(grad.shape[0]):
            dvi = np.zeros_like(dv)
            dvi[i] = dv[i]

            grad_new = self.grad_WuYang(dvi)

            hess_app[i,:] = (grad_new - grad)/dv[i]

        hess_app = 0.5 * (hess_app + hess_app.T)
        print(np.trace(hess_app.dot(hess.T))/np.linalg.norm(hess_app)/np.linalg.norm(hess))
        print(np.linalg.norm(hess - hess_app))

        return hess, hess_app

    def find_vxc_scipy_WuYang(self, maxiter=14000, opt_method="BFGS", opt=None):

        if self.three_overlap is None:
            self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap(self.molecule.wfn.basisset(),
                                                                             self.molecule.wfn.basisset(),
                                                                             self.vp_basis.wfn.basisset()))
            if self.ortho_basis:
                
                self.three_overlap = np.einsum("ijk,kl->ijl", self.three_overlap, self.vp_basis.A.np)
        print("Zero the old result for a new calculation..")
        self.v_output = np.zeros_like(self.v_output)

        Vks_a = psi4.core.Matrix.from_array(self.v0_input_Fock)
        Vks_b = psi4.core.Matrix.from_array(self.v0_input_Fock)
        self.molecule.scf_inversion(100, [Vks_a, Vks_b])

        print("<<<<<<<<<<<<<<<<<<<<<<WuYang vxc Inversion<<<<<<<<<<<<<<<<<<<")

        dDa = self.input_wfn.Da().np - self.molecule.Da.np
        dDb = self.input_wfn.Db().np - self.molecule.Db.np
        dn = self.molecule.to_grid(dDa+dDb)
        print("|n| before", np.sum(np.abs(dn)*self.molecule.w))
        if opt is None:
            opt = {
                "disp": True,
                "maxiter": maxiter,
                # "eps": 1e-7
                "norm": 2,
                "gtol": 1e-7
            }

        vp_array = optimizer.minimize(self.Lagrangian_WuYang, self.v_output,
                                      jac=self.grad_WuYang,
                                      hess=self.hess_WuYang,
                                      method=opt_method,
                                      options=opt)
        nbf = int(vp_array.x.shape[0] / 2)
        v_output_a = vp_array.x[:nbf]
        v_output_b = vp_array.x[nbf:]

        Vks_a = psi4.core.Matrix.from_array(
            np.einsum("ijk,k->ij", self.three_overlap, v_output_a) + self.v0_input_Fock)
        Vks_b = psi4.core.Matrix.from_array(
            np.einsum("ijk,k->ij", self.three_overlap, v_output_b) + self.v0_input_Fock)
        self.molecule.scf_inversion(100, [Vks_a, Vks_b])

        dDa = self.input_wfn.Da().np - self.molecule.Da.np
        dDb = self.input_wfn.Db().np - self.molecule.Db.np
        dn = self.molecule.to_grid(dDa+dDb)
        print("|n| after", np.sum(np.abs(dn)*self.molecule.w), "L after", vp_array.fun)
        print("Ts", self.molecule.Da.vector_dot(self.molecule.T)+self.molecule.Db.vector_dot(self.molecule.T))
        print("dTs", np.trace(np.dot(self.input_wfn.Da().np+self.input_wfn.Db().np-
                                     self.molecule.Da.np-self.molecule.Db.np, self.molecule.T.np)))
        print("eigenA")
        print(self.input_wfn.epsilon_a().np[:self.molecule.nalpha])
        print(self.molecule.eig_a.np[:self.molecule.nalpha])
        print("wfnDiff", self.input_wfn.Ca().vector_dot(self.molecule.Ca)
              /np.linalg.norm(self.input_wfn.Ca().np)/np.linalg.norm(self.molecule.Ca.np))
        # Update info
        self.v_output = vp_array.x
        self.get_vxc()
        return vp_array

    def find_vxc_manualNewton(self, maxiter=49, svd_rcond=None, mu=1e-4,
                              svd_parameter=None, back_tracking_method="L",
                              BT_beta_threshold=1e-7, rho_conv_threshold=1e-3):
        if self.three_overlap is None:
            self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap(self.molecule.wfn.basisset(),
                                                                             self.molecule.wfn.basisset(),
                                                                             self.vp_basis.wfn.basisset()))
            if self.ortho_basis:

                self.three_overlap = np.einsum("ijk,kl->ijl", self.three_overlap, self.vp_basis.A.np)

        print("Zero the old result for a new calculation..")
        self.v_output = np.zeros_like(self.v_output)

        Vks_a = psi4.core.Matrix.from_array(self.v0_input_Fock)
        Vks_b = psi4.core.Matrix.from_array(self.v0_input_Fock)
        self.molecule.scf_inversion(100, [Vks_a, Vks_b])

        ## Tracking rho and changing beta
        n = self.molecule.to_grid(self.molecule.Da.np, Duv_b=self.molecule.Db.np)
        n_input = self.molecule.to_grid(self.input_wfn.Da().np+self.input_wfn.Db().np)
        dn_before = np.sum(np.abs(n - n_input) * self.molecule.w)
        L_old = self.Lagrangian_WuYang()

        print("Initial dn:", dn_before, "Initial L:",L_old)

        BT_converge_flag = False
        beta = 2

        ls = 0
        ns = 0

        cycle_n = np.inf
        print("<<<<<<<<<<<<<<<<<<<<<<WuYang vxc Inversion manual Newton<<<<<<<<<<<<<<<<<<<")
        for scf_step in range(1, maxiter + 1):

            # if scf_step == 2:
            #     self.update_v0()

            hess = self.hess_WuYang(v=self.v_output)
            jac = self.grad_WuYang(v=self.v_output)

            if svd_rcond is None:
                dv = -np.linalg.solve(hess, jac)
            # The svd pseudo-inverse could hopefully be avoided, with orthogonal v_basis.
            elif type(svd_rcond) is float:
                hess_inv = np.linalg.pinv(hess, rcond=svd_rcond)
                dv = -np.dot(hess_inv, jac)
            elif type(svd_rcond) is int:
                s = np.linalg.svd(hess)[1]
                self.svd_index = svd_rcond
                svd = s[self.svd_index]
                # print(svd)
                svd = svd * 0.9999 / s[0]
                hess_inv = np.linalg.pinv(hess, rcond=svd)
                dv = -np.dot(hess_inv, jac)
            elif svd_rcond == "input_once":
                s = np.linalg.svd(hess)[1]

                if scf_step == 1:
                    print(repr(s))

                    plt.figure(dpi=200)
                    # plt.scatter(range(s.shape[0]), np.log10(s), s=2)
                    plt.scatter(range(s.shape[0]), np.log10(s), s=2)
                    plt.title(str(self.ortho_basis))
                    plt.show()
                    plt.close()

                    self.svd_index = int(input("Enter svd cut index (0-based indexing): "))
                svd = s[self.svd_index]
                # print(svd)
                svd = svd * 0.9999 / s[0]

                hess_inv = np.linalg.pinv(hess, rcond=svd)
                dv = -np.dot(hess_inv, jac)
            elif svd_rcond == "input_every":
                s = np.linalg.svd(hess)[1]

                # print(repr(s))

                plt.figure(dpi=200)
                # plt.scatter(range(s.shape[0]), np.log10(s), s=2)
                plt.scatter(range(s.shape[0]), np.log10(s), s=2)
                plt.title(str(self.ortho_basis))
                plt.show()
                plt.close()

                self.svd_index = int(input("Enter svd cut index (0-based indexing): "))
                svd = s[self.svd_index]
                # print(svd)
                svd = svd * 0.95 / s[0]

                hess_inv = np.linalg.pinv(hess, rcond=svd)
                dv = -np.dot(hess_inv, jac)

            elif svd_rcond == "increase":
                if svd_parameter is None:
                    svd_parameter = [0, 10] # [Starting percentile, update step ]
                s = np.linalg.svd(hess)[1]

                if scf_step == 1:
                    print(repr(s))

                    plt.figure(dpi=200)
                    # plt.scatter(range(s.shape[0]), np.log10(s), s=2)
                    plt.scatter(range(s.shape[0]), np.log10(s), s=2)
                    plt.title(str(self.ortho_basis))
                    plt.show()
                    plt.close()

                    end = int(input("Enter svd_rcond end number: "))
                    svd_parameter.append(end)

                if scf_step == 1:
                    self.svd_index = int(s.shape[0]*svd_parameter[0])
                elif beta <= BT_beta_threshold:
                    # SVD move on
                    self.svd_index += int(svd_parameter[1])

                if self.svd_index >= svd_parameter[-1]:
                    self.svd_index = svd_parameter[-1]
                    BT_converge_flag = True
                    
                svd = s[self.svd_index-1]
                # print(svd)
                svd = svd * 0.95 / s[0]

                hess_inv = np.linalg.pinv(hess, rcond=svd)
                dv = -np.dot(hess_inv, jac)

            elif svd_rcond == "cycle_segment":
                if svd_parameter is None:
                    # [seg step length, current start idx]
                    svd_parameter = [0.1, 0]

                s = np.linalg.svd(hess)[1]
                # SVD move on
                s_shape = s.shape[0]

                if scf_step == 1:
                    plt.figure(dpi=200)
                    # plt.scatter(range(s.shape[0]), np.log10(s), s=2)
                    plt.scatter(range(s.shape[0]), np.log10(s), s=2)
                    plt.title(str(self.ortho_basis))
                    plt.show()
                    plt.close()

                    end = int(input("Enter svd_rcond end number: "))
                    svd_parameter.append(end)

                # another cycle
                elif beta <= BT_beta_threshold:
                    if svd_parameter[-1] == -1:
                        if svd_parameter[1] >= s_shape:
                            svd_parameter[1] = 0
                    else:
                        assert svd_parameter[-1] > 0
                        if svd_parameter[1] >= svd_parameter[-1]:
                            svd_parameter[1] = 0

                    start = svd_parameter[1]
                    svd_parameter[1] += int(svd_parameter[0]*s_shape)

                    if svd_parameter[-1] == -1:
                        if svd_parameter[1] > s_shape:
                            svd_parameter[1] = s_shape
                    else:
                        assert svd_parameter[-1] > 0
                        if svd_parameter[1] > svd_parameter[-1]:
                            svd_parameter[1] = svd_parameter[-1]

                    end = svd_parameter[1]

                self.svd_index = [start, end]

                hess_inv = pdft.inv_pinv(hess, self.svd_index[0], self.svd_index[1])
                dv = -np.dot(hess_inv, jac)

            elif svd_rcond == "segment_cycle":
                if svd_parameter is None:
                    # [seg step length, current start idx]
                    svd_parameter = [0.1, 0]

                s = np.linalg.svd(hess)[1]
                # SVD move on
                s_shape = s.shape[0]

                if scf_step == 1:
                    print(repr(s))

                    plt.figure(dpi=200)
                    # plt.scatter(range(s.shape[0]), np.log10(s), s=2)
                    plt.scatter(range(s.shape[0]), np.log10(s), s=1)
                    plt.title(str(self.ortho_basis))
                    plt.show()
                    plt.close()

                    end = int(input("Enter svd_rcond end number: "))
                    svd_parameter.append(end)

                # another cycle
                if svd_parameter[-1] == -1:
                    if svd_parameter[1] >= s_shape:
                        svd_parameter[1] = 0
                else:
                    assert svd_parameter[-1] > 0
                    if svd_parameter[1] >= svd_parameter[-1]:
                        svd_parameter[1] = 0

                start = svd_parameter[1]
                svd_parameter[1] += int(svd_parameter[0]*s_shape)

                if svd_parameter[-1] == -1:
                    if svd_parameter[1] > s_shape:
                        svd_parameter[1] = s_shape
                else:
                    assert svd_parameter[-1] > 0
                    if svd_parameter[1] > svd_parameter[-1]:
                        svd_parameter[1] = svd_parameter[-1]

                end = svd_parameter[1]

                self.svd_index = [start, end]

                hess_inv = pdft.inv_pinv(hess, self.svd_index[0], self.svd_index[1])

                dv = -np.dot(hess_inv, jac)

            elif svd_rcond == "input_segment_cycle":
                s = np.linalg.svd(hess)[1]
                # SVD move on
                s_shape = s.shape[0]

                if scf_step == 1:
                    print(repr(s))

                    plt.figure(dpi=200)
                    # plt.scatter(range(s.shape[0]), np.log10(s), s=2)
                    plt.scatter(range(s.shape[0]), np.log10(s), s=1)
                    plt.title(str(self.ortho_basis))
                    plt.show()
                    plt.close()

                    # Get the segment partation.
                    svd_parameter = [0]
                    ns = int(input("Enter number of segments: "))
                    for i in range(0, ns):
                        ele = int(input()) + 1
                        svd_parameter.append(ele)  # adding the element

                # Cycle number
                # ns=4
                # [0-10,10-20,20-30,30-40]
                # svd_parameter = [0,10,20,30,40]
                # start = [0, 10, 20, 30]
                # end = [10, 20, 30, 40]
                # cn = [1, 2, 3, 0]
                cn = scf_step % ns

                start = svd_parameter[:-1][cn-1]
                end = svd_parameter[1:][cn-1]

                self.svd_index = [start, end]

                hess_inv = pdft.inv_pinv(hess, self.svd_index[0], self.svd_index[1])

                dv = -np.dot(hess_inv, jac)

            elif svd_rcond == "search_segment_cycle":
                s = np.linalg.svd(hess)[1]
                # SVD move on
                s_shape = s.shape[0]

                if svd_parameter is None:
                    svd_parameter = 10  # the cutoff threshold.

                # Segmentation
                if scf_step==1 or ns==ls:
                    if np.sum(np.abs(cycle_n-n)*self.molecule.w) < rho_conv_threshold:
                        print("Break because n is not improved in this segment cycle.", np.abs(cycle_n-n))
                        break
                    cycle_n = n
                    seg_list = [0]
                    for i in range(1,s_shape):
                        if s[i-1]/s[i] > svd_parameter:
                            print(s[i], s[i-1])
                            seg_list.append(i)
                    seg_list.append(s_shape)
                    ls = len(seg_list) - 1  # length of segments
                    ns = 0  # segment number starts from 0
                    print("\nSegment", seg_list)
                    print("\n")
                start = seg_list[ns]
                end = seg_list[ns+1]

                self.svd_index = [start, end]

                hess_inv = pdft.inv_pinv(hess, self.svd_index[0], self.svd_index[1])

                dv = -np.dot(hess_inv, jac)

                ns += 1

            elif svd_rcond == "segments":
                if svd_parameter is None:
                    svd_parameter = [10, 0]
                s = np.linalg.svd(hess)[1]
                s_shape = s.shape[0]
                start = svd_parameter[1]
                svd_parameter[1] += int(s_shape / svd_parameter[0])
                if svd_parameter[1] > s_shape:
                    svd_parameter[1] = s_shape
                    BT_converge_flag = True

                end = svd_parameter[1]

                self.svd_index = [start, end+1]

                hess_inv = pdft.inv_pinv(hess, start, end)
                dv = -np.dot(hess_inv, jac)

            elif svd_rcond == "find_optimal_w_bruteforce":
                back_tracking_method = None
                mu = 1e-3
                rcondlist = np.zeros(100)
                dnlist = np.zeros(100)
                Llist = np.zeros(100)

                idx = 0

                for rcond_idx in np.linspace(0, 10, 100):
                    hess_inv = np.linalg.pinv(hess, rcond=10 ** -rcond_idx)
                    dv = -np.dot(hess_inv, jac)

                    beta = 2
                    while True:
                        beta *= 0.5
                        if beta < BT_beta_threshold:
                            break
                        # Traditional WuYang
                        v_temp = self.v_output + beta * dv
                        L = self.Lagrangian_WuYang(v=v_temp)
                        # n = self.molecule.to_grid(self.molecule.Da.np, Duv_b=self.molecule.Db.np)
                        # dn = np.sum(np.abs(n - n_input) * self.molecule.w)
                        # print(beta, L - L_old, dn - dn_before, mu * beta * np.sum(jac * dv))
                        if L - L_old <= mu * beta * np.sum(jac * dv) and \
                                beta * np.sum(jac * dv) < 0:
                            # dn - dn_before <= - mu * beta * dn_before and \
                            rcondlist[idx] = 10 ** -rcond_idx
                            # dnlist[idx] = dn
                            Llist[idx] = L
                            break
                    idx += 1

                L_idx = np.argmin(L)
                print(rcond_idx, L_idx)
                hess_inv = np.linalg.pinv(hess,
                                          rcond=10 ** -np.linspace(0, 10, 100)[L_idx])
                dv = - np.dot(hess_inv, jac)
                beta = 2
                while True:
                    beta *= 0.5
                    if beta < BT_beta_threshold:
                        break
                    # Traditional WuYang
                    v_temp = self.v_output + beta * dv
                    L = self.Lagrangian_WuYang(v=v_temp)
                    print(beta, L - L_old, mu * beta * np.sum(jac * dv))
                    if L - L_old <= mu * beta * np.sum(jac * dv) and beta * np.sum(jac * dv) < 0:
                        L_old = L
                        self.v_output = v_temp
                        n = self.molecule.to_grid(self.molecule.Da.np+self.molecule.Db.np)
                        dn_before = np.sum(np.abs(n_input - n) * self.molecule.w)
                        break

            if back_tracking_method == "L":
                beta = 2
                while True:
                    beta *= 0.5
                    if beta < BT_beta_threshold:
                        BT_converge_flag = True
                        break
                    # Traditional WuYang
                    v_temp = self.v_output + beta * dv
                    L = self.Lagrangian_WuYang(v=v_temp)
                    print(beta, L - L_old, mu * beta * np.sum(jac * dv))
                    if L - L_old <= mu * beta * np.sum(jac * dv) and beta * np.sum(jac * dv) < 0:
                        L_old = L
                        self.v_output = v_temp
                        n = self.molecule.to_grid(self.molecule.Da.np+self.molecule.Db.np)
                        dn_before = np.sum(np.abs(n_input - n) * self.molecule.w)
                        break
            elif back_tracking_method == "D":
                beta = 2
                while True:
                    beta *= 0.5
                    if beta < BT_beta_threshold:
                        BT_converge_flag = True
                        break
                    # Traditional WuYang
                    v_temp = self.v_output + beta * dv
                    L = self.Lagrangian_WuYang(v=v_temp)
                    n = self.molecule.to_grid(self.molecule.Da.np, Duv_b=self.molecule.Db.np)
                    dn = np.sum(np.abs(n - n_input) * self.molecule.w)
                    print(beta, L - L_old, dn - dn_before, mu * beta * np.sum(jac * dv))
                    if dn - dn_before <= - mu * beta * dn_before and beta * np.sum(jac * dv) < 0:
                        L_old = L
                        dn_before = dn
                        self.v_output = v_temp
                        break
            elif back_tracking_method == "LD":
                beta = 2
                while True:
                    beta *= 0.5
                    if beta < BT_beta_threshold:
                        BT_converge_flag = True
                        break
                    # Traditional WuYang
                    v_temp = self.v_output + beta * dv
                    L = self.Lagrangian_WuYang(v=v_temp)
                    print(beta, L - L_old, mu * beta * np.sum(jac * dv))
                    if L - L_old <= mu * beta * np.sum(jac * dv) and \
                            beta * np.sum(jac * dv) < 0:
                        n = self.molecule.to_grid(self.molecule.Da.np, Duv_b=self.molecule.Db.np)
                        dn = np.sum(np.abs(n - n_input) * self.molecule.w)
                        print(beta, L - L_old, dn - dn_before, mu * beta * np.sum(jac * dv))
                        if dn - dn_before <= - mu * beta * dn_before:
                            L_old = L
                            dn_before = dn
                            self.v_output = v_temp
                            break

            print(
                F'------BT: {back_tracking_method} SVD: {self.svd_index} Reg: {self.regul_const} '
                F'Ortho: {self.ortho_basis} SVDmoveon: {beta < BT_beta_threshold} ------\n'
                F'Iter: {scf_step} beta: {beta} |jac|: {np.linalg.norm(jac)} L: {L_old} d_rho: {dn_before}\n')

            if BT_converge_flag and \
                    not (svd_rcond=="segments" or svd_rcond=="segment_cycle"
                         or svd_rcond=="increase" or svd_rcond=="input_segment_cycle"
                         or svd_rcond=="search_segment_cycle"):
                print("Converge")
                break
            elif dn_before < rho_conv_threshold:
                print("Break because rho difference (cost) is small.")
                break
            elif scf_step == maxiter:
                print("Maximum number of SCF cycles exceeded for vp.")


        if svd_rcond == "find_optimal_w_bruteforce":
            return rcondlist, dnlist, Llist

        print("Evaluation: ", self.L_counter, self.grad_counter, self.hess_counter)
        print("Ts", self.molecule.Da.vector_dot(self.molecule.T)+self.molecule.Db.vector_dot(self.molecule.T))
        print("dTs", np.trace(np.dot(self.input_wfn.Da().np+self.input_wfn.Db().np-
                                     self.molecule.Da.np-self.molecule.Db.np, self.molecule.T.np)))
        print("eigenA input", self.input_wfn.epsilon_a().np[:self.molecule.nalpha])
        print("eigenA mol", self.molecule.eig_a.np[:self.molecule.nalpha])
        print("wfnDiff", self.input_wfn.Ca().vector_dot(self.molecule.Ca)
              /np.linalg.norm(self.input_wfn.Ca().np)/np.linalg.norm(self.molecule.Ca.np))
        self.L_counter = 0
        self.grad_counter = 0
        self.hess_counter = 0

        self.get_vxc()

        return

    def Lagrangian_constrainedoptimization(self, v=None):
        """
        Return Lagrange Multipliers from Nafziger and Jensen's constrained optimization.
        :return: L
        """
        self.L_counter += 1

        if self.three_overlap is None:
            self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap(self.molecule.wfn.basisset(),
                                                                             self.molecule.wfn.basisset(),
                                                                             self.vp_basis.wfn.basisset()))
            if self.ortho_basis:
                
                self.three_overlap = np.einsum("ijk,kl->ijl", self.three_overlap, self.vp_basis.A.np)

        if v is not None:
            nbf = int(v.shape[0]/2)
            v_output_a = v[:nbf]
            v_output_b = v[nbf:]

            Vks_a = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_a) + self.v0_input_Fock)
            Vks_b = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_b) + self.v0_input_Fock)
            self.molecule.scf_inversion(1000, [Vks_a, Vks_b])

        dDa = self.molecule.Da.np - self.input_wfn.Da().np
        dDb = self.molecule.Db.np - self.input_wfn.Db().np
        dD = dDa + dDb

        g_uv = self.CO_weighted_cost()

        L = np.sum(dD * g_uv)

        return L

    def grad_constrainedoptimization(self, v=None):
        """
        To get Jaccobi vector, which is jac_j = sum_i p_i*psi_i*phi_j.
        p_i = sum_j b_ij*phi_j
        MO: psi_i = sum_j C_ji*phi_j
        AO: phi
        --
        A: frag, a: spin, i: MO
        """
        self.grad_counter += 1

        if v is not None:
            nbf = int(v.shape[0] / 2)
            v_output_a = v[:nbf]
            v_output_b = v[nbf:]

            Vks_a = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_a) + self.v0_input_Fock)
            Vks_b = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_b) + self.v0_input_Fock)

            self.molecule.scf_inversion(100, [Vks_a, Vks_b])

        # gradient on grad
        jac_real_up = np.zeros((self.vp_basis.nbf, self.vp_basis.nbf))
        jac_real_down = np.zeros((self.vp_basis.nbf, self.vp_basis.nbf))

        # Pre-calculate \int (n - nf)*phi_u*phi_v
        g_uv = - self.CO_weighted_cost()
        # Fragments
        A = self.molecule
        # spin up
        for i in range(A.nalpha):
            u = 2 * np.einsum("u,v,uv->", A.Cocca.np[:,i], A.Cocca.np[:,i], g_uv)
            LHS = A.Fa.np - A.S.np*A.eig_a.np[i]
            RHS = 4 * np.einsum("uv,v->u", g_uv, A.Cocca.np[:,i]) - 2 * u * np.dot(A.S.np, A.Cocca.np[:,i])
            p = np.linalg.solve(LHS, RHS)
            # s = np.linalg.svd(LHS)[1]
            # if self.svd_index is None:
            #     s = np.linalg.svd(LHS)[1]
            #     print(repr(s))
            #
            #     f, ax = plt.subplots(1,1,dpi=200)
            #     ax.scatter(range(s.shape[0]), np.log10(s), s=3)
            #     ax.set_title(str(self.ortho_basis))
            #     f.show()
            #     plt.close()
            #
            #     self.svd_index = int(input("Enter svd cut index (0-based indexing): "))
            #
            # p = np.dot(np.linalg.pinv(LHS, rcond=s[self.svd_index]*1.01/s[0]), RHS)

            # Gram–Schmidt
            p = p - np.sum(p * np.dot(A.S.np, A.Cocca.np[:,i])) * A.Cocca.np[:,i]
            # assert np.allclose([np.sum(p * np.dot(A.S.np, A.Cocca.np[:,i])), np.linalg.norm(np.dot(LHS,p)-RHS), np.sum(RHS*A.Cocca.np[:,i])], 0, atol=1e-4), \
            #     [np.sum(p * np.dot(A.S.np, A.Cocca.np[:,i])), np.linalg.norm(np.dot(LHS,p)-RHS), np.sum(RHS*A.Cocca.np[:,i])]
            jac_real_up += np.dot(p[:, None], A.Cocca.np[:,i:i+1].T)

        # spin down
        for i in range(A.nbeta):
            u = 2 * np.einsum("u,v,uv->", A.Coccb.np[:,i], A.Coccb.np[:,i], g_uv)
            LHS = A.Fb.np - A.S.np*A.eig_b.np[i]
            RHS = 4 * np.einsum("uv,v->u", g_uv, A.Coccb.np[:,i]) - 2 * u * np.dot(A.S.np, A.Coccb.np[:,i])
            p = np.linalg.solve(LHS, RHS)
            # s = np.linalg.svd(LHS)[1]
            # p = np.dot(np.linalg.pinv(LHS, rcond=s[self.svd_index]/s[0]*1.01), RHS)
            #
            # Gram–Schmidt
            p = p - np.sum(p * np.dot(A.S.np, A.Coccb.np[:,i]))*A.Coccb.np[:,i]
            # assert np.allclose([np.sum(p * np.dot(A.S.np, A.Coccb.np[:,i])), np.linalg.norm(np.dot(LHS,p)-RHS), np.sum(RHS*A.Coccb.np[:,i])], 0, atol=1e-4), \
            #     [np.sum(p * np.dot(A.S.np, A.Coccb.np[:,i])), np.linalg.norm(np.dot(LHS,p)-RHS), np.sum(RHS*A.Coccb.np[:,i])]
            jac_real_down += np.dot(p[:, None], A.Coccb.np[:,i:i+1].T)

        # jac = int jac_real*phi_w
        jac_up = np.einsum("uv,uvw->w", jac_real_up, self.three_overlap)
        jac_down = np.einsum("uv,uvw->w", jac_real_down, self.three_overlap)

        if v is None:
            return np.concatenate((jac_up, jac_down)), LHS, RHS, p
        else:
            return np.concatenate((jac_up, jac_down))

    def hess_constrainedoptimization(self, v=None):
        self.hess_counter += 1

        if v is not None:
            nbf = int(v.shape[0] / 2)
            v_output_a = v[:nbf]
            v_output_b = v[nbf:]

            Vks_a = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_a) + self.v0_input_Fock)
            Vks_b = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_b) + self.v0_input_Fock)

            self.molecule.scf_inversion(100, [Vks_a, Vks_b])

        hess = np.zeros((self.vp_basis.nbf*2, self.vp_basis.nbf*2))

        # Pre-calculate \int (n - nf)*phi_u*phi_v
        g_uv = - self.CO_weighted_cost()
        # Fragments
        A = self.molecule
        # spin up
        for i in range(A.nalpha):
            pi_psi_up = np.zeros((self.vp_basis.nbf, self.vp_basis.nbf))
            psii_psi_up = np.zeros((self.vp_basis.nbf, self.vp_basis.nbf))

            u = 2 * np.einsum("u,v,uv->", A.Cocca.np[:,i], A.Cocca.np[:,i], g_uv)
            LHS = A.Fa.np - A.S.np*A.eig_a.np[i]
            RHS = 4 * np.einsum("uv,v->u", g_uv, A.Cocca.np[:,i]) - 2 * u * np.dot(A.S.np, A.Cocca.np[:,i])
            p = np.linalg.solve(LHS, RHS)
            # s = np.linalg.svd(LHS)[1]
            # if self.svd_index is None:
            #     s = np.linalg.svd(LHS)[1]
            #     print(repr(s))
            #
            #     f, ax = plt.subplots(1,1,dpi=200)
            #     ax.scatter(range(s.shape[0]), np.log10(s), s=3)
            #     ax.set_title(str(self.ortho_basis))
            #     f.show()
            #     plt.close()
            #
            #     self.svd_index = int(input("Enter svd cut index (0-based indexing): "))
            #
            # p = np.dot(np.linalg.pinv(LHS, rcond=s[self.svd_index]*1.01/s[0]), RHS)

            # Gram–Schmidt
            p = p - np.sum(p * np.dot(A.S.np, A.Cocca.np[:,i])) * A.Cocca.np[:,i]
            assert np.allclose([np.sum(p * np.dot(A.S.np, A.Cocca.np[:,i])), np.linalg.norm(np.dot(LHS,p)-RHS), np.sum(RHS*A.Cocca.np[:,i])], 0, atol=1e-4), \
                [np.sum(p * np.dot(A.S.np, A.Cocca.np[:,i])), np.linalg.norm(np.dot(LHS,p)-RHS), np.sum(RHS*A.Cocca.np[:,i])]

            for j in range(A.nbf):
                if i == j:
                    continue
                # p_i * psi_j
                pi_psi_up = np.dot(p[:, None], A.Ca.np[:,j:j+1].T)
                # psi_i * psi_j / (Ei-Ej)
                psii_psi_up = np.dot(A.Cocca.np[:,i:i+1], A.Ca.np[:,j:j+1].T) / (A.eig_a.np[i] - A.eig_a.np[j])
                assert psii_psi_up.shape == pi_psi_up.shape

                hess[0:self.vp_basis.nbf, 0:self.vp_basis.nbf] += np.einsum("mn,uv,mni,uvj->ij", pi_psi_up, psii_psi_up,
                                                                            self.three_overlap, self.three_overlap,
                                                                            optimize=True)

        # spin down
        for i in range(A.nbeta):
            pi_psi_dw = np.zeros((self.vp_basis.nbf, self.vp_basis.nbf))
            psii_psi_dw = np.zeros((self.vp_basis.nbf, self.vp_basis.nbf))

            u = 2 * np.einsum("u,v,uv->", A.Coccb.np[:,i], A.Coccb.np[:,i], g_uv)
            LHS = A.Fb.np - A.S.np*A.eig_b.np[i]
            RHS = 4 * np.einsum("uv,v->u", g_uv, A.Coccb.np[:,i]) - 2 * u * np.dot(A.S.np, A.Coccb.np[:,i])
            p = np.linalg.solve(LHS, RHS)
            # s = np.linalg.svd(LHS)[1]
            # p = np.dot(np.linalg.pinv(LHS, rcond=s[self.svd_index]/s[0]*1.01), RHS)

            # Gram–Schmidt
            p = p - np.sum(p * np.dot(A.S.np, A.Coccb.np[:,i]))*A.Coccb.np[:,i]
            assert np.allclose([np.sum(p * np.dot(A.S.np, A.Coccb.np[:,i])), np.linalg.norm(np.dot(LHS,p)-RHS), np.sum(RHS*A.Coccb.np[:,i])], 0, atol=1e-4), \
                [np.sum(p * np.dot(A.S.np, A.Coccb.np[:,i])), np.linalg.norm(np.dot(LHS,p)-RHS), np.sum(RHS*A.Coccb.np[:,i])]
            for j in range(A.nbf):
                if i == j:
                    continue
                # p_i * psi_j
                pi_psi_dw = np.dot(p[:, None], A.Cb.np[:, j:j + 1].T)
                # psi_i * psi_j / (Ei-Ej)
                psii_psi_dw = np.dot(A.Coccb.np[:, i:i + 1], A.Cb.np[:, j:j + 1].T) / (A.eig_b.np[i] - A.eig_b.np[j])

                hess[self.vp_basis.nbf:, self.vp_basis.nbf:] += np.einsum("mn,uv,mni,uvj->ij", pi_psi_dw, psii_psi_dw,
                                                                          self.three_overlap, self.three_overlap,
                                                                          optimize=True)
        return hess



    def CO_weighted_cost(self):
        """
        To get the function g_uv = \int w*(n-n_in)*phi_u*phi_v.
        So that the cost function is dD_uv*g_uv.
        The first term in g funtion is -2*C_v*g_uv.
        w = n_mol**-self.CO_weight_exponent
        :return:
        """
        if self.four_overlap is None:
            self.four_overlap = pdft.fouroverlap(self.molecule.wfn, self.molecule.geometry,
                                                 self.molecule.basis, self.molecule.mints)[0]

        D_mol = self.input_wfn.Da().np + self.input_wfn.Db().np

        dDa = self.molecule.Da.np - self.input_wfn.Da().np
        dDb = self.molecule.Db.np - self.input_wfn.Db().np
        dD = dDa + dDb

        g_uv = np.zeros_like(self.molecule.H.np)
        if self.CO_weight_exponent is None:
            return np.einsum("mn,mnuv->uv", dD, self.four_overlap, optimize=True)
        else:
            vpot = self.molecule.Vpot
            points_func = vpot.properties()[0]
            f_grid = np.array([])
            # Loop over the blocks
            for b in range(vpot.nblocks()):
                # Obtain block information
                block = vpot.get_block(b)
                points_func.compute_points(block)
                w = block.w()
                npoints = block.npoints()
                lpos = np.array(block.functions_local_to_global())

                # Compute phi!
                phi = np.array(points_func.basis_values()["PHI"])[:npoints, :lpos.shape[0]]

                # Build a local slice of D
                ldD = dD[(lpos[:, None], lpos)]
                lD_mol = D_mol[(lpos[:, None], lpos)]

                # Copmute dn and n_mol
                n_mol = np.einsum('pm,mn,pn->p', phi, lD_mol, phi)

                g_uv[(lpos[:, None], lpos)] += np.einsum('pm,pn,pu,pv,p,mn,p->uv', phi, phi, phi, phi, (1/n_mol)**self.CO_weight_exponent,
                                  ldD, w, optimize=True)
            return g_uv

    def CO_p(self, v=None):
        if v is not None:
            nbf = int(v.shape[0] / 2)
            v_output_a = v[:nbf]
            v_output_b = v[nbf:]

            Vks_a = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_a) + self.v0_input_Fock)
            Vks_b = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_b) + self.v0_input_Fock)

            self.molecule.scf_inversion(100, [Vks_a, Vks_b])

        p_up = np.zeros((self.molecule.nbf, self.molecule.nalpha))
        p_dw = np.zeros((self.molecule.nbf, self.molecule.nbeta))

        # Pre-calculate \int (n - nf)*phi_u*phi_v
        g_uv = - self.CO_weighted_cost()
        # Fragments
        A = self.molecule
        # spin up
        for i in range(A.nalpha):
            u = 2 * np.einsum("u,v,uv->", A.Cocca.np[:,i], A.Cocca.np[:,i], g_uv)
            LHS = A.Fa.np - A.S.np*A.eig_a.np[i]
            RHS = 4 * np.einsum("uv,v->u", g_uv, A.Cocca.np[:,i]) - 2 * u * np.dot(A.S.np, A.Cocca.np[:,i])
            p = np.linalg.solve(LHS, RHS)
            # s = np.linalg.svd(LHS)[1]
            # if self.svd_index is None:
            #     s = np.linalg.svd(LHS)[1]
            #     print(repr(s))
            #
            #     f, ax = plt.subplots(1,1,dpi=200)
            #     ax.scatter(range(s.shape[0]), np.log10(s), s=3)
            #     ax.set_title(str(self.ortho_basis))
            #     f.show()
            #     plt.close()
            #
            #     self.svd_index = int(input("Enter svd cut index (0-based indexing): "))
            #
            # p = np.dot(np.linalg.pinv(LHS, rcond=s[self.svd_index]*1.01/s[0]), RHS)

            # Gram–Schmidt
            p = p - np.sum(p * np.dot(A.S.np, A.Cocca.np[:,i])) * A.Cocca.np[:,i]
            assert np.allclose([np.sum(p * np.dot(A.S.np, A.Cocca.np[:,i])), np.linalg.norm(np.dot(LHS,p)-RHS), np.sum(RHS*A.Cocca.np[:,i])], 0, atol=1e-4), \
                [np.sum(p * np.dot(A.S.np, A.Cocca.np[:,i])), np.linalg.norm(np.dot(LHS,p)-RHS), np.sum(RHS*A.Cocca.np[:,i])]

            p_up[:, i] = p

        # spin down
        for i in range(A.nbeta):
            u = 2 * np.einsum("u,v,uv->", A.Coccb.np[:,i], A.Coccb.np[:,i], g_uv)
            LHS = A.Fb.np - A.S.np*A.eig_b.np[i]
            RHS = 4 * np.einsum("uv,v->u", g_uv, A.Coccb.np[:,i]) - 2 * u * np.dot(A.S.np, A.Coccb.np[:,i])
            p = np.linalg.solve(LHS, RHS)
            # s = np.linalg.svd(LHS)[1]
            # p = np.dot(np.linalg.pinv(LHS, rcond=s[self.svd_index]/s[0]*1.01), RHS)

            # Gram–Schmidt
            p = p - np.sum(p * np.dot(A.S.np, A.Coccb.np[:,i]))*A.Coccb.np[:,i]
            assert np.allclose([np.sum(p * np.dot(A.S.np, A.Coccb.np[:,i])), np.linalg.norm(np.dot(LHS,p)-RHS), np.sum(RHS*A.Coccb.np[:,i])], 0, atol=1e-4), \
                [np.sum(p * np.dot(A.S.np, A.Coccb.np[:,i])), np.linalg.norm(np.dot(LHS,p)-RHS), np.sum(RHS*A.Coccb.np[:,i])]

            p_dw[:, i] = p
        return p_up, p_dw

    def check_gradient_constrainedoptimization(self, dv=None):
        if self.three_overlap is None:
            self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap(self.molecule.wfn.basisset(),
                                                                             self.molecule.wfn.basisset(),
                                                                             self.vp_basis.wfn.basisset()))
            if self.ortho_basis:
                
                self.three_overlap = np.einsum("ijk,kl->ijl", self.three_overlap, self.vp_basis.A.np)

        Vks_a = psi4.core.Matrix.from_array(self.v0_input_Fock)
        Vks_b = psi4.core.Matrix.from_array(self.v0_input_Fock)
        self.molecule.scf_inversion(100, [Vks_a, Vks_b])

        L = self.Lagrangian_constrainedoptimization()
        grad = self.grad_constrainedoptimization()[0]

        if dv is None:
            dv = 1e-7*np.ones_like(self.v_output)

        grad_app = np.zeros_like(dv)

        for i in range(dv.shape[0]):
            dvi = np.zeros_like(dv)
            dvi[i] = dv[i]

            L_new = self.Lagrangian_constrainedoptimization(dvi)

            grad_app[i] = (L_new-L) / dvi[i]

            # print(L_new, L, i + 1, "out of ", dv.shape[0])

        print(np.sum(grad*grad_app)/np.linalg.norm(grad)/np.linalg.norm(grad_app))
        print(np.linalg.norm(grad_app-grad))

        return grad, grad_app

    def check_hess_constrainedoptimization(self, dv=None):
        if self.three_overlap is None:
            self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap(self.molecule.wfn.basisset(),
                                                                             self.molecule.wfn.basisset(),
                                                                             self.vp_basis.wfn.basisset()))
            if self.ortho_basis:
                
                self.three_overlap = np.einsum("ijk,kl->ijl", self.three_overlap, self.vp_basis.A.np)

        Vks_a = psi4.core.Matrix.from_array(self.v0_input_Fock)
        Vks_b = psi4.core.Matrix.from_array(self.v0_input_Fock)
        self.molecule.scf_inversion(100, [Vks_a, Vks_b])

        hess = self.hess_constrainedoptimization()
        grad = self.grad_constrainedoptimization()[0]

        if dv is None:
            dv = 1e-7*np.ones_like(self.v_output)

        hess_app = np.zeros_like(hess)

        for i in range(grad.shape[0]):
            dvi = np.zeros_like(dv)
            dvi[i] = dv[i]

            grad_new = self.grad_constrainedoptimization(dvi)

            hess_app[i,:] = (grad_new - grad)/dv[i]

        hess_app = 0.5 * (hess_app + hess_app.T)
        print("SYMMETRY", np.linalg.norm(hess - hess.T))
        print("DIRECTION", np.trace(hess_app.dot(hess.T))/np.linalg.norm(hess_app)/np.linalg.norm(hess))
        print("SIMILARITY", np.linalg.norm(hess - hess_app))
        return hess, hess_app

    def find_vxc_scipy_constrainedoptimization(self, maxiter=1400, opt_method="BFGS"):

        if self.three_overlap is None:
            self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap(self.molecule.wfn.basisset(),
                                                                             self.molecule.wfn.basisset(),
                                                                             self.vp_basis.wfn.basisset()))
            if self.ortho_basis:
                
                self.three_overlap = np.einsum("ijk,kl->ijl", self.three_overlap, self.vp_basis.A.np)

        print("Zero the old result for a new calculation..")
        self.v_output = np.zeros_like(self.v_output)

        print("<<<<<<<<<<<<<<<<<<<<<<Constrained Optimization vxc Inversion<<<<<<<<<<<<<<<<<<<")

        dDa = self.input_wfn.Da().np - self.molecule.Da.np
        dDb = self.input_wfn.Db().np - self.molecule.Db.np
        dn = self.molecule.to_grid(dDa+dDb)
        print("|n| before", np.sum(np.abs(dn)*self.molecule.w))
        opt = {
            "disp": True,
            "maxiter": maxiter,
            # "eps": 1e-7
            # "norm": 2,
            "gtol": 1e-7
        }

        vp_array = optimizer.minimize(self.Lagrangian_constrainedoptimization,
                                      self.v_output,
                                      jac=self.grad_constrainedoptimization,
                                      method=opt_method,
                                      options=opt)

        nbf = int(vp_array.x.shape[0] / 2)
        v_output_a = vp_array.x[:nbf]
        v_output_b = vp_array.x[nbf:]

        Vks_a = psi4.core.Matrix.from_array(
            np.einsum("ijk,k->ij", self.three_overlap, v_output_a) + self.v0_input_Fock)
        Vks_b = psi4.core.Matrix.from_array(
            np.einsum("ijk,k->ij", self.three_overlap, v_output_b) + self.v0_input_Fock)

        self.molecule.scf_inversion(100, [Vks_a, Vks_b])
        dDa = self.input_wfn.Da().np - self.molecule.Da.np
        dDb = self.input_wfn.Db().np - self.molecule.Db.np
        dn = self.molecule.to_grid(dDa+dDb)
        print("|n| after", np.sum(np.abs(dn)*self.molecule.w), "L after", vp_array.fun)
        print("Ts", self.molecule.Da.vector_dot(self.molecule.T)+self.molecule.Db.vector_dot(self.molecule.T))
        print("dTs", np.trace(np.dot(self.input_wfn.Da().np+self.input_wfn.Db().np-
                                     self.molecule.Da.np-self.molecule.Db.np, self.molecule.T.np)))
        print("eigenA")
        print(self.input_wfn.epsilon_a().np[:self.molecule.nalpha])
        print(self.molecule.eig_a.np[:self.molecule.nalpha])
        print("wfnDiff", self.input_wfn.Ca().vector_dot(self.molecule.Ca)
              /np.linalg.norm(self.input_wfn.Ca().np)/np.linalg.norm(self.molecule.Ca.np))
        # Update info
        self.v_output = vp_array.x
        self.get_vxc()
        return vp_array