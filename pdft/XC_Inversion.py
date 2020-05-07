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

    def scf(self, maxiter, V=None, print_energies=False, vp_matrix=None):
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
    def __init__(self, molecule, input_wfn, vp_basis=None, vp_T=None):
        super().__init__([], molecule, vp_basis=vp_basis, vp_T=vp_T)

        self.input_wfn = input_wfn

        # Get reference vxc
        self.input_vxc_a = None
        self.input_vxc_b = None
        self.input_vxc_cube = None # To be implemented.
        self.get_input_vxc()

        # v_output = [v_output_a, v_output_b]
        self.v_output = np.zeros(int(self.vp_basis.nbf())*2)
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
        self.v_FermiAmaldi_input_Fock = None
        self.get_vH_vext()

        # vH_mol
        self.vH_mol = None

    def get_input_vxc(self):
        self.input_vxc_a, self.input_vxc_b = pdft.U_xc(self.input_wfn.Da().np, self.input_wfn.Db().np,
                                                       self.input_wfn.V_potential())[-1]

    def get_input_esp(self):
        assert self.esp_input is None

        nthreads = psi4.get_num_threads()
        psi4.set_num_threads(1)

        x, y, z, _ = self.molecule.Vpot.get_np_xyzw()
        grid = np.array([x, y, z])
        grid = psi4.core.Matrix.from_array(grid.T)
        assert grid.shape[1] == 3, "Grid should be N*3 np.array"

        esp_calculator = psi4.core.ESPPropCalc(self.input_wfn)
        self.esp_input = - esp_calculator.compute_esp_over_grid_in_memory(grid).np

        psi4.set_num_threads(nthreads)
        return

    def get_mol_vH(self):
        self.molecule.update_wfn_info()
        nthreads = psi4.get_num_threads()
        psi4.set_num_threads(1)

        x, y, z, _ = self.molecule.Vpot.get_np_xyzw()
        grid = np.array([x, y, z])
        grid = psi4.core.Matrix.from_array(grid.T)
        assert grid.shape[1] == 3, "Grid should be N*3 np.array"

        esp_calculator = psi4.core.ESPPropCalc(self.molecule.wfn)
        self.vH_mol = - esp_calculator.compute_esp_over_grid_in_memory(grid).np - self.vext

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
        self.input_wfn.nalpha() + self.input_wfn.nbeta()
        nocc = self.molecule.ndocc
        self.v_FermiAmaldi_input_Fock = self.molecule.grid_to_fock((nocc-1)/nocc*self.vH_input)
        return

    def get_vxc(self):
        """
        WuYang (25)
        """
        # Get vH_mol
        self.get_mol_vH()

        nocc = self.molecule.ndocc
        if self.ortho_basis:
            self.vxc_a_grid = self.molecule.to_grid(np.dot(self.molecule.A.np, self.v_output_a))
            self.vxc_a_grid += (nocc-1)/nocc*self.vH_input - self.vH_mol

            self.vxc_b_grid = self.molecule.to_grid(np.dot(self.molecule.A.np, self.v_output_b))
            self.vxc_b_grid += (nocc-1)/nocc*self.vH_input - self.vH_mol

        else:
            self.vxc_a_grid = self.molecule.to_grid(self.v_output_a)
            self.vxc_a_grid += (nocc-1)/nocc*self.vH_input - self.vH_mol

            self.vxc_b_grid = self.molecule.to_grid(self.v_output_b)
            self.vxc_b_grid += (nocc-1)/nocc*self.vH_input - self.vH_mol
        return


    def Laplacian_WuYang(self, v=None):
        """
        L = - <T> - \int (vks_a*(n_a-n_a_input)+vks_b*(n_b-n_b_input))
        :return: L
        """
        if self.three_overlap is None:
            if self.vp_basis is self.molecule.wfn.basisset():
                self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap())
            else:
                assert not self.ortho_basis
                self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap(self.molecule.wfn.basisset(),
                                                      self.molecule.wfn.basisset(), self.vp_basis))
            if self.ortho_basis:
                assert self.vp_basis is self.molecule.wfn.basisset()
                self.three_overlap = np.einsum("ijk,kl->ijl", self.three_overlap, self.molecule.A.np)

        if v is not None:
            nbf = int(v.shape[0]/2)
            v_output_a = v[:nbf]
            v_output_b = v[nbf:]

            Vks_a = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_a) + self.v_FermiAmaldi_input_Fock)
            Vks_b = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_b) + self.v_FermiAmaldi_input_Fock)
            self.molecule.scf(1000, [Vks_a, Vks_b])

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
        if v is not None:
            nbf = int(v.shape[0] / 2)
            v_output_a = v[:nbf]
            v_output_b = v[nbf:]

            Vks_a = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_a) + self.v_FermiAmaldi_input_Fock)
            Vks_b = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_b) + self.v_FermiAmaldi_input_Fock)

            self.molecule.scf(100, [Vks_a, Vks_b])

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
        if v is not None:
            nbf = int(v.shape[0] / 2)
            v_output_a = v[:nbf]
            v_output_b = v[nbf:]

            Vks_a = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_a) + self.v_FermiAmaldi_input_Fock)
            Vks_b = psi4.core.Matrix.from_array(np.einsum("ijk,k->ij", self.three_overlap, v_output_b) + self.v_FermiAmaldi_input_Fock)

            self.molecule.scf(100, [Vks_a, Vks_b])

        epsilon_occ_a = self.molecule.eig_a.np[:self.molecule.nalpha, None]
        epsilon_occ_b = self.molecule.eig_b.np[:self.molecule.nbeta, None]
        epsilon_unocc_a = self.molecule.eig_a.np[self.molecule.nalpha:]
        epsilon_unocc_b = self.molecule.eig_b.np[self.molecule.nbeta:]
        epsilon_a = epsilon_occ_a - epsilon_unocc_a
        epsilon_b = epsilon_occ_b - epsilon_unocc_b

        hess = np.zeros((self.molecule.nbf*2, self.molecule.nbf*2))
        # Alpha electrons
        hess[0:self.molecule.nbf, 0:self.molecule.nbf] = - 2 * self.molecule.omega * np.einsum('ai,bj,ci,dj,ij,abm,cdn -> mn',
                                                                                             self.molecule.Ca.np[:, :self.molecule.nalpha],
                                                                                             self.molecule.Ca.np[:, self.molecule.nalpha:],
                                                                                             self.molecule.Ca.np[:, :self.molecule.nalpha],
                                                                                             self.molecule.Ca.np[:, self.molecule.nalpha:],
                                                                                             np.reciprocal(epsilon_a), self.three_overlap,
                                                                                             self.three_overlap, optimize=True)
        # Beta electrons
        hess[self.molecule.nbf:, self.molecule.nbf:] = - 2 * self.molecule.omega * np.einsum('ai,bj,ci,dj,ij,abm,cdn -> mn',
                                                                                           self.molecule.Cb.np[:, :self.molecule.nbeta],
                                                                                           self.molecule.Cb.np[:, self.molecule.nbeta:],
                                                                                           self.molecule.Cb.np[:, :self.molecule.nbeta],
                                                                                           self.molecule.Cb.np[:, self.molecule.nbeta:],
                                                                                           np.reciprocal(epsilon_b),self.three_overlap,
                                                                                           self.three_overlap, optimize=True)
        return hess

    def check_gradient(self, dv=None):
        if self.three_overlap is None:
            if self.vp_basis is self.molecule.wfn.basisset():
                self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap())
            else:
                assert not self.ortho_basis
                self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap(self.molecule.wfn.basisset(),
                                                self.molecule.wfn.basisset(), self.vp_basis))
            if self.ortho_basis:
                assert self.vp_basis is self.molecule.wfn.basisset()
                self.three_overlap = np.einsum("ijk,kl->ijl", self.three_overlap, self.molecule.A.np)

        Vks_a = psi4.core.Matrix.from_array(self.v_FermiAmaldi_input_Fock)
        Vks_b = psi4.core.Matrix.from_array(self.v_FermiAmaldi_input_Fock)
        self.molecule.scf(100, [Vks_a, Vks_b])

        L = self.Laplacian_WuYang()
        grad = self.grad_WuYang()

        if dv is None:
            dv = 1e-7*np.ones_like(self.v_output)

        grad_app = np.zeros_like(dv)

        for i in range(dv.shape[0]):
            dvi = np.zeros_like(dv)
            dvi[i] = dv[i]

            L_new = self.Laplacian_WuYang(dvi)

            grad_app[i] = (L_new-L) / dvi[i]

            # print(L_new, L, i + 1, "out of ", dv.shape[0])

        print(np.sum(grad*grad_app)/np.linalg.norm(grad)/np.linalg.norm(grad_app))
        print(np.linalg.norm(grad_app-grad))

        return grad, grad_app

    def check_hess(self, dv=None):
        if self.three_overlap is None:
            if self.vp_basis is self.molecule.wfn.basisset():
                self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap())
            else:
                assert not self.ortho_basis
                self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap(self.molecule.wfn.basisset(),
                                                self.molecule.wfn.basisset(), self.vp_basis))
            if self.ortho_basis:
                assert self.vp_basis is self.molecule.wfn.basisset()
                self.three_overlap = np.einsum("ijk,kl->ijl", self.three_overlap, self.molecule.A.np)

        Vks_a = psi4.core.Matrix.from_array(self.v_FermiAmaldi_input_Fock)
        Vks_b = psi4.core.Matrix.from_array(self.v_FermiAmaldi_input_Fock)
        self.molecule.scf(100, [Vks_a, Vks_b])

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

            print(i+1, "out of ", grad.shape[0])

        hess_app = 0.5 * (hess_app + hess_app.T)
        print(np.trace(hess_app.dot(hess.T))/np.linalg.norm(hess_app)/np.linalg.norm(hess))
        print(np.linalg.norm(hess - hess_app))

        return hess, hess_app

    def find_vxc_scipy(self, maxiter=1400, opt_method="BFGS"):

        if self.three_overlap is None:
            if self.vp_basis is self.molecule.wfn.basisset():
                self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap())
            else:
                assert not self.ortho_basis
                self.three_overlap = np.squeeze(self.molecule.mints.ao_3coverlap(self.molecule.wfn.basisset(),
                                                self.molecule.wfn.basisset(), self.vp_basis))
            if self.ortho_basis:
                assert self.vp_basis is self.molecule.wfn.basisset()
                self.three_overlap = np.einsum("ijk,kl->ijl", self.three_overlap, self.molecule.A.np)

        print("<<<<<<<<<<<<<<<<<<<<<<WuYang vxc Inversion<<<<<<<<<<<<<<<<<<<")

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

        vp_array = optimizer.minimize(self.Laplacian_WuYang, self.v_output,
                                      jac=self.grad_WuYang, method=opt_method,
                                      options=opt)

        dDa = self.input_wfn.Da().np - self.molecule.Da.np
        dDb = self.input_wfn.Db().np - self.molecule.Db.np
        dn = self.molecule.to_grid(dDa+dDb)
        print("|n| after", np.sum(np.abs(dn)*self.molecule.w))

        # Update info
        self.v_output = vp_array.x
        nbf = int(self.v_output.shape[0] / 2)
        self.v_output_a = self.v_output[:nbf]
        self.v_output_b = self.v_output[nbf:]
        self.get_vxc()
        return vp_array