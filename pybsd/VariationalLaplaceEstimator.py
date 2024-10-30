import jax.numpy as jnp
import jax.scipy as jsp
import jax
from jax import lax
from functools import partial
from jax import jacfwd
import numpy as np
from abc import ABC, abstractmethod
import warnings
import time


class VariationalLaplaceEstimator(ABC):
    def __init__(self, pE, pC, hE, hC, Q=None, atol=1e-3):
        self._atol = atol
        self.set_priors(pE, pC)
        self._check_set_forward()
        self.set_hyperpriors(hE, hC, Q)


    # ------------------------
    # ---     FORWARD      ---
    # ------------------------
    @abstractmethod
    def forward(self, u):
        pass

    def _check_set_forward(self):
        self._y0 = self.forward(self._pE)

        if len(self._y0.shape) > 2:
            raise ValueError('Wrong output dimension for forward function.')

        self._ny = self._y0.shape[-1]

        if len(self._y0.shape) == 1:
            self._forward = self.forward
            self.forward = lambda x: self._forward(x).reshape((1, self._ny))
            self._nb = 1
        else:
            self._nb = self._y0.shape[0]

    # ------------------------
    # --- PROJECTION UTILS ---
    # ------------------------
    def _get_proj(self, C):
        U, s, _ = jnp.linalg.svd(C, hermitian=True)

        if np.any(s < 0):
            warnings.warn('Prior covariance is not positive semi-definite, removing negative subspaces.')
        
        ii = np.abs(s) > self._atol
        U  = U[:, ii]
        s  = s[ii]

        return U, s

    def _project(self, U, E=None, C=None):
        if E is not None: 
            E = E @ U
            if C is None: 
                return E

        if C is not None: 
            C = U.T @ C @ U
            if E is None: 
                return C

        return E, C

    # --- PRIORS
    def set_priors(self, pE, pC):
        pE = np.reshape(pE, (-1,))

        if len(pC.shape) == 1 or pC.shape[0] != pC.shape[1]:
            pC = np.reshape(pC, (-1,))

        if pE.shape[0] != pC.shape[0]: 
            raise ValueError(f'Prior expectration has shape {pE.shape},'
                f' while prior covariance has shape {pC.shape}!')
        
        self._pE = pE

        # Get SVD of prior covariance 
        # U, s = self._get_proj(pC)   

        # Project priors cov.
        # self._pC  = s
        # self._ipC = jnp.diag(1./s)

        self._pC  = pC
        self._ipC = jnp.diag(1./pC)

        # Store log det computation
        self._logdetpC = jnp.log(pC).sum()

        # Store projection matrices
        # self._Up  = U

        # Set number of parameters
        self._np = self._pE.shape[0]

    def get_priors(self):
        return self._project(self._Up.T, self._pE, self._pC)

    # --- HYPERPRIORS
    def set_hyperpriors(self, hE, hC, Q=None): 
        hE = np.reshape(hE, (-1,))

        if len(hC.shape) == 1 or hC.shape[0] != hC.shape[1]:
            hC = np.reshape(hC, (-1,))

        if hE.shape[0] != hC.shape[0]: 
            raise ValueError(f'Hyperprior expectration has shape {hE.shape},'
                f' while hyperprior covariance has shape {hC.shape}!')

        if Q is None: 
            Q = jnp.eye(self._ny)

        if len(Q.shape) == 2: 
            Q = Q.reshape((1, *Q.shape))
        
        if len(Q.shape) == 3:
            if Q.shape[0] == 1:
                Q = jnp.repeat(Q, hE.shape[0], axis=0)
            elif Q.shape[0] != hE.shape[0]:
                raise ValueError(f'Shape mismatch between Q array and hE.')
        else: 
            raise ValueError(f'Wrong shape for Q array: {Q.shape}.') 
        
        self._hE = hE

        # Get SVD of hyperprior covariance
        # U, s  = self._get_proj(hC)  

        # # Project hyperpriors
        # self._hC = s        
        # self._ihC = jnp.diag(1./s)

        self._hC  = hC
        self._ihC = jnp.diag(1./hC)

        # Store log det computation
        self._logdethC = jnp.log(hC).sum()

        # Store projection matrix
        # self._Uh  = U

        # Number of hyperpriors
        self._nh = self._hE.shape[-1]

        # Project h-dim of precision components
        # Q  = jnp.tensordot(U, Q, axes=(1,0))    
        
        # Save projected precision components
        self._Q  = Q

        # Compute full precision
        S = self.get_likelihood_precision(self._hE)
        
        # Project precision components
        U, _ = self._get_proj(S)
        Q = self._project(U, C=Q)
        
        # Save projected precision components
        # Shape (nh, ny, ny) 
        self._Q  = Q

        # Save projection matrix
        self._Uy = U

    def get_hyperpriors(self, hE, hC): 
        return  self._project(self._Uh.T, self._hE, self._hC)

    def get_likelihood_precision(self, h):
        return jnp.vecdot(jnp.exp(h) + jnp.exp(-32), self._Q, axis=0)


    @partial(jax.jit, static_argnums=0)
    def free_energy(self, y, qE, qC, gE, gC):
        try:
            qE - self._pE
        except: 
            qE, qC = self._project(self._Up, qE, qC)

        try: 
            gE - self._hE
        except: 
            gE, gC = self._project(self._Uh, gE, gC)

        
        logdetqC = jnp.linalg.slogdet(qC)[1]
        logdetgC = jnp.linalg.slogdet(gC)[1]

        # Log joint  
        # ---------
        Fl = self._logjoint(y, qE, qC, gE, gC)

        # Entropy  
        # -------
        Fe = 0.5 * (logdetqC + logdetgC)

        # Dimension 
        # ---------
        Fk = -0.5*(self._np + self._nh)*np.log(2*np.pi)

        # Total
        # =====
        F = Fl + Fe + Fk
        return F.squeeze()

    @partial(jax.jit, static_argnums=0)
    def _logjoint(self, y, qE, qC, gE, gC):
        ep = qE - self._pE
        eh = gE - self._hE
        ey =  (y - self.forward(qE @ self._Up.T)) @ self._Uy

        Sy = self.get_likelihood_precision(gE)
        
        logdetSy = jnp.linalg.slogdet(Sy)[1]

        # Likelihood 
        # ----------
        Fy = -0.5 * (logdetSy + jnp.vecdot(ey @ Sy, ey, axis=-1).sum(-1))

        # Parameters 
        # ----------
        Fp = -0.5 * (-self._logdetpC + (self._pC * ep * ep).sum())

        # Hyperparameters 
        # ---------------
        Fh = -0.5 * (-self._logdethC + (self._hC * eh * eh).sum())

        # Total
        # =====
        F = Fy + Fp + Fh
        return F.squeeze()

    def fit(self, y, qE=None, gE=None, Emax=64, Mmax=8): 
        # Handle q shape
        if qE is None:
            ep = jnp.zeros(self._np)
        else:
            try:
                ep = (qE - self._pE) 
            except: 
                raise ValueError('Wrong shape for qE.')

        # Handle g shape
        if gE is None: 
            eh = jnp.zeros(self._nh) 
        else:
            try:
                eh = (gE - self._hE) 
            except: 
                raise ValueError('Wrong shape for gE.')

        # Check y shape
        # ny = y.shape[0] 
        # nb = 1

        if len(y.shape) == 1:
            ny = y.shape[0] # size of dependent space (features) 
            ns = 1          # size of independent space (samples) 
        elif len(y.shape) == 2:
            ny = y.shape[1]
            ns = y.shape[0]
        # if len(y.shape) > 2: 
        #     nb = y.shape[2]

        y = y.reshape((ns, ny)) @ self._Uy

        # Priors in projected space 
        pE = self._pE 
        hE = self._hE 

        # Jacobian function for forward function 
        forward = lambda ep: self.forward(pE + ep).reshape((ny,)) @ self._Uy
        try:
            forward = jax.jit(forward)
        except: 
            pass

        Jf = jacfwd(forward, argnums=0)
        try:
            Jf = jax.jit(Jf)
        except: 
            pass


        # Prior covariances
        pC  = self._pC
        hC  = self._hC
        
        # Prior precisions
        ipC = self._ipC
        ihC = self._ihC

        # Initial posterior covariances
        qC  = jnp.diag(pC)
        gC  = jnp.diag(hC)

        # Loop variables
        criterion = [False]*5
        C = dict()
        v = -4

        # Correction terms
        Iy = jnp.eye(ny)*1e-16
        Ip = jnp.eye(self._np)*1e-16
        Ih = jnp.eye(self._nh)*1e-16

        # Decomposed free-energy
        L = np.zeros(6)

        # E-step (with nested M-step)
        # ===========================
        for i in range(Emax):   
            tstart = time.perf_counter_ns()

            # Compute error and jacobian
            ey   = y - forward(ep)
            dfdp = Jf(ep)

            # Check norm and nans 
            normdfdp = jnp.abs(dfdp).sum(-1).max() 
            revert   = jnp.isnan(normdfdp) or normdfdp > 1e32;
            
            if revert and i > 1:
                for j in range(8): 
                    # Increase regularisation 
                    v = min(v - 2,-32)

                    # Update parameters
                    ep = C['ep'] + self._compute_dx(dFdp, dFdpp, v)

                    # Compute error and jacobian
                    ey   = y - forward(ep)
                    dfdp = Jf(ep)

                    # Check norm and nans 
                    normdfdp = jnp.abs(dfdp).sum(-1).max() 
                    revert   = jnp.isnan(normdfdp) or normdfdp > 1e32;

                    if not revert: 
                        break

            # Exit on failure to converge
            if revert: 
                raise RuntimeError('Convergence error.')


            # M-step
            # ======
            for j in range(Mmax):

                # Compute Pi matrices
                li = (jnp.exp(hE + eh) + jnp.exp(-32)).reshape((-1,1,1))
                Pi = li * self._Q

                # Compute data precision 
                Sy  = Pi.sum(0)
                iSy = jnp.linalg.inv(Sy + Iy)

                # Compute posterior covariance over parameters
                Hp = - dfdp.mT @ Sy @ dfdp - ipC
                Cp = jnp.linalg.inv(- Hp + Ip)

                # Precompute some terms
                PS     = Pi @ iSy
                trPS   = 0.5 * jnp.trace(PS, axis1=1, axis2=2)
                trPSPS = 0.5 * jnp.trace(PS @ PS, axis1=1, axis2=2)
                ePe    = 0.5 * jnp.vecdot(ey @ Pi, ey, axis=-1).sum(-1)
                trCJPJ = 0.5 * jnp.trace(Cp @ dfdp.mT @ Pi @ dfdp, axis1=1, axis2=2)

                # Compute gradient wrt h
                dFdh = trPS - ePe - ihC @ eh - trCJPJ

                # Compute hessian wrt h
                dFdhh = jnp.diag(- jnp.diag(ihC) + trPS - trPSPS - ePe - trCJPJ)

                # Compute update
                dh = self._compute_dx(dFdh, dFdhh, 4)
                dh = jnp.clip(dh, -1, 1)
                eh += dh

                # Check convergence
                dF    = jnp.vecdot(dFdh, dh) 
                if dF <  self._atol: 
                    break
                
            # Posterior covariance of h
            Ch = jnp.linalg.inv(-dFdhh + Ih)

            # Prediction loss 
            eSe = ePe.sum(0) 

            L[0] = - 0.5 * jnp.linalg.slogdet(iSy + Iy)[1] - eSe
            L[1] = - 0.5 * (self._logdetpC + ep @ ipC @ ep.T)
            L[2] = - 0.5 * (self._logdethC + eh @ ihC @ eh.T)
            L[3] = + 0.5 * jnp.linalg.slogdet(Cp + Ip)[1]
            L[4] = + 0.5 * jnp.linalg.slogdet(Ch + Ih)[1]
            L[5] = - 0.5 * (self._nh + self._np) * jnp.log(2*jnp.pi)

            F = L.sum().squeeze()

            if i > 0:
                print(f' (actual): {F-C["F"]:.2e} ({format_time(time.perf_counter_ns() - tstart)})')


            if jnp.isnan(F):
                raise RuntimeError('Free energy is nan - check problem conditioning.')

            if i < 1 or F > C['F']:     
                C['F']  = F
                C['L']  = L
                C['ep'] = ep
                C['Cp'] = Cp
                C['eh'] = eh
                C['Ch'] = Ch

                dFdp = (dfdp.mT @ iSy @ ey.mT).sum(-1) - ipC @ ep
                dFdpp = Hp

                v = min(v + 1, 8)

                print(f'EM: (+) ', end='')
            else:
                ep = C['ep']
                Cp = C['Cp']
                eh = C['eh']
                Ch = C['Ch']
                v = max(v - 2, -16)

                print(f'EM: (-) ', end='')

            dp = self._compute_dx(dFdp, dFdpp, v)
            ep += dp 

            if i == 0:
                F0 = F

            dF = dFdp.T @ dp
            print(f'{i}: F-F0: {C["F"]-F0:.2e} dF (predicted): {dF:.2e}', end='')

            criterion.pop(0)
            criterion.append(dF < self._atol)

            if all(criterion):
                print(' convergence.')
                break

        C['Ep'] = self._pE + C['ep']
        C['Eh'] = self._hE + C['eh']

        return C


    @partial(jax.jit, static_argnums=0) 
    def _compute_dx(self, f, dfdx, t, ): 
        if len(f.shape) == 1:
            f = jnp.expand_dims(f, -1)
        if len(dfdx.shape) == 1:
            dfdx = jnp.expand_dims(dfdx, -1)
            
        # if isreg we use t as a regularization parameter   
        t  = jnp.exp(t - jnp.linalg.slogdet(dfdx)[1] / f.shape[0]).squeeze()

        if f.shape[0] != dfdx.shape[0]: 
            raise ValueError(f'Shape mismatch: first dim of f {f.shape} must match that of df/dx {dfdx.shape}.')

        if len(f) == len(dfdx) == 0:
            return jnp.array([[]])

        return lax.cond(t > np.exp(16), 
            lambda f, dfdx: -jnp.linalg.pinv(dfdx) @ f.squeeze(-1), 
            lambda f, dfdx: 
                # use the exponentiation trick to avoid inverting dfdx
                jsp.linalg.expm(jnp.block([
                    [jnp.zeros((1,1)), jnp.zeros((1, dfdx.shape[-1]))], 
                    [          f * t, dfdx * t]
                ]))[1:, 0], 
            f, dfdx)


    # def fit_grad(self, X, y, qE=None, gE=None, Nmax=64, Hmax=16): 
    #     if qE is None:
    #         qE = self._pE
    #     elif qE.shape != self._pE.shape: 
    #         qE = self._project(self._Up, qE)  

    #     if gE is None: 
    #         gE = self._hE 
    #     elif gE.shape != self._hE.shape: 
    #         gE = self._project(self._Uh, gE)

    #     meanfield_p = lambda y, qE, qC, gE, gC: self._logjoint(y, qE, qC, gE, gC) + 0.5*jnp.trace(gC * )
    #     dFdp  = jax.jit(grad(self._logjoint, argnums=1))
    #     dFdpp = jax.jit(hessian(self._logjoint, argnums=1))

    #     dFdh  = jax.jit(grad(self._logjoint, argnums=3))
    #     dFdhh = jax.jit(hessian(self._logjoint, argnums=3))

    #     C = dict(F=None, qE=None, qC=None, gE=None, gC=None)
    #     qC = jnp.diag(self._pC)
    #     gC = jnp.diag(self._hC)
    #     criterion = [False]*5
    #     v = 4
    #     for i in range(Nmax):   
    #         tstart = time.perf_counter_ns()
    #         for j in range(Hmax):
    #             Hg = dFdhh(y, qE, qC, gE, gC)
    #             Jg =  dFdh(y, qE, qC, gE, gC)

    #             dg = self._compute_dx(Jg, Hg, 4)

    #             gE = gE + dg.reshape(gE.shape)

    #             gC = - jnp.linalg.pinv(Hg + jnp.eye(self._nh) * 1e-32)
    #             if Jg.T @ dg  < 1e-2:
    #                 break

    #         F = self.free_energy(y, qE, qC, gE, gC)

    #         if i > 0:
    #             print(f' (actual): {F-C["F"]:.2e} ({format_time(time.perf_counter_ns() - tstart)})')


    #         if jnp.isnan(F):
    #             raise RuntimeError('Free energy is nan - check problem conditioning.')

    #         if i < 3 or F > C['F']: 
    #             C['F'] = F
    #             C['qE'] = qE
    #             C['qC'] = qC
    #             C['gE'] = gE
    #             C['gC'] = gC

    #             Jq = dFdp(y, qE, qC, gE, gC)
    #             Hq = dFdpp(y, qE, qC, gE, gC)

    #             v = min(v+1/2, 4)
    #         else:
    #             F = C['F']
    #             qE = C['qE']
    #             qC = C['qC']
    #             gE = C['gE']
    #             gC = C['gC']
    #             v = max(v-2, -4)

    #         dp = self._compute_dx(Jq, Hq, v)
    #         qE = qE + dp 

    #         if i == 0:
    #             F0 = F

    #         dF = Jq.T @ dp
    #         print(f'({i}) F-F0: {C["F"]-F0:.2e} dF (predicted): {dF:.2e}', end='')

    #         criterion.pop(0)
    #         criterion.append(dF < 1e-1)

    #         if all(criterion):
    #             print(' convergence.')
    #             break


    #     return C



def format_time(ns):
    if ns < 1e3:  # Less than microsecond
        return f"{ns:.2f} ns"
    elif ns < 1e6:  # Less than millisecond
        return f"{ns * 1e-3:.2f} µs"
    elif ns < 1e9:  # Less than second
        return f"{ns * 1e-6:.2f} ms"
    else:  
        return f"{ns * 1e-9:.2f} s"