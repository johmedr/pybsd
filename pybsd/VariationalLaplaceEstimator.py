import jax.numpy as jnp
import jax.scipy as jsp
import jax
from functools import partial
from jax import vmap, grad, hessian
import numpy as np
from abc import ABC, abstractmethod
import warnings


class VariationalLaplaceEstimator(ABC):
	def __init__(self, pE, pC, hE, hC, Q=None, atol=1e-8):
		self._atol = atol
		self.set_priors(pE, pC)
		self._checkset_forward()
		self.set_hyperpriors(hE, hC, Q)


	# ------------------------
	# ---     FORWARD      ---
	# ------------------------
	@abstractmethod
	def forward(self, u):
		pass

	def _checkset_forward(self): 
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
	def _get_proj(self, C, inv=False):
		U,s,_ = jnp.linalg.svd(C, hermitian=True)

		if np.any(s < 0):
			warnings.warn('Prior covariance is not positive semi-definite, removing negative subspaces.')

		if inv: 
			s = 1./s 
		
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
			pC = np.diag(pC)

		if pE.shape[0] != pC.shape[0]: 
			raise ValueError(f'Prior expectration has shape {pE.shape},'
				f' while prior covariance has shape {pC.shape}!')
		
		# Get SVD of prior covariance
		U, s = self._get_proj(pC)	

		# Project priors
		self._pE = pE @ U
		self._pC = s

		# Store log det computation
		self._logdetpC = jnp.log(s).sum()

		# Store projection matrices
		self._Up  = U

		# Set number of parameters
		self._np = self._pE.shape[0]

	def get_priors(self):
		return self._project(self._Up.T, self._pE, self._pC)

	# --- HYPERPRIORS
	def set_hyperpriors(self, hE, hC, Q=None): 
		hE = np.reshape(hE, (-1,))

		if len(hC.shape) == 1 or hC.shape[0] != hC.shape[1]:
			hC = np.reshape(hC, (-1,))
			hC = np.diag(hC)

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

		
		# Get SVD of hyperprior covariance
		U, s  = self._get_proj(hC)	

		# Project hyperpriors
		self._hE = hE @ U 
		self._hC = s 		
		
		# Store log det computation
		self._logdethC = jnp.log(s).sum()

		# Store projection matrix
		self._Uh  = U

		# Number of hyperpriors
		self._nh = self._hE.shape[-1]

		# Project h-dim of precision components
		Q  = jnp.tensordot(U, Q, axes=(1,0))	
		
		# Save projected precision components
		self._Q  = Q

		# Compute full precision
		S = self.get_likelihood_precision(self._hE)
		
		# Project precision components
		U, s = self._get_proj(S, inv=True)
		Q = self._project(U, C=Q)
		
		# Save projected precision components
		self._Q  = Q

		# Save projection matrices
		self._Uy = U


	def get_hyperpriors(self, Q, hE, hC): 
		Q = self._project(self._G.T, C=self._Q)

		return Q, self._hE, self._hC

	def get_likelihood_precision(self, h):
		return jnp.vecdot(jnp.exp(h), self._Q, axis=0)


	@partial(jax.jit, static_argnums=0)
	def free_energy(self, y, qE, qC, gE, gC):
		try:
			ep = qE - self._pE
		except: 
			qE, qC = self._project(self._Up, qE, qC)
			ep = qE - self._pE

		try: 
			eh = gE - self._hE
		except: 
			gE, gC = self._project(self._Uh, gE, gC)
			eh = gE - self._hE

		ey =  (y - self.forward(qE)) @ self._Uy
		Sy = self.get_likelihood_precision(gE)
		
		logdetSy = jnp.linalg.slogdet(Sy)[1]
		logdetqC = jnp.linalg.slogdet(qC)[1]
		logdetgC = jnp.linalg.slogdet(gC)[1]

		# Likelihood 
		# ----------
		Fy = -0.5 * (logdetSy + ey @ Sy @ ey.T)

		# Parameters 
		# ----------
		Fp = -0.5 * (-self._logdetpC + (self._pC * ep * ep).sum())

		# Hyperparameters 
		# ---------------
		Fh = -0.5 * (-self._logdethC + (self._hC * eh * eh).sum())

		# Entropy  
		# -------
		Fe = 0.5 * (logdetqC + logdetgC)

		# Dimension 
		# ---------
		Fk = -0.5*(self._np + self._nh)*np.log(2*np.pi)

		# Total
		# =====
		F = Fy + Fp + Fh + Fe + Fk
		return F.squeeze()



	# @partial(jax.jit, static_argnums=0)	
	def fit(self, X, y, qE=None, gE=None, Nmax=64, Hmax=16): 
		if qE is None:
			qE = self._pE
		elif qE.shape != self._pE.shape: 
			qE = self._project(self._Up, qE)

		if gE is None: 
			gE = self._hE 
		elif gE.shape != self._hE.shape: 
			gE = self._project(self._Uh, gE)

		dFdp  = grad(self.free_energy, argnums=1)
		dFdpp = hessian(self.free_energy, argnums=1)

		dFdh  = grad(self.free_energy, argnums=3)
		dFdhh = hessian(self.free_energy, argnums=3)

		C = dict(F=None, qE=None, qC=None, gE=None, gC=None)
		qC = jnp.diag(self._pC)
		gC = jnp.diag(self._hC)
		criterion = [False]*5
		F0 = -1e9
		for i in range(Nmax):
			for j in range(Hmax):
				gC = - jnp.linalg.inv(dFdhh(y, qE, qC, gE, gC))
				qC = - jnp.linalg.inv(dFdpp(y, qE, qC, gE, gC))

				gE = gE + gC @ dFdh(y, qE, qC, gE, gC)
			qE = qE + qC @ dFdp(y, qE, qC, gE, gC)
			F = self.free_energy(y, qE, qC, gE, gC)
			print(F, qE)
			if i < 3 or F > C['F']: 
				C['F'] = F
				C['qE'] = qE
				C['qC'] = qC
				C['gE'] = gE
				C['gC'] = gC
			else:
				F = C['F']
				qE = C['qE']
				qC = C['qC']
				gE = C['gE']
				gC = C['gC']

			criterion.pop(0)
			criterion.append(F-F0 < 1e-2)
			if all(criterion):
				break
			F0 = F

		return C
