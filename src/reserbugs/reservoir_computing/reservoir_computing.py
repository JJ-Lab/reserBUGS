import numpy as np
from sklearn.linear_model import PoissonRegressor, GammaRegressor, TweedieRegressor
from sklearn.preprocessing import MinMaxScaler
from reservoirpy.nodes import Reservoir 
import random
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

"""
Reservoir-based forecasting wrapper using ReservoirPy + scikit-learn GLMs.

This module exposes the ReservoirComputing class which:
- Builds a ReservoirPy reservoir with given hyperparameters.
- Transforms input features through the reservoir states.
- Fits a generalized linear model (Poisson, Gamma, or Tweedie) on
  the reservoir-transformed features.
- Provides deterministic prediction and utilities to generate
  stochastic sample paths by refitting with different random seeds
  and (optionally) jittered hyperparameters.

Notes
-----
- The class attaches a MinMaxScaler to the augmented input (features + lags).
- Predictions are clipped to respect the domain of the chosen family:
  Gamma: strictly > 0, Poisson/Tweedie: >= 0.
"""

class ReservoirComputing:
    """
    Reservoir computing model with a GLM output layer for forecasting.

    Parameters
    ----------
    reservoir_size : int
        Number of reservoir units/neurons.
    spectral_radius : float
        Spectral radius of the reservoir weight matrix (sr).
    input_scaling : float
        Scaling applied to input weights.
    leaking_rate : float
        Reservoir leaking rate (lr).
    activation : str
        Activation function used by ReservoirPy node (e.g. 'tanh').
    input_connectivity, rc_connectivity, fb_connectivity : float
        Connectivity hyperparameters for the reservoir.
    seed : int
        Random seed used for reservoir initialization.
    family : str
        One of 'gamma', 'poisson', or 'tweedie' — selects the GLM family.

    Attributes
    ----------
    scaler_X : sklearn.preprocessing.MinMaxScaler or None
        Scaler fitted to the augmented training inputs.
    reservoir : reservoirpy.nodes.Reservoir
        The underlying reservoir node.
    model : sklearn linear model or None
        Trained GLM after calling fit().

    Examples
    --------
    >>> rc = ReservoirComputing(reservoir_size=500, family="tweedie")
    >>> rc.fit(X_train, y_train)
    >>> preds = rc.predict(X_test, y_init)
    """
    def __init__(self,
                 reservoir_size=1000,
                 spectral_radius=0.9,
                 input_scaling=0.1,
                 leaking_rate=1.0,
                 activation='tanh',
                 input_connectivity=0.1,
                 rc_connectivity=0.1,
                 fb_connectivity=0.1,
                 seed=42,
                 family="tweedie"):
        # Reservoir params
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leaking_rate = leaking_rate
        self.activation = activation
        self.input_connectivity = input_connectivity
        self.rc_connectivity = rc_connectivity
        self.fb_connectivity = fb_connectivity
        self.seed = seed

        # Model family: "gamma", "poisson", or "tweedie"
        self.family = family.lower()

        # Scalers
        self.scaler_X = None
        self.scaler_Y = None

        # Reservoir
        self.reservoir = Reservoir(
            units=self.reservoir_size,
            sr=self.spectral_radius,
            input_scaling=self.input_scaling,
            lr=self.leaking_rate,
            activation=self.activation,
            input_connectivity=self.input_connectivity,
            rc_connectivity=self.rc_connectivity,
            fb_connectivity=self.fb_connectivity,
            seed=self.seed
        )

        # Placeholder for GLM
        self.model = None

    def _augment_with_lags(self, X, Y, n_lags=1):
        """
        Create features by concatenating X[t] with previous n_lags of Y.

        Parameters
        ----------
        X : array-like, shape (T, n_features)
            Exogenous features.
        Y : array-like, shape (T,)
            Target series.
        n_lags : int
            Number of past Y values to include as autoregressive features.

        Returns
        -------
        X_aug : np.ndarray, shape (T - n_lags, n_features + n_lags)
            Augmented feature matrix aligned with trimmed Y.
        Y_trimmed : np.ndarray, shape (T - n_lags,)
            Target vector aligned to X_aug (Y[n_lags:]).
        """

        X = np.asarray(X)
        Y = np.asarray(Y).ravel()
        rows = []
        for t in range(n_lags, len(Y)):
            row = np.hstack([X[t], Y[t - n_lags:t][::-1]])  # add [y_{t-1}, y_{t-2}, ...]
            rows.append(row)
        return np.array(rows), Y[n_lags:]

    def scale_and_expand_features(self, X_train, Y_train, X_test, Y_test=None, previous_scaler=False, n_lags=1):
        """
        Fit or reuse a MinMaxScaler on augmented training inputs and run them through the reservoir.

        Parameters
        ----------
        X_train, Y_train : training arrays
            Inputs and target used to create lag-augmented training matrix.
        X_test : array-like
            Raw test input matrix (not augmented inside this function).
        Y_test : optional
            Unused placeholder (kept for API symmetry).
        previous_scaler : bool
            If True and self.scaler_X exists, reuse it instead of refitting.
        n_lags : int
            Number of lags used when augmenting inputs.

        Returns
        -------
        X_train_res : np.ndarray
        X_train_scaled : np.ndarray
        X_test_aug : np.ndarray
        _ : None
        Y_train : np.ndarray
        _ : None
        
        Notes
        -----
        The return signature mirrors a legacy API and includes unused placeholders.
        """
        X_train_aug, Y_train = self._augment_with_lags(X_train, Y_train, n_lags=n_lags)
        X_test_aug = np.asarray(X_test)
    
        if previous_scaler and self.scaler_X is not None:
            X_train_scaled = self.scaler_X.transform(X_train_aug)
        else:
            self.scaler_X = MinMaxScaler()
            X_train_scaled = self.scaler_X.fit_transform(X_train_aug)
    
        X_train_res = self.reservoir.run(X_train_scaled, reset=True)
        return X_train_res, X_train_scaled, X_test_aug, None, Y_train, None

    def fit(self, X_train, Y_train, n_lags=1):
        """
        Fit the GLM on reservoir-transformed features.

        Steps:
        - Augment X_train with past Y lags and scale (MinMax).
        - Run the scaled augmented inputs through the reservoir to obtain features.
        - Validate target domain according to chosen family.
        - Fit the corresponding sklearn GLM.

        Returns
        -------
        self : ReservoirComputing
            Fitted estimator.
        """
        X_train_res, _, _, _, y_train, _ = self.scale_and_expand_features(
            X_train, Y_train, X_train, previous_scaler=False, n_lags=n_lags
        )

        # Domain checks for y
        y_arr = np.asarray(y_train)
        if self.family == "gamma":
            if np.any(y_arr <= 0):
                raise ValueError("Gamma requires y > 0.")
        elif self.family in ("poisson", "tweedie"):
            if np.any(y_arr < 0):
                raise ValueError(f"{self.family.title()} requires y >= 0.")
        
        if self.family == "gamma":
            self.model = GammaRegressor(alpha=1e-4, max_iter=10000, verbose=0)
        elif self.family == "poisson":
            self.model = PoissonRegressor(alpha=1e-4, max_iter=10000, verbose=0)
        elif self.family == "tweedie":
            self.model = TweedieRegressor(power=1.5, link="log", alpha=1e-4, max_iter=10000, verbose=0)
        else:
            # (4) Better error message
            raise ValueError("family must be 'gamma', 'poisson' or 'tweedie'")
    
        self.model.fit(X_train_res, np.ravel(y_train))
        return self

    def predict(self, X_test, Y_init, n_lags=1):
        """
        Recursive multi-step prediction using the trained model.

        Parameters
        ----------
        X_test : array-like, shape (T, n_features)
            Exogenous covariates for the forecasting horizon.
        Y_init : array-like
            Initial history of Y used to seed autoregressive lags.
            Must contain at least n_lags values.
        n_lags : int
            Number of lags included during training/prediction.

        Returns
        -------
        preds : np.ndarray, shape (T,)
            Predicted values for the forecast horizon.

        Notes
        -----
        - Predictions are generated sequentially: each predicted value is
          appended to the lag history and used for subsequent steps.
        - The scaler fitted on training augmented inputs is used to scale
          each new augmented row before passing it to the reservoir.

        Examples
        --------
        >>> preds = rc.predict(X_test, y_init)
        """
        preds, y_hist = [], list(np.asarray(Y_init)[-n_lags:])

        # Validate Y_init length
        if len(y_hist) < n_lags:
            raise ValueError(f"Y_init must have at least {n_lags} values; got {len(y_hist)}.")

        for t in range(len(X_test)):
            x_row = X_test[t] if isinstance(X_test, np.ndarray) else X_test.iloc[t].values
            row = np.hstack([x_row, y_hist[::-1]])
            row_scaled = self.scaler_X.transform(row.reshape(1, -1))
            res_features = self.reservoir.run(row_scaled, reset=False)
            y_pred = self.model.predict(res_features)[0]
            # respeta los dominios:
            if self.family == "gamma":
                y_pred = max(1e-9, y_pred)   # Gamma: estrictamente positivo
            else:
                y_pred = max(0.0, y_pred)    # Poisson/Tweedie: no negativo
            preds.append(y_pred)
            y_hist = (y_hist + [y_pred])[-n_lags:]
        return np.array(preds)


    def sample_paths(
        self,
        X_train, Y_train, X_test, Y_init,
        n_lags=1,
        N=100,
        base_seed=1234,
        # jitter ranges; set to None to disable
        sr_range=None,         # e.g., (0.85, 1.00)
        lr_range=None,         # e.g., (0.80, 0.95)
        is_range=None,         # e.g., (0.08, 0.12) for input_scaling
        return_stats=True,
        quantiles=(0.05, 0.5, 0.95),
    ):
        """
        Generate N stochastic sample paths by refitting independent RC instances.

        Each path uses a different seed (base_seed + i) and optionally
        random jitter for reservoir hyperparameters.

        Parameters
        ----------
        X_train, Y_train, X_test, Y_init : arrays
            Data used for fitting and forecasting (Y_init seeds the recursion).
        N : int
            Number of sample paths to generate.
        base_seed : int
            Base seed; path i uses base_seed + i.
        sr_range, lr_range, is_range : tuple or None
            If provided, used to uniformly sample jittered spectral radius,
            leaking rate and input_scaling per path.
        return_stats : bool
            If True, return (preds, stats); otherwise return preds only.
        quantiles : tuple
            Quantiles to compute across the ensemble for summary statistics.

        Returns
        -------
        preds : np.ndarray, shape (N, T)
            Simulated paths.
        stats : dict
            If return_stats is True, returns mean, quantiles and seeds.

        Notes
        -----
        - Each path is generated by fitting an independent model.
        - Randomness arises from:
          - reservoir initialization (seed)
          - optional hyperparameter jitter
        - Results are reproducible given the same base_seed.

        Examples
        --------
        >>> paths, stats = rc.sample_paths(X_train, y_train, X_test, y_init, N=100)
        """
        # --- basic validation ---
        y_hist = list(np.asarray(Y_init)[-n_lags:])
        if len(y_hist) < n_lags:
            raise ValueError(f"Y_init must have at least {n_lags} values; got {len(y_hist)}.")

        T = len(X_test)
        preds = np.empty((N, T), dtype=float)
        seeds = []

        # copy constructor params from self
        base_kwargs = dict(
            reservoir_size=self.reservoir_size,
            spectral_radius=self.spectral_radius,
            input_scaling=self.input_scaling,
            leaking_rate=self.leaking_rate,
            activation=self.activation,
            input_connectivity=self.input_connectivity,
            rc_connectivity=self.rc_connectivity,
            fb_connectivity=self.fb_connectivity,
            family=self.family,
        )

        for i in range(N):
            seed = base_seed + i
            seeds.append(seed)

            # jitter hyperparams if ranges are provided
            sr = random.uniform(*sr_range) if sr_range else base_kwargs["spectral_radius"]
            lr = random.uniform(*lr_range) if lr_range else base_kwargs["leaking_rate"]
            ins = random.uniform(*is_range) if is_range else base_kwargs["input_scaling"]

            rc_i = ReservoirComputing(
                **{**base_kwargs, "spectral_radius": sr, "leaking_rate": lr, "input_scaling": ins, "seed": seed}
            )

            # Fit and predict with this independent instance
            rc_i.fit(X_train, Y_train, n_lags=n_lags)
            preds[i, :] = rc_i.predict(X_test, Y_init, n_lags=n_lags)

        if not return_stats:
            return preds

        # summary stats
        mean = preds.mean(axis=0)
        qs = np.quantile(preds, q=quantiles, axis=0)
        stats = {
            "mean": mean,
            "quantiles": {float(q): qs[j] for j, q in enumerate(quantiles)},
            "seeds": seeds,
        }
        return preds, stats


    def sample_paths_parallel(
        self,
        X_train, Y_train, X_test, Y_init,
        n_lags=1,
        N=100,
        base_seed=1234,
        sr_range=None,              # e.g., (0.85, 1.00)
        lr_range=None,              # e.g., (0.80, 0.95)
        input_scaling_range=None,   # e.g., (0.08, 0.12)
        return_stats=True,
        quantiles=(0.05, 0.5, 0.95),
        n_jobs=1,                   # -1 for all CPUs
        max_nbytes="50M",           # memmap large args to avoid copying
    ):
        """
        Parallel version of sample_paths using joblib.

        Differences vs sample_paths:
        - Uses threadpool_limits to avoid OpenMP/BLAS oversubscription inside workers.
        - Uses numpy Generator for per-worker randomness.

        Parameters
        ----------
        n_jobs : int
            Number of parallel jobs for joblib. -1 uses all CPUs.
        max_nbytes : str
            Threshold for memmapping large arguments when using loky backend.

        Returns
        -------
        preds : np.ndarray, shape (N, T)
            Ensemble of simulated paths.
        stats : dict
            If return_stats is True, returns mean, quantiles and seeds.
        """
        # --- basic validation ---
        y_hist = list(np.asarray(Y_init)[-n_lags:])
        if len(y_hist) < n_lags:
            raise ValueError(f"Y_init must have at least {n_lags} values; got {len(y_hist)}.")
    
        T = len(X_test)
        seeds = [base_seed + i for i in range(N)]
    
        base_kwargs = dict(
            reservoir_size=self.reservoir_size,
            spectral_radius=self.spectral_radius,
            input_scaling=self.input_scaling,
            leaking_rate=self.leaking_rate,
            activation=self.activation,
            input_connectivity=self.input_connectivity,
            rc_connectivity=self.rc_connectivity,
            fb_connectivity=self.fb_connectivity,
            family=self.family,
        )
    
        def _one_path(i, seed):
            # Limit BLAS/OpenMP threads inside the worker to avoid oversubscription
            with threadpool_limits(limits=1):
                rng = np.random.default_rng(seed)
    
                sr  = rng.uniform(*sr_range)            if sr_range            else base_kwargs["spectral_radius"]
                lr  = rng.uniform(*lr_range)            if lr_range            else base_kwargs["leaking_rate"]
                ins = rng.uniform(*input_scaling_range) if input_scaling_range else base_kwargs["input_scaling"]
    
                rc_i = ReservoirComputing(
                    **{**base_kwargs,
                       "spectral_radius": sr,
                       "leaking_rate": lr,
                       "input_scaling": ins,
                       "seed": seed}
                )
                rc_i.fit(X_train, Y_train, n_lags=n_lags)
                y_pred = rc_i.predict(X_test, Y_init, n_lags=n_lags)
                return y_pred  # shape (T,)
    
        if n_jobs == 1:
            paths = [_one_path(i, s) for i, s in enumerate(seeds)]
        else:
            paths = Parallel(n_jobs=n_jobs, backend="loky", max_nbytes=max_nbytes)(
                delayed(_one_path)(i, s) for i, s in enumerate(seeds)
            )
    
        preds = np.vstack(paths)  # (N, T)
    
        if not return_stats:
            return preds
    
        mean = preds.mean(axis=0)
        qs = np.quantile(preds, q=quantiles, axis=0)
        stats = {
            "mean": mean,
            "quantiles": {float(q): qs[j] for j, q in enumerate(quantiles)},
            "seeds": seeds,
        }
        return preds, stats
