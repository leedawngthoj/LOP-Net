# DataBalance (clean, no leakage)
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bayes_opt import BayesianOptimization

class DataBalance:
    """
    Auto-select an oversampling strategy and sampling ratio for a given training set,
    WITHOUT causing data leakage.

    Usage:
        balancer = DataBalance(X_train, y_train, random_state=42)
        X_res, y_res = balancer.auto_select()
    """

    def __init__(self, X_train, y_train, threshold=0.7, random_state=99, scale_inside = True, verbose=True):
        # Accept pandas or numpy
        self.X_train = X_train.values if hasattr(X_train, "values") else np.asarray(X_train)
        self.y_train = y_train.values if hasattr(y_train, "values") else np.asarray(y_train)
        self.threshold = threshold
        self.random_state = random_state
        self.best_method = None
        self.best_score = -1.0
        self.best_strategy = None
        self.results = {}   # store (method -> (best_strategy, best_score))
        self.verbose = verbose
        self.scale_inside = scale_inside

    def imbalance_ratio(self):
        counts = np.bincount(self.y_train.astype(int))
        minority, majority = min(counts), max(counts)
        return minority / majority

    def candidate_methods(self):
        return {
            "Random": RandomOverSampler(random_state=self.random_state),
            "SMOTE": SMOTE(random_state=self.random_state),
            "ADASYN": ADASYN(random_state=self.random_state),
            "SMOTEENN": SMOTEENN(random_state=self.random_state)
        }

    def _train_small_mlp_and_eval(self, X_tr, y_tr, X_val, y_val, epochs=8, lr=5e-4):
        """
        Train a small MLP (PyTorch) on X_tr/y_tr and return f1 on X_val/y_val.
        Assumes X_tr and X_val are numpy arrays (already scaled).
        """
        device = torch.device("cpu")
        X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device).unsqueeze(1)
        y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device).unsqueeze(1)

        input_dim = X_tr.shape[1]
        torch.manual_seed(self.random_state)
        clf = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(device)

        optimizer = torch.optim.Adam(clf.parameters(), lr=lr)
        loss_fn = nn.BCELoss()

        clf.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            y_pred = clf(X_tr_t)
            loss = loss_fn(y_pred, y_tr_t)
            loss.backward()
            optimizer.step()

        clf.eval()
        with torch.no_grad():
            y_pred_val = clf(X_val_t).cpu().numpy()
            y_pred_label = (y_pred_val > 0.5).astype(int)
            f1 = f1_score(y_val_t.cpu().numpy(), y_pred_label)
        return float(f1)

    def evaluate_balance(self, X, y, method, sampling_strategy):
        """
        Correct evaluation WITHOUT leakage:
        1) Split (X,y) into inner train / val
        2) Fit scaler on inner-train (ONLY)
        3) Transform inner-train and val
        4) Apply sampler.fit_resample on scaled inner-train
        5) Train MLP on resampled-scaled inner-train, evaluate on scaled val (val untouched by resampling)
        Returns f1 (float).
        """
        # 1) split BEFORE any resampling
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )

        # 2) scale: fit on X_tr only
        if self.scale_inside:
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)
        else:
            X_tr_s, X_val_s = X_tr, X_val

        # 3) apply sampler only on the training partition (scaled)
        try:
            # SMOTEENN does not accept float sampling_strategy in many versions -> handle similarly
            if isinstance(method, SMOTEENN):
                sampler = method  # already a SMOTEENN instance
            else:
                sampler = method.set_params(sampling_strategy=sampling_strategy)
            X_tr_res, y_tr_res = sampler.fit_resample(X_tr_s, y_tr)
        except Exception as e:
            # in case sampler fails for given mid: fall back to original X_tr (unresampled)
            if self.verbose:
                print(f"[evaluate_balance] sampler failed with strategy={sampling_strategy}: {e}. Using raw train.")
            X_tr_res, y_tr_res = X_tr_s, y_tr

        # 4) train small MLP and evaluate
        try:
            f1 = self._train_small_mlp_and_eval(X_tr_res, y_tr_res, X_val_s, y_val)
        except Exception as e:
            if self.verbose:
                print(f"[evaluate_balance] model training failed: {e}")
            f1 = -1.0
        return f1

    def binary_search_sampling(self, method, start=0.7, end=1.0, tol=0.01):
        """
        Search for best sampling_strategy in [start, end] that maximizes validation F1.
        Important: at each candidate 'mid' we split first and then resample only inner-train.
        For SMOTEENN we treat sampling_strategy as 'auto'.
        Returns (best_ratio, best_score)
        """
        best_score, best_ratio = -1.0, None

        # Handle SMOTEENN: use 'auto' but still split-first inside evaluate_balance
        if isinstance(method, SMOTEENN):
            try:
                score = self.evaluate_balance(self.X_train, self.y_train, method, 'auto')
                return 'auto', score
            except Exception as e:
                if self.verbose:
                    print(f"[binary_search_sampling] SMOTEENN failed: {e}")
                return None, -1.0

        # For samplers that accept a float sampling_strategy
        while end - start > tol:
            mid = (start + end) / 2.0
            # Evaluate mid by splitting inside evaluate_balance (no pre-resampling)
            try:
                score = self.evaluate_balance(self.X_train, self.y_train, method, mid)
            except Exception as e:
                if self.verbose:
                    print(f"[binary_search_sampling] evaluate failed for mid={mid}: {e}")
                return None, -1.0

            if score > best_score:
                best_score, best_ratio = score, mid

            # Heuristic: check whether this sampling produced >50% positive in the resampled training
            # To decide direction, we perform a short resample on a single split (deterministic)
            try:
                # split deterministically
                # compute mean prop over several splits to stabilize
                n_splits = 3
                props = []
                for i in range(n_splits):
                    X_tr, X_val, y_tr, y_val = train_test_split(
                        self.X_train, self.y_train, test_size=0.2, stratify=self.y_train, random_state=self.random_state + i
                    )
                    if self.scale_inside:
                        scaler = StandardScaler().fit(X_tr)
                        X_tr_s = scaler.transform(X_tr)
                    else:
                        X_tr_s = X_tr
                    sampler = method.set_params(sampling_strategy=mid)
                    X_res, y_res = sampler.fit_resample(X_tr_s, y_tr)
                    props.append(np.mean(y_res))
                prop_mean = np.mean(props)

                # update interval with small margin to avoid oscillation
                margin = 0.02
                if prop_mean > 0.5 + margin:
                    end = mid
                elif prop_mean < 0.5 - margin:
                    start = mid
                else:
                    break  # stop early if near-balanced

            except Exception:
                start = mid

        return best_ratio, best_score

    def bayesian_optimize_strategy(self, method, init_points=4, n_iter=12):
        """
        Optimize sampling_strategy in [0.5, 1.0] using Bayesian Optimization.
        Returns best_strategy (float).
        """

        def objective(ratio):

            val = self.evaluate_balance(self.X_train, self.y_train, method, ratio)
            return float(val)

        pbounds = { "ratio": (0.5, 0.9) }

        optimizer = BayesianOptimization(
            f=objective,
            pbounds=pbounds,
            random_state=self.random_state,
            verbose=0
        )

        optimizer.maximize(init_points=init_points, n_iter=n_iter)

        best_ratio = optimizer.max["params"]["ratio"]
        best_score = optimizer.max["target"]

        if self.verbose:
            print(f"[BayesOpt] Best ratio = {best_ratio:.4f} | score = {best_score:.4f}")

        return best_ratio, best_score

    def auto_select(self):
        """
        Main entry: search across candidate methods and pick the best method+strategy.
        Finally, apply the chosen sampler to the ORIGINAL self.X_train (NOT scaled here)
        and return resampled arrays (X_res, y_res).
        """
        ratio = self.imbalance_ratio()
        if ratio > self.threshold:
            if self.verbose:
                print("[auto_select] Dataset is balanced enough (ratio={:.3f}).".format(ratio))
            return self.X_train, self.y_train

        if self.verbose:
            print(f"Imbalance detected (minority/majority={ratio:.3f}). Starting auto-balancing...")

        for name, method in self.candidate_methods().items():
            try:
                best_ratio, score = self.binary_search_sampling(method)
                self.results[name] = (best_ratio, score)
                if self.verbose:
                    print(f"[auto_select] Method={name} -> best_ratio={best_ratio} | score={score:.4f}")
                if score > self.best_score:
                    self.best_score = score
                    self.best_method = name
                    self.best_strategy = best_ratio
            except Exception as e:
                if self.verbose:
                    print(f"[auto_select] Method {name} failed entirely: {e}")
                continue

        if self.best_method is None:
            if self.verbose:
                print("[auto_select] No sampling method succeeded. Returning original dataset.")
            return self.X_train, self.y_train

        if self.verbose:
            print(f"Best method: {self.best_method} (strategy={self.best_strategy}) | validation F1={self.best_score:.4f}")

        # Apply chosen sampler to the ORIGINAL training data (unscaled) to produce final X_res, y_res
        # Apply chosen sampler to ORIGINAL data
        final_sampler = self.candidate_methods()[self.best_method]

        try:
            if self.best_method != "SMOTEENN" and self.best_strategy not in (None, 'auto'):
                final_sampler = final_sampler.set_params(sampling_strategy=self.best_strategy)

            X_res, y_res = final_sampler.fit_resample(self.X_train, self.y_train)

            # ======== NEW: CHECK IMBALANCE AFTER RESAMPLING ========
            counts = np.bincount(y_res.astype(int))
            minority = min(counts)
            majority = max(counts)

            imbalance_ratio = minority / majority
            if imbalance_ratio < 0.80:   # nghĩa là majority > minority * 1.25
                if self.verbose:
                    print(f"Post-resample imbalance detected: ratio={imbalance_ratio:.3f}. Switching to Bayesian Optimization...")

                # Re-run Bayesian optimization to find best sampling ratio
                method_instance = self.candidate_methods()[self.best_method]
                new_ratio, new_score = self.bayesian_optimize_strategy(method_instance)

                # Apply again using optimized ratio
                sampler_opt = method_instance.set_params(sampling_strategy=new_ratio)
                X_res, y_res = sampler_opt.fit_resample(self.X_train, self.y_train)

                if self.verbose:
                    print(f"Updated strategy from {self.best_strategy} → {new_ratio:.4f} (BayesOpt)")

                self.best_strategy = new_ratio
                self.best_score = new_score

        except Exception as e:
            if self.verbose:
                print(f"[auto_select] Final resampling failed ({e}). Returning original data.")
            return self.X_train, self.y_train

        return X_res, y_res
