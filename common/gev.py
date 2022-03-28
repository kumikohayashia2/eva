from typing import Optional, List, Tuple, Any

import numpy as np
import scipy.stats as st


class GEV:
    xi: float
    xi_se: float
    mu: float
    mu_se: float
    sigma: float
    sigma_se: float
    cov: Any

    def __init__(self, data: List[float], method: Optional[str] = None, *args, **kwargs):
        self.data = np.sort(data)

        if method == "rpy2":
            self.fit_by_rpy2(**kwargs)
        elif method == "pyper":
            self.fit_by_pyper(**kwargs)
        else:
            self.fit_by_scipy(*args, **kwargs)

    def fit_by_scipy(self, *args, **kwargs):
        p = st.genextreme.fit(self.data, *args, **kwargs)
        self.xi = -p[0]
        self.mu = p[1]
        self.sigma = p[2]

    def fit_by_rpy2(self, **kwargs):
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        from rpy2.robjects import numpy2ri

        try:
            numpy2ri.activate()

            ro.r.assign("data", self.data)
            importr("ismev")
            r = ro.r("gev.fit(data)")
            if kwargs.get("verbose"):
                from pprint import pprint
                del r["data"]
                del r["vals"]
                pprint(r)

            self.xi = r["mle"][2]
            self.xi_se = r["se"][2]
            self.mu = r["mle"][0]
            self.mu_se = r["se"][0]
            self.sigma = r["mle"][1]
            self.sigma_se = r["se"][1]
            self.cov = r["cov"]
        finally:
            numpy2ri.deactivate()

    def fit_by_pyper(self, **kwargs):
        import pyper

        pr = pyper.R(use_numpy=True, dump_stdout=kwargs.get("verbose") is not None)
        pr("library(ismev)")
        pr.assign("data", self.data)
        pr("g <- gev.fit(data)")
        r = pr.get("g")
        if kwargs.get("verbose") and r:
            from pprint import pprint
            del r["data"]
            del r["vals"]
            pprint(r)
        if not r:
            raise ArithmeticError("GEV calculation failed.")

        self.xi = r["mle"][2]
        self.xi_se = r["se"][2]
        self.mu = r["mle"][0]
        self.mu_se = r["se"][0]
        self.sigma = r["mle"][1]
        self.sigma_se = r["se"][1]
        self.cov = r["cov"]

    def __str__(self):
        result = f"x_max = {self.bound:8.3f} ± {self.bound_se:7.2f}\n"
        result += f"xi    = {self.xi:8.3f} ± {self.xi_se:7.2f}\n"
        result += f"mu    = {self.mu:8.3f} ± {self.mu_se:7.2f}\n"
        result += f"sigma = {self.sigma:8.3f} ± {self.sigma_se:7.2f}"

        return result

    def confidence_intervals(self, f: np.array) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        import pyper

        pr = pyper.R(use_numpy=True)
        pr("library(ismev)")
        pr(f"mle <- c({self.mu}, {self.sigma}, {self.xi})")
        pr.assign("cov", self.cov)
        pr.assign("f", np.sort(f))

        pr("""
            q <- gevq(mle, 1 - f)
            d <- t(gev.rl.gradient(a = mle, p = 1 - f))
            v <- apply(d, 1, q.form, m = cov)
        """)
        q = pr.get("q")
        v = pr.get("v")

        return [
            (
                -1 / np.log(f_i),
                q_i + 1.96 * np.sqrt(v_i)
            )
            for f_i, q_i, v_i in zip(f, q, v)
        ], [
            (
                -1 / np.log(f_i),
                q_i - 1.96 * np.sqrt(v_i)
            )
            for f_i, q_i, v_i in zip(f, q, v)
        ]

    @property
    def distribution(self):
        return st.genextreme(c=-self.xi, loc=self.mu, scale=self.sigma)

    @property
    def bound(self):
        return self.mu - self.sigma / self.xi

    @property
    def bound_se(self):
        return np.sqrt(
            np.power(self.mu_se, 2)
            +
            np.power(self.sigma_se / self.xi, 2)
            +
            np.power(self.sigma * self.xi_se / np.power(self.xi, 2), 2)
        )

    def value(self, i):
        return (
                self.mu
                -
                self.sigma
                /
                self.xi
                *
                (
                    1
                    -
                    np.power(
                        -np.log(
                            (i + 1)
                            /
                            (len(self.data) + 1)
                        ),
                        -self.xi
                    )
                )

        )

    def value_se(self, i):
        return np.sqrt(
            np.power(
                self.bound
                *
                np.log(
                    -np.log(
                        (i + 1)
                        /
                        (len(self.data) + 1)
                    )
                )
                *
                np.power(
                    -np.log(
                        (i + 1)
                        /
                        (len(self.data) + 1)
                    ),
                    -self.xi
                )
                *
                self.xi_se,
                2
            )
            +
            np.power(
                self.bound_se
                *
                (
                    1
                    -
                    -np.log(
                        (i + 1)
                        /
                        (len(self.data) + 1)
                    )
                ),
                2
            )
        )
