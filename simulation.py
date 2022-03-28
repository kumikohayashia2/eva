from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Sequence, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from numbers import Real
from typing import List, Optional, Tuple, Callable, Dict, NoReturn, Union, NamedTuple

import numpy as np
import scipy.optimize as opt
import scipy.stats as st
from matplotlib import pyplot as plt

from common import GEV, plots

# Python 3.9 compatibility
try:
    from typing import TypeAlias
except ImportError:
    TypeAlias = ...

FloatArrayLike: TypeAlias = Union[float, Iterable[float], Sequence[float]]
FloatFunction: TypeAlias = Callable[[FloatArrayLike], FloatArrayLike]
ModelFunction: TypeAlias = Callable[[FloatArrayLike, ...], FloatArrayLike]


class Point(NamedTuple):
    x: float
    y: float


class Line(ABC):
    ratio: float
    x_max: float
    y_max: float
    additional_load: Optional[FloatArrayLike]
    additional_load_probability: Optional[float]


@dataclass
class GeneralLine(Line):
    ratio: float
    x_max: float
    y_max: float

    # y = f(x), 0 <= x <= 1
    function: FloatFunction
    # x = f^-1(y), If not specified, a numerical solution using Newton's method is used.
    inverse_function: Optional[FloatFunction] = None
    is_vector_function: bool = True
    is_vector_inverse_function: bool = True
    is_normalized_function: bool = True
    is_normalized_inverse_function: bool = True
    additional_load: Optional[FloatArrayLike] = None
    additional_load_probability: Optional[float] = None


@dataclass
class LinearLine(Line):
    ratio: float
    x_max: float
    y_max: float
    additional_load: Optional[FloatArrayLike]
    additional_load_probability: Optional[float]

    params: Dict[float, Point]
    is_strict: bool

    @classmethod
    def create(cls, ratio: float, x_max: float, y_max: float, points: List[Point], additional_load: Optional[FloatArrayLike] = None, additional_load_probability: Optional[float] = None, is_strict: bool = False) -> LinearLine:
        points.append(Point(0, y_max))

        # Calculate the slope and intercept of the line from the direction of x_max.
        params = OrderedDict()
        last_p = Point(x_max, 0)

        for p in sorted(points, key=lambda q: q.x, reverse=True):
            params[p.x] = cls.get_linear_parameters(
                Point(last_p.x, last_p.y),
                Point(p.x, p.y)
            )
            last_p = p

        return LinearLine(ratio=ratio, x_max=x_max, y_max=y_max, additional_load=additional_load, additional_load_probability=additional_load_probability, params=params, is_strict=is_strict)

    # Calculate the slope a and intercept b of the line y=ax+b passing through p1(x1, y1), p2(x2, y2).
    @staticmethod
    def get_linear_parameters(p1: Point, p2: Point) -> Point:
        return Point(
            (p1.y - p2.y) / (p1.x - p2.x),
            p1.y - (p1.y - p2.y) / (p1.x - p2.x) * p1.x
        )


class Simulation:
    class LineInternal(ABC):
        base_precision: int
        line: Line

        @property
        def x(self) -> Iterable[float]:
            return np.linspace(0, self.line.x_max, int(self.base_precision * self.line.x_max)) * self.line.x_max

        @property
        def y(self) -> Iterable[float]:
            return self.solve_y(self.x)

        @abstractmethod
        def solve_y(self, x: FloatArrayLike) -> FloatArrayLike:
            raise NotImplementedError

        @abstractmethod
        def solve_x(self, y: FloatArrayLike) -> FloatArrayLike:
            raise NotImplementedError

        @abstractmethod
        def solve(self, a: float, estimate: Optional[float] = None) -> Point:
            raise NotImplementedError

        def solve_with_additional_load(self, a: float, estimate: Optional[float] = None) -> Point:
            solution = self.solve(a, estimate)
            load = self.sample_additional_load()
            if not load:
                return solution

            return Point(solution.x + load, self.solve_y(solution.x + load))

        def choose_additional_load(self) -> Optional[Real]:
            if not self.line.additional_load:
                return
            if isinstance(self.line.additional_load, Real):
                return self.line.additional_load

            return np.random.randint(*self.line.additional_load)

        def sample_additional_load(self) -> Optional[float]:
            load = self.choose_additional_load()
            if not load:
                return

            probability = self.line.additional_load_probability or 0
            return np.random.choice([0, load], p=[1 - probability, probability])

    class GeneralLineInternal(LineInternal):
        def __init__(self, line: GeneralLine, base_precision: int):
            self.line: GeneralLine = line
            self.base_precision = base_precision

            unnormalized_function = (lambda x: np.array(line.function(np.array(x) / line.x_max)) * line.y_max) if line.is_normalized_function else line.function
            self.function = unnormalized_function if line.is_vector_function else np.vectorize(unnormalized_function)

            self.inverse_function = line.inverse_function
            if self.inverse_function:
                unnormalized_inverse_function = (lambda y: np.array(line.inverse_function(np.array(y) / line.y_max)) * line.x_max) if line.is_normalized_inverse_function else line.inverse_function
                self.inverse_function = unnormalized_inverse_function if line.is_vector_inverse_function else np.vectorize(unnormalized_inverse_function)

        def solve_y(self, x: FloatArrayLike) -> FloatArrayLike:
            return self.function(x)

        def solve_x(self, y: FloatArrayLike) -> FloatArrayLike:
            if self.inverse_function:
                return self.inverse_function(y)

            if not isinstance(y, float):
                return [self.solve_x(y_i) for y_i in y]

            try:
                return opt.newton(lambda t: self.function(t) - y, 1e-3)
            except RuntimeError:
                return opt.bisect(lambda t: self.function(t) - y, 0, self.line.x_max)

        def solve_y_normalized(self, x: FloatArrayLike) -> FloatArrayLike:
            return self.solve_y(x * self.line.x_max)

        def solve_x_normalized(self, y: FloatArrayLike) -> FloatArrayLike:
            return self.solve_x(y * self.line.y_max)

        # https://qiita.com/sus304/items/3080b47f783411595404
        def solve(self, a: float, estimate: Optional[float] = None) -> Point:
            x = opt.newton(lambda t: self.function(t) - a * t, estimate or 1e-3)
            return Point(x, self.solve_y(x))

    class LinearLineInternal(LineInternal):
        def __init__(self, line: LinearLine, base_precision: int):
            self.line: LinearLine = line
            self.base_precision = base_precision

        def solve_y(self, x: FloatArrayLike) -> FloatArrayLike:
            if isinstance(x, Iterable):
                return np.vectorize(self.solve_y)(x)

            local_x_max = self.line.x_max
            if x > local_x_max:
                if self.line.is_strict:
                    raise ArithmeticError(f"x ({x}) > x_max ({local_x_max})")
                x = local_x_max

            x_min = 0
            for x_min, point in self.line.params.items():
                if x_min <= x <= local_x_max:
                    return point.x * x + point.y

                local_x_max = x_min

            if self.line.is_strict:
                raise ArithmeticError(f"x ({x}) < x_min ({x_min})")
            return self.line.y_max

        def solve_x(self, y: FloatArrayLike) -> FloatArrayLike:
            if isinstance(y, Iterable):
                return np.vectorize(self.solve_x)(y)

            y_min = 0
            if y < y_min:
                if self.line.is_strict:
                    raise ArithmeticError(f"y ({y}) < y_min ({y_min})")
                y = y_min

            local_y_max = 0
            for y_min, point in self.line.params.items():
                local_y_max = self.solve_y(y_min)
                if y_min <= y <= local_y_max:
                    return (y - point.y) / point.x

            if self.line.is_strict:
                raise ArithmeticError(f"y ({y}) > y_max ({local_y_max})")
            return 0

        # Find the intersection between this line and y=ax+b.
        def solve(self, a: float, estimate: Optional[float] = None) -> Point:
            local_x_max = self.line.x_max
            for x_min, point in self.line.params.items():
                x = point.y / (a - point.x)
                if x_min <= x <= local_x_max:
                    y = a * x
                    return Point(x, y)

                local_x_max = x_min

            raise ArithmeticError(f"No crossing point with y = {a:.2f}x.")

    values: Dict[int, List[Point]]
    extreme_values: List[Point]

    def __init__(
        self,
        lines: List[Line],
        velocity_range: Tuple[float, float],
        base_precision: Optional[int] = None,
        steps: Optional[int] = None,
        block_size: Optional[int | Tuple[int]] = None,
        stokes_law_slope_tuning: Optional[float] = None,
        stokes_law_slope_ppf: Optional[float] = None,
        stokes_law_slope_gamma_a: Optional[float] = None,
        stokes_law_slope_gamma_b: Optional[float] = None,
        seed: Optional[int] = None
    ):
        self.velocity_range = velocity_range
        self.base_precision = base_precision or 1000
        self.steps = steps or 200
        self.block_size = block_size or 10

        if stokes_law_slope_tuning is not None:
            warnings.warn("Specifying stokes_law_slope_tuning is deprecated. Keep this None.")
        self.stokes_law_slope_tuning = stokes_law_slope_tuning or 1

        self.stokes_law_slope_ppf = stokes_law_slope_ppf or 0.99
        self.stokes_law_slope_gamma_a = stokes_law_slope_gamma_a or 6.54
        self.stokes_law_slope_gamma_b = stokes_law_slope_gamma_b or 201

        self.seed = seed
        if seed:
            self.set_seed(seed)

        def raiser() -> NoReturn:
            raise RuntimeError("Unknown Line instance.")

        self.lines = [
            self.GeneralLineInternal(
                line,
                self.base_precision
            ) if isinstance(line, GeneralLine) else (
                self.LinearLineInternal(
                    line,
                    self.base_precision
                ) if isinstance(line, LinearLine) else raiser()
            )
            for line in lines
        ]

        ratio_sum = sum([line.ratio for line in lines])
        self.line_ratios = [line.ratio / ratio_sum for line in lines]

        # Find the range of slope of Stokes' equation from the minimum and maximum velocity from the experiment.
        self.stokes_law_slope_range = (
            max(self.velocity_range[0] / line.solve_x(self.velocity_range[0]) for line in self.lines),
            min(self.velocity_range[1] / line.solve_x(self.velocity_range[1]) for line in self.lines)
        )
        if self.stokes_law_slope_range[0] > self.stokes_law_slope_range[1] or not all(r > 0 for r in self.stokes_law_slope_range):
            raise RuntimeError(f"Stokes' Law's slope is invalid. ({self.stokes_law_slope_range})")

        # Probability distribution function of the slope of Stokes' equation
        self.stokes_law_slope_distribution = st.gamma(self.stokes_law_slope_gamma_a, scale=1 / self.stokes_law_slope_gamma_b)
        # Factor of proportionality multiplied by the slope of Stokes' equation.
        self.stokes_law_slope_amplifier = (self.stokes_law_slope_range[1] - self.stokes_law_slope_range[0]) / self.stokes_law_slope_distribution.ppf(self.stokes_law_slope_ppf) / self.stokes_law_slope_tuning

    def __enter__(self) -> Simulation:
        self.start_time = datetime.now()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = datetime.now()
        print(f"Elapsed: {self.elapsed}")

    @property
    def elapsed(self) -> Optional[timedelta]:
        if not self.start_time or not self.end_time:
            return

        return self.end_time - self.start_time

    def test(self, normalized: bool = False, *args, **kwargs) -> Simulation:
        with self:
            self.execute()

        if normalized:
            self.draw_model()
        else:
            self.draw_demo_model()

        self.draw_stokes_law_slope()
        self.draw_gev(normalized, *args, **kwargs)

        return self

    def set_seed(self, value: Optional[int] = None):
        value = value or np.random.randint(0, 2 ** 31 - 1)
        np.random.seed(value)

        self.seed = value
        print(f"Seed = {value}")

    def choose_line(self) -> LineInternal:
        return np.random.choice(self.lines, p=self.line_ratios)

    def choose_block_size(self) -> int:
        if isinstance(self.block_size, int):
            return self.block_size

        return np.random.randint(*self.block_size)

    def generate_slopes(self) -> List[float]:
        return [
            self.stokes_law_slope_range[0] + self.stokes_law_slope_amplifier * a
            for a in self.stokes_law_slope_distribution.rvs(size=self.choose_block_size())
        ]

    def execute(self):
        # Calculate data.
        self.values = {
            i: [
                # Find the intersection of the randomly generated Stokes equation v=aF and F-v.
                self.choose_line().solve_with_additional_load(a)
                # Generate a random slope a of the Stokes equation.
                for a in self.generate_slopes()
            ]
            for i in range(self.steps)
        }

        # Extract extreme value data.
        self.extreme_values = [max(self.values[i], key=lambda t: t.y) for i in range(self.steps)]

    def gev(self, method: Optional[str] = None, filter_value: Optional[Callable[[float], bool]] = None, *args, **kwargs) -> GEV:
        return GEV([v for _, v in self.extreme_values if not filter_value or filter_value(v)], method, *args, **kwargs)

    def draw(self):
        self.draw_model()

    def draw_model(self):
        # Plot the F-v line.
        for line in self.lines:
            plt.plot(line.x, np.array(line.y) / 1000)

        # Plot the range of slope of Stokes' equation.
        x = np.linspace(0, max([line.line.x_max for line in self.lines]), 2)
        [
            plt.plot(x, slope * x / 1000, "--", color="black", label=f"y={slope:.1f}x")
            for slope in self.stokes_law_slope_range
        ]

        # Plot of all data generated
        plt.plot(*zip(*[(q.x, q.y / 1000) for p in self.values.values() for q in p]), "o", ms=2, color="black")
        # Plot of extreme value data generated
        plt.plot(*zip(*[(p.x, p.y / 1000) for p in self.extreme_values]), "x", ms=4, color="orangered")

        plt.title("F-v relation", fontsize=16)
        plt.xlabel("$ F $ (pN)", fontsize=16)
        plt.ylabel("$ v $ (μm/s)", fontsize=16)
        plt.tick_params(labelsize=14)
        plt.grid()

        plt.xlim(0, max([x.line.x_max for x in self.lines]))
        plt.ylim(0, max([x.line.y_max / 1000 for x in self.lines]))

        plt.legend()
        plt.tight_layout()
        plt.show()

    def draw_demo_model(self, color: Optional[str] = None, start: Optional[int] = None, end: Optional[int] = None, draw_data: bool = True, draw_range: bool = True, draw_label: bool = True):
        # Plot the F-v line.
        for line in self.lines[start:end] if start is not None and end is not None else self.lines:
            plt.plot(line.x, np.array(line.y), lw=2, color=color)

        if draw_range:
            # Plot the range of slope of Stokes' equation.
            x = np.linspace(0, max([line.line.x_max for line in self.lines]), 2)
            [
                plt.plot(x, slope * x, "--", color="black", label=f"y={slope:.1f}x")
                for slope in self.stokes_law_slope_range
            ]

        if draw_data:
            # Plot of all data generated
            plt.plot(*zip(*[(q.x, q.y) for p in self.values.values() for q in p]), "o", ms=2, color="black")
            # Plot of extreme value data generated
            plt.plot(*zip(*[(p.x, p.y) for p in self.extreme_values]), "x", ms=4, color="orangered")

        # Plot of critical points
        [
            plt.plot(p_x, line.solve_y(p_x), "o", color="black", ms=4)
            for line in (self.lines[start:end] if start is not None and end is not None else self.lines)
            if isinstance(line, Simulation.LinearLineInternal)
            for p_x in line.line.params.keys()
            if p_x > 0
        ]

        if draw_label:
            plt.xlabel("Normalized $ F $", fontsize=16)
            plt.ylabel("Normalized $ v $", fontsize=16)
        plt.tick_params(labelsize=14)
        plt.grid(lw=0.5, color="gray", linestyle="--", alpha=0.4)

        x_max = max([x.line.x_max for x in self.lines])
        plt.xlim(0, x_max)
        # plt.xticks([0.0] + [x.line.x_max for x in self.lines], [0] + [i + 1 for i, _ in enumerate(self.lines)])
        x_max_min = min([x.line.x_max for x in self.lines])
        plt.xticks([0, x_max_min / 4, x_max_min / 2, x_max_min * 3 / 4] + [x.line.x_max for x in self.lines], ["0", "0.25", "0.5", "0.75"] + [str(round(x.line.x_max / x_max_min)) for x in self.lines])

        y_max = max([x.line.y_max for x in self.lines])
        plt.ylim(0, y_max)
        y_max_min = min([x.line.y_max for x in self.lines])
        plt.yticks([0, y_max_min / 4, y_max_min / 2, y_max_min * 3 / 4] + [x.line.y_max for x in self.lines], ["0", "0.25", "0.5", "0.75"] + [str(round(x.line.y_max / y_max_min)) for x in self.lines])

        plt.tight_layout()
        plt.show()

    def draw_stokes_law_slope(self):
        a = np.linspace(*self.stokes_law_slope_range, 10000)
        y = self.stokes_law_slope_distribution.pdf(a / self.stokes_law_slope_amplifier)

        ax = plt.gca()
        ax.plot(a, y)
        ax.set_xlabel("Slope $ a $", fontsize=16)
        ax.set_ylabel("Probability", fontsize=16)

        ax2 = ax.twinx()
        for i, line in enumerate(self.lines):
            ax2.plot(a, line.solve_y([line.solve(a_i).x for a_i in a]) / 1000, "--", label=f"Line #{i}")
        ax2.set_ylabel("Velocity (μm/s)", fontsize=16)

        plt.title("Stokes' Law Slope")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def draw_gev(self, normalized: bool, *args, **kwargs):
        gev = self.gev("pyper", *args, **kwargs)
        plots.draw_gev(gev)
        plots.draw_return_level_plot(gev, y_max=max(line.line.y_max for line in self.lines) if normalized else None)


# noinspection PyPep8Naming
class ModelLines:
    # noinspection PyTypeChecker
    @classmethod
    def normalize_model_brentq(cls, x: FloatArrayLike, base_model: ModelFunction, root_range: Optional[Tuple[float, float]] = None, *args) -> FloatArrayLike:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html
        model_x_max = opt.brentq(base_model, *(root_range or (7, 8)), args=args)
        model_y_max = base_model(0, *args)

        return cls.normalize_model(x, base_model, model_x_max, model_y_max)

    @classmethod
    def normalize_model_simple(cls, x: FloatArrayLike, base_model: ModelFunction, x_max: int, *args):
        model_x_max = x_max
        model_y_max = base_model(0, *args)

        return cls.normalize_model(x, base_model, model_x_max, model_y_max)

    @staticmethod
    def normalize_model(x: FloatArrayLike, base_model: ModelFunction, x_max: float, y_max: float, *args) -> FloatArrayLike:
        return base_model(x * x_max, *args) / y_max

    @staticmethod
    def one_state_model(x: FloatArrayLike, x_max: float, a: float, y_min: float) -> FloatArrayLike:
        return y_min * (1 - np.exp(-a * (x - x_max)))

    @classmethod
    def normalized_one_state_model(cls, x: FloatArrayLike, x_max: float, a: float, y_min: float) -> FloatArrayLike:
        return cls.normalize_model_brentq(x, cls.one_state_model, (7, 8), x_max, a, y_min)

    @staticmethod
    def one_state_model_inverse(y: FloatArrayLike, x_max: float, a: float, y_min: float) -> FloatArrayLike:
        return x_max - np.log(1 - y / y_min) / a

    @classmethod
    def one_state_model_example(cls, x: FloatArrayLike) -> FloatArrayLike:
        return cls.one_state_model(x, 8, 1 / 4.12, -100)

    @classmethod
    def one_state_model_inverse_example(cls, y: FloatArrayLike) -> FloatArrayLike:
        return cls.one_state_model_inverse(y, 8, 1 / 4.12, -100)

    @classmethod
    def normalized_one_state_model_example(cls, x: FloatArrayLike) -> FloatArrayLike:
        return cls.normalize_model_brentq(x, cls.one_state_model_example)

    @classmethod
    def normalized_one_state_model_inverse_example(cls, y: FloatArrayLike) -> FloatArrayLike:
        return cls.normalize_model_simple(y, cls.one_state_model_inverse_example, x_max=10000)

    @staticmethod
    def three_state_model(x: FloatArrayLike, k1: float, lambda1: float, d1: float, lambda2: float, d2: float, l0: float, kBT: float) -> FloatArrayLike:
        return (k1 * lambda1 / (lambda1 + k1 * np.exp(d1 * x / kBT)) - lambda2 * np.exp(d2 * x / kBT)) * l0

    @staticmethod
    def three_state_model_inverse(y: FloatArrayLike, k1: float, lambda1: float, d1: float, lambda2: float, d2: float, l0: float, kBT: float) -> FloatArrayLike:
        return (k1 * lambda1 * l0 - lambda1 * y - lambda2 * l0 - k1 * y) / (k1 * y * d1 / kBT + lambda2 * l0 * d2 / kBT)

    @classmethod
    def three_state_model_kinesin(cls, x: FloatArrayLike) -> FloatArrayLike:
        k1 = 110
        lambda1 = 1.7e3
        d1 = 3.5
        lambda2 = 2.1
        d2 = 0.27
        l0 = 8.2
        kBT = 4.12

        return cls.three_state_model(x, k1, lambda1, d1, lambda2, d2, l0, kBT)

    @classmethod
    def three_state_model_inverse_kinesin(cls, y: FloatArrayLike) -> FloatArrayLike:
        k1 = 110
        lambda1 = 1.7e3
        d1 = 3.5
        lambda2 = 2.1
        d2 = 0.27
        l0 = 8.2
        kBT = 4.12

        return cls.three_state_model_inverse(y, k1, lambda1, d1, lambda2, d2, l0, kBT)

    @classmethod
    def three_state_model_dynein(cls, x: FloatArrayLike) -> FloatArrayLike:
        k1 = 203
        lambda1 = 0.27e3
        d1 = 2.1
        lambda2 = 0.24
        d2 = 1.8
        l0 = 8.2
        kBT = 4.12

        return cls.three_state_model(x, k1, lambda1, d1, lambda2, d2, l0, kBT)

    @classmethod
    def three_state_model_inverse_dynein(cls, y: FloatArrayLike) -> FloatArrayLike:
        k1 = 203
        lambda1 = 0.27e3
        d1 = 2.1
        lambda2 = 0.24
        d2 = 1.8
        l0 = 8.2
        kBT = 4.12

        return cls.three_state_model_inverse(y, k1, lambda1, d1, lambda2, d2, l0, kBT)

    @classmethod
    def normalized_three_state_model_kinesin(cls, x: FloatArrayLike) -> FloatArrayLike:
        return cls.normalize_model_brentq(x, cls.three_state_model_kinesin)

    @classmethod
    def normalized_three_state_model_dynein(cls, x: FloatArrayLike) -> FloatArrayLike:
        return cls.normalize_model_brentq(x, cls.three_state_model_dynein)

    @classmethod
    def energy_landscape_model(cls, x: FloatArrayLike) -> FloatArrayLike:
        d = 8
        atp = 4
        kBT = 4.12

        k_cat0 = 488
        q_cat = 6.2 * 1e-3
        delta = 3.7
        k_cat = k_cat0 / (1 - q_cat * (1 - np.exp(x * delta / kBT)))

        k_b0 = 1.3 * 1e3
        q_b = 4 * 1e-2
        k_b = k_b0 / (1 - q_b * (1 - np.exp(x * delta / kBT)))

        return d / (1 / k_cat + 1 / k_b / atp)

    @classmethod
    def normalized_energy_landscape_model(cls, x: FloatArrayLike) -> FloatArrayLike:
        return cls.normalize_model_simple(x, cls.energy_landscape_model, 10)
