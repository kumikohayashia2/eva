from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

from .gev import GEV


def apply_common_parameters(dpi=300):
    for font in font_manager.findSystemFonts(fontpaths=["../fonts"]):
        font_manager.fontManager.addfont(font)

    plt.rc("text", usetex=False)
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["IBM Plex Sans", "IBM Plex Sans JP"]

    # plt.rcParams["mathtext.fontset"] = "custom"
    #
    # plt.rcParams["mathtext.it"] = "Cambria:italic"
    # plt.rcParams["mathtext.bf"] = "Cambria:italic"
    # plt.rcParams["mathtext.rm"] = "IBM Plex Sans"
    # plt.rcParams["mathtext.sf"] = "Cambria:italic"
    # plt.rcParams["mathtext.tt"] = "IBM Plex Sans"
    # plt.rcParams['font.sans-serif'] = "cm"
    # plt.rcParams['mathtext.default'] = "it"

    from matplotlib import mathtext
    plt.rcParams['mathtext.fontset'] = 'stix'
    mathtext.FontConstantsBase = mathtext.ComputerModernFontConstants
    mathtext.FontConstantsBase.script_space = 0.01
    mathtext.FontConstantsBase.delta = 0.01
    mathtext.FontConstantsBase.sub = 0
    mathtext.FontConstantsBase.sub2 = 0
    mathtext.FontConstantsBase.subdrop = 0

    plt.rcParams["font.size"] = 11


def draw_gev(gev: GEV, title: Optional[str] = None, color: Optional[str] = None, show: bool = True):
    d = gev.data / 1000
    x = np.linspace(0, 7, 1000)

    plt.plot(x, gev.distribution.pdf(x * 1000) * 1000, color=color)
    plt.hist(d, density=True, alpha=0.4, color=color)

    plt.title(title)
    # plt.xlabel(r"$ v_{\mathrm{max}} $ (μm/s)", fontsize=16)
    # plt.ylabel(r"$ w(v_{\mathrm{max}}) $", fontsize=16)
    # plt.tick_params(labelsize=14)
    plt.grid()
    plt.tight_layout()

    if show:
        plt.show()


def draw_return_level_plot(gev: GEV, title: Optional[str] = None, color: Optional[str] = None, show: bool = True, y_max: Optional[float] = None):
    n = gev.data.size
    y_max = y_max or 1000
    d = [(1 / -np.log((i + 1) / (n + 1)), x_i / y_max) for i, x_i in enumerate(gev.data)]
    plt.plot(*zip(*d), "o", ms=3, color=color)

    # x 軸を片対数にする
    ax = plt.gca()
    ax.set_xscale("log")

    plt.title(title)
    # plt.xlabel("Return Period", fontsize=16)
    # plt.ylabel("Return Level (μm/s)", fontsize=16)
    # plt.tick_params(labelsize=14)
    plt.grid()
    plt.xlim(1e-1, 1e3)
    plt.tight_layout()

    if show:
        plt.show()
