import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

def plot_results(results, fit_range, steps_ahead, lenSeries, xlim_lo=None, xlim_hi=None,
                 ylim_lo=None, ylim_hi=None, xres_lo=None, xres_hi=None):
    plt.rcParams["figure.figsize"] = (10,8)
    fig, sub = plt.subplots(3, 1)
    results_comp = results.loc[(fit_range+steps_ahead-1):(lenSeries-1),:]  # Only where observations exist
    for i in range(1,len(results.columns)):
        sub[0].plot(results[results.columns[i]], linewidth=1.0, label = results.columns[i])
        if i>1:
            data = np.asarray(results_comp['Close']-results_comp[results_comp.columns[i]], dtype=np.float64)
            sub[1].plot(results_comp.index, data, label=results_comp.columns[i])
            density = gaussian_kde(data)
            density.covariance_factor = lambda : .1
            density._compute_covariance()
            xs = np.linspace(xres_lo, xres_hi, 100)
            sub[2].plot(xs, density(xs), label=results_comp.columns[i])
    sub[0].set_xlabel("Time")
    sub[0].set_ylabel("Close")
    sub[0].set_xlim(xlim_lo, xlim_hi)
    sub[0].set_ylim(ylim_lo, ylim_hi)
    sub[0].legend()
    sub[1].set_xlabel("Time")
    sub[1].set_ylabel("Residuals")
    sub[1].set_xlim(xlim_lo, xlim_hi)
    sub[1].legend()
    sub[2].set_xlabel("Residuals")
    sub[2].set_ylabel("KDE")
    sub[2].legend()
    fig.tight_layout()
    plt.show()
