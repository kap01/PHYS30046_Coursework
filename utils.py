import pandas as pd
import numpy as np
import uproot
import vector
np.random.seed(123456789)
import matplotlib.pyplot as plt
import iminuit
import mplhep as hep
from scipy.integrate import quad
from functools import partial
from scipy.interpolate import UnivariateSpline

plt.style.use(hep.style.LHCb2)


''' 
Imports required for the likelihood minimisation algorithm that we will use to fit the
generated data in the above cell in order to determine gZ
'''
# everything in iminuit is done through the Minuit object, so we import it
from iminuit import Minuit

# we also need a cost function to fit and import the Binned Log Likelihood function
from iminuit.cost import BinnedNLL
from iminuit.cost import LeastSquares

# display iminuit version
import iminuit
print("iminuit version:", iminuit.__version__)


import numpy as np
from scipy.interpolate import interp1d

def gen_data(n_events, func, x_min, x_max, npoints=100_000, *args, **kwargs):
    """
    Generate random samples distributed according to an arbitrary
    non‑negative function using inverse transform sampling.

    Parameters
    ----------
    n_events : int
        Number of samples to draw.
    func : callable
        The target function. It must accept the sample variable as its first
        argument. Additional parameters for func can be supplied via *args
        and **kwargs.
    x_min, x_max : float
        Domain over which to sample.
    npoints : int, optional
        Number of grid points used to build the CDF (higher values give
        a better approximation of the integral).
    *args, **kwargs :
        Additional arguments passed to func.

    Returns
    -------
    ndarray
        Array of samples distributed according to func on [x_min, x_max].
    """
    # Create a grid on the desired domain
    x_grid = np.linspace(x_min, x_max, npoints)

    # Evaluate the function on the grid.  Try vectorised evaluation,
    # otherwise fall back to a loop.
    try:
        f_vals = func(x_grid, *args, **kwargs)
    except Exception:
        f_vals = np.array([func(x, *args, **kwargs) for x in x_grid])

    # Ensure the function is non‑negative; if it can be negative, clip to zero
    f_vals = np.clip(f_vals, a_min=0.0, a_max=None)

    # Construct the CDF by cumulative summation and normalise it
    cdf = np.cumsum(f_vals)
    cdf /= cdf[-1]  # now cdf[-1] == 1

    # Create the inverse CDF via interpolation
    inv_cdf = interp1d(cdf, x_grid, bounds_error=False,
                       fill_value=(x_min, x_max))

    # Sample uniformly on [0,1] and map through the inverse CDF
    random_probs = np.random.rand(n_events)
    sampled_x = inv_cdf(random_probs)

    return sampled_x


'''
The function below defines the cumulative distribution function of the cross-section expression
that we will feed into the minimisation algorithm
'''
def func_cdf(func, bin_edges,*params):
    xmin, xmax = bin_edges[0], bin_edges[-1]

    
    def indef_integr(xx):
        val, _ = quad(func, xmin, xx,args=params)
        return val

    norm = indef_integr(xmax)

    # evaluate for each bin edge
    integr = np.array([indef_integr(xx) for xx in np.atleast_1d(bin_edges)])
    return (integr - indef_integr(xmin)) / norm


'''
The function below defines a binned version of a function that can
pass into a minimisation algorithm. Note pass bins but don't use.
This is because minuit requires x variable to have same lenght as
input measurements
'''
def func_binned(func, bins,*params, **kwargs):

    bin_edges = kwargs.get("bin_edges")
    xmin, xmax = bin_edges[0], bin_edges[-1]
    def indef_integr(xx):
        val, _ = quad(func, xmin, xx,args=params)
        return val

    # evaluate for each bin edge
    integr = np.array([indef_integr(xx) for xx in np.atleast_1d(bin_edges)])

    bin_widths = np.diff(bin_edges)
    binned_func = np.diff(integr)
    
    return binned_func/bin_widths


'''
The function below defines a binned version of a function that can
pass into a minimisation algorithm. It computes the rate average
observable where the Z lineshape gives the rate. Note pass bins 
but don't use. This is because minuit requires x variable to have same lenght as
input measurements
'''
def z_lineshape(x):
        mZ = 91.1876 # GeV
        gZ = 2.4955  # GeV
        s = x**2
        mZsq = mZ**2
        sigma = np.pi*s/(mZsq*(s-mZsq)**2+mZsq*gZ**2)
        return sigma

def func_rate_avg_binned(func, bins, *params, **kwargs):
        
    bin_edges = kwargs.get("bin_edges")
    xmin, xmax = bin_edges[0], bin_edges[-1]
    def indef_integr(xx):
        val, _ = quad(func, xmin, xx,args=params)
        return val

    def indef_integr_rate(xx):
        val, _ = quad(z_lineshape, xmin, xx)
        return val
    
    # evaluate for each bin edge
    integr = np.array([indef_integr(xx) for xx in np.atleast_1d(bin_edges)])
    rate_integr = np.array([indef_integr_rate(xx) for xx in np.atleast_1d(bin_edges)])
    
    bin_widths = np.diff(bin_edges)
    binned_func = np.diff(integr)
    binned_rate_func = np.diff(rate_integr)

    return binned_func/binned_rate_func






'''
The function below opens rootfiles from predetermined location and converts into 
pandas dataframe with muon kinematics flattened and in the right ordering
'''
def open_dimuon_files_and_reduce():
    # --- Input files (add as many as you want) ---
    input_files = [
        "skimmed_dimuon-Run2012B_DoubleMuParked_v2.root",
        "skimmed_dimuon-Run2012C_DoubleMuParked_v2.root"
    ]
    # --- Open all files and load branches ---
    mu_pt_list     = []
    mu_eta_list    = []
    mu_phi_list    = []
    mu_mass_list   = []
    mu_charge_list = []

    
    for fname in input_files:
        with uproot.open("input_files/"+fname) as f:
            tree = f["Events"]
            print(f"File: {fname}, entries: {tree.num_entries}")
            arrays = tree.arrays(
                ["Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass","Muon_charge"],
                library="np"
            )
    
        mu_pt_list.append(arrays["Muon_pt"])
        mu_eta_list.append(arrays["Muon_eta"])
        mu_phi_list.append(arrays["Muon_phi"])
        mu_mass_list.append(arrays["Muon_mass"])
        mu_charge_list.append(arrays["Muon_charge"])

        # --- Concatenate across files ---
        mu_pt     = np.concatenate(mu_pt_list)
        mu_eta    = np.concatenate(mu_eta_list)
        mu_phi    = np.concatenate(mu_phi_list)
        mu_mass   = np.concatenate(mu_mass_list)
        mu_charge = np.concatenate(mu_charge_list)
        # --- invariant mass of the muon pair ---
        px = mu_pt * np.cos(mu_phi)
        py = mu_pt * np.sin(mu_phi)
        pz = mu_pt * np.sinh(mu_eta)
        E  = np.sqrt(px**2 + py**2 + pz**2 + mu_mass**2)
        Q  = mu_charge
        mass_mumu = np.sqrt(
            (E[:, 0] + E[:, 1])**2 -
            ((px[:, 0] + px[:, 1])**2 + (py[:, 0] + py[:, 1])**2 + (pz[:, 0] + pz[:, 1])**2)
        )
        
    data = {
        "mu1_px": px[:,0],
        "mu2_px": px[:,1],
        "mu1_py": py[:,0],
        "mu2_py": py[:,1],
        "mu1_pz": pz[:,0],
        "mu2_pz": pz[:,1],
        "mu1_E": E[:,0],
        "mu2_E": E[:,1],
        "mu1_Q": Q[:,0],
        "mu2_Q": Q[:,1]        
    }
    df = pd.DataFrame(data)
    print(df)

    # mask: rows where muon1 is actually the positive muon
    # we do this as the paper needs muon1 to be Q=-1 and muon2 Q=+1
    mask = (df['mu1_Q'] == +1) & (df['mu2_Q'] == -1)
    df.loc[mask, ['mu1_px','mu2_px']] = df.loc[mask, ['mu2_px','mu1_px']].to_numpy()
    df.loc[mask, ['mu1_py','mu2_py']] = df.loc[mask, ['mu2_py','mu1_py']].to_numpy()
    df.loc[mask, ['mu1_pz','mu2_pz']] = df.loc[mask, ['mu2_pz','mu1_pz']].to_numpy()
    df.loc[mask, ['mu1_E','mu2_E']] = df.loc[mask, ['mu2_E','mu1_E']].to_numpy()
    df.loc[mask, ['mu1_Q','mu2_Q']] = df.loc[mask, ['mu2_Q','mu1_Q']].to_numpy()
    print(df)

    # Write
    with uproot.recreate("input_files/cms_muons.root") as f:
        # "Events" will be the TTree name
        f["Events"] = {col: df[col].to_numpy() for col in df.columns}
    
    return df




def compute_afb_correction():
    # Obtain corrections regarding AFB dilution
    # Load CSV with no header
    afb_raw_files = [pd.read_csv("input_files/AFB_raw_qqbar_0.0-0.4.csv", header=None),
                     pd.read_csv("input_files/AFB_raw_qqbar_0.4-0.8.csv", header=None),
                     pd.read_csv("input_files/AFB_raw_qqbar_0.8-1.2.csv", header=None),
                     pd.read_csv("input_files/AFB_raw_qqbar_1.2-1.6.csv", header=None),
                     pd.read_csv("input_files/AFB_raw_qqbar_1.6-2.0.csv", header=None),
                     pd.read_csv("input_files/AFB_raw_qqbar_2.0-2.4.csv", header=None)
                    ]
    afb_true_file = pd.read_csv("input_files/AFB_true_qqbar.csv")
    afb_raw_qqbar_all_file = pd.read_csv("input_files/AFB_raw_qqbar_all.csv")
    bins = np.array([60,70,78,84,87,89,91,93,95,98,104,112,120]) # binning as per paper
    def create_spline(x,y,descr="",plot=False):
        # Create spline (exact interpolation)
        spline = UnivariateSpline(x, y, s=0)
        # Evaluate spline on a finer grid
        x_fine = np.linspace(x.min(), x.max(), 500)
        y_fine = spline(x_fine)
        # Plot
        if plot is True:
            plt.figure(figsize=(5,5))
            plt.scatter(x, y, label=f"Data {descr}", color="red")
            plt.plot(x_fine, y_fine, label="Spline", color="blue")
            plt.xlabel(r"$m_{\ell\ell}$ [GeV]")
            plt.ylabel(r"$A_{FB}$")
            plt.legend()
            plt.show()
        return spline

    afb_raw_qqbar = []
    for df in afb_raw_files:
        df.columns = ["mass", "AFB"]
        x = df["mass"].values
        y = df["AFB"].values
        # Create spline (exact interpolation)
        spline = create_spline(x,y)
        afb_raw_qqbar.append(spline)
    
   
    afb_true_file.columns = ["mass", "AFB"]
    x = afb_true_file["mass"].values
    y = afb_true_file["AFB"].values
    # Create spline (exact interpolation)
    afb_true_spline = create_spline(x,y,"qqbar true")

    afb_raw_qqbar_all_file.columns = ["mass", "AFB"]
    x = afb_raw_qqbar_all_file["mass"].values
    y = afb_raw_qqbar_all_file["AFB"].values
    # Create spline (exact interpolation)
    afb_raw_qqbar_all_spline = create_spline(x,y,"qqbar true")

    # weights for rapidity regions extracted from dataframe thusly
    '''
    pt_ok = ((df["pt1"] > 25) & (df["pt2"] > 15)) | ((df["pt1"] > 15) & (df["pt2"] > 25))
    masks = [ pt_ok & ( (np.abs(df["rap"])<0.4) & (np.abs(df["rap"])>0.0) ) ,
          pt_ok & ( (np.abs(df["rap"])<0.8) & (np.abs(df["rap"])>0.4) ) ,
          pt_ok & ( (np.abs(df["rap"])<1.2) & (np.abs(df["rap"])>0.8) ) ,
          pt_ok & ( (np.abs(df["rap"])<1.6) & (np.abs(df["rap"])>1.2) ) ,
          pt_ok & ( (np.abs(df["rap"])<2.0) & (np.abs(df["rap"])>1.6) ) ,
          pt_ok & ( (np.abs(df["rap"])<2.4) & (np.abs(df["rap"])>2.0) )]
    n_evts_per_rap = np.array([df.loc[mask].shape[0] for mask in masks])
    n_wgts_rap = n_evts_per_rap/np.sum(n_evts_per_rap)
    '''
    n_wgts_rap = [0.2222552,  0.22249677, 0.21543019, 0.18288642, 0.11795448, 0.03897694]
   
    # now compute rapidity weighted average AFB raw
    fine_mass = np.linspace(bins[0],bins[-1],500)
    afb_raw_wgt = np.zeros_like(fine_mass, dtype=float)  # start with zeros
    tot_wgt = 0
    for afbs,wgt in zip(afb_raw_qqbar,n_wgts_rap):
        afb_raw_wgt += np.array(afbs(fine_mass))*wgt
        tot_wgt+=wgt

    afb_raw_wgt_spline =  create_spline(fine_mass,afb_raw_wgt,descr="correction",plot=True)
    #plt.figure(figsize=(5,5))
    #plt.plot(fine_mass, afb_raw_wgt, label="Spline", color="blue")
    #plt.xlabel(r"$m_{\ell\ell}$ [GeV]")
    #plt.ylabel(r"$A_{FB}$")
    #plt.legend()
    #plt.show()


    # correction... note only works for integrated rapidity
    ad_hoc_corr = 0.6*np.array([0.9,0.8,0.5,-0.000,-2.7,2.0,0.4,0.5,0.7,1.,1.3,1.])
    bc = 0.5*np.array(bins[1:]+bins[:-1]) 
    ad_hoc_corr_spline_vals = np.interp(fine_mass, bc, ad_hoc_corr)
   
    
    # now compute the true AFB in the same mass values
    # and take correction
    afb_true_vals = np.array(afb_true_spline(fine_mass))
    afb_raw_qqbar_all_vals = np.array(afb_raw_qqbar_all_spline(fine_mass)) 
    afb_corr = afb_raw_wgt/afb_true_vals*ad_hoc_corr_spline_vals 

    afb_corr_spline = create_spline(fine_mass,afb_corr,descr="correction",plot=True)
    return afb_corr_spline, afb_raw_wgt_spline 
