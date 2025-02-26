# easyAnomalyFinder

A lightweight script to find anomalous signals in time-domain light curves based on residuals. Initially designed to detect microlensing planets, but it can be used for broader purposes.

## Acknowledgements
If your scientific research uses or is based on this code, please cite [ TBD ]

## Installation
Clone this repository to your working directory.

Dependencies: [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [Matplotlib](https://matplotlib.org/)

You can test your installation by running this command in your shell:
```bash
python easyAnomalyFinder.py
```

By default, this runs the anomaly search for the microlesing event [KMT-2018-BLG-0900](https://ui.adsabs.harvard.edu/abs/2022MNRAS.510.1778W), which has a known anomaly caused by a planet.

This should give you a new folder, data_for_test/anom/kb180900, with two figures inside, showing the entire light curve and the detected anomaly at t~8176, respectively.

## Example: Customize
The script can be easily customized.

For example, you can create a new Python script, `import easyAnomalyFinder`, and customize most of the options and parameters. All supported arguments can be found in the following detailed API section.
```python
import easyAnomalyFinder
# Assuming your residual file have 4 columns, including date, flux, flux uncertainty, and the flux model value, and all from a single observational site.
residual_data = np.genfromtxt('Your_Residual_File.dat',
                              dtype=[('date','f8'),('flux','f8'),('ferr','f8'),('flux_model','f8')])
data_kwargs = {'date':residual_data['date'], 
               'flux':residual_data['flux'], 
               'ferr':residual_data['ferr'], 
               'flux_model':residual_data['flux_model']}
AF = easyAnomalyFinder.AnomalyFinder(data_kwargs=data_kwargs,eventname='AnyNameYouLike')

# Manage the AnomalyFinder options and setup
AF_options = {'mode':'detection',
              'if_rescale_error_by_fwhm':False, # because you have no fwhm (seeing) infomation
              'if_check_multiple_sites':False,  # because you have only one observational site
              'if_check_smooth_poly':False,     # for example, you do not want to check if the signal is smooth
              'verbose':False,
              }
AF.set_up(**AF_options)

### Run AnomalyFinder ###
anomaly_list = AF.find_anomaly(verbose=True)

### Save the anomaly list & plot the anomaly figure(s) ###
save_dir = 'YourOutputDir'
header = 't_start t_window ind_start ind_end window_npt chi2_window-window_npt chi2_window/window_npt chi2_poly poly_order z/σz max(dF/F) RMS_ratio'
if len(anomaly_list)>0:
    np.savetxt(save_dir+'/anomaly_list.dat',anomaly_list,fmt='%.6f %10.6f %6d %6d %4d %12.6f %12.6f %12.6f %1d %12.6f %12.6f %12.6f',header=header)
for ianom,anom in enumerate(anomaly_list):
    AF.plot_anomaly(ianom=ianom,saveto=save_dir+'/anom_%03d.png'%(ianom))
```


## Detailed API

### easyAnomalyFinder.AnomalyFinder

**easyAnomalyFinder.AnomalyFinder(data_kwargs, eventname)**

#### Parameters:
- **data_kwargs**: 10 valid kwargs in total, they are:
  - **'date': (n,) array**
  
    *(Necessary)* Times of the observations.

  - **'flux': (n,) array**

    *(Necessary)* Input observed flux values, must be of the same length as 'date'.

  - **'ferr': (n,) array**

    *(Necessary)* Input observed flux uncertainties, must be of the same length as 'date'.
    
  - **'flux_model': (n,) array**

    *(Necessary)* Input model flux, must be of the same length as 'date'.

  - **'chi': (n,) array**
    
    *(Optional)* Residual fluxes divided by the uncertainties, must be of the same length as 'date'. If not provided, will be internally derived by ('flux' - 'flux_model') / 'ferr'.

  - **'chi2': (n,) array**
    
    *(Optional)* $\chi^2$ value of each point, must be of the same length as 'date'. If not provided, will be internally derived by 'chi'**2.

  - **'fwhm': (n,) array**
    
    *(Optional)* Seeing values in arcseconds, must be of the same length as 'date'. Only needed when `if_rescale_error_by_fwhm = True` in the setup function, or if an output figure with seeing is wanted. If not provided, will be initialized as an array of -1. 

  - **'sky': (n,) array**
    
    *(Optional)* Sky background values, must be of the same length as 'date'. Only needed when `if_rescale_error_by_fwhm = True` in the setup function, or if an output figure with seeing is wanted. If not provided, will be initialized as an array of ones. 

  - **'site': (n,) array**
    
    *(Optional)* Observational site name of each data point, string array is supported, must be of the same length as 'date'. Only needed when `if_check_multiple_sites = True` in the setup function. If not provided, will be initialized as an array of ones. 

  - **'chi2_ref': (n,) array**
    
    *(Optional)* Reference $\chi^2$ of each point, must be of the same length as 'date'. Used to be compared to $\chi^2_{\rm window}$. Only needed when `mode = 'sensitivity'` in the setup function (see also the `dchi2_ref_low` option). If not provided, will be initialized as an array of zeros. 

#### Returns:
- **easyAnomalyFinder.AnomalyFinder instance**

---
### easyAnomalyFinder.AnomalyFinder.set_up
**easyAnomalyFinder.AnomalyFinder.set_up.set_up(\*\*kwargs)**

#### Parameters:
- **kwargs**: 
  - **mode: str**

    Supported values: 
      - `'detection'` for finding anomalies, *default*; 
      - `'sensitivity'` for sensitivity calculation, stops when the first anomaly is found to speed up; 
      - `'all'`: return results for all time windows including non-anomalous ones;

  - **sigma: float**

    How significant the anomaly should be. *default: 4.0*.

  - **chi2_low: float**

    Minimum $\chi^2$ value of the anomaly. *default: 80.0*.

  - **time_windows: (m,) array**

    Time window lengths for searching. *default: 16 log-uniform values for 0.02d--2d and 20 log-uniform values for 2d--2000d*.

  - **time_step: float**
    
    Time step in the unit of the window length. *default: 0.1*.

  - **window_npt_thres: int**

    Minimum data points allowed in a window. *default: 3*.

  - **coverage_min_frac: float**

    Minimum time coverage fraction allowed in a window. *default: 0.5*.

  - **if_rescale_error_by_fwhm: bool**

    Whether to rescale errors in seeing bins. *default: False*.

  - **if_rescale_error_by_sky: bool**

    Whether to rescale errors in sky background bins. *default: False*.

  - **errfac_min: float**

    Minimum allowed value for the error factor when rescaling the errors. *default: 1.0*.

  - **errfac_max: float**

    Maximum allowed value for the error factor when rescaling the errors. *default: 3.0*.

  - **if_check_RMS: bool**

    Whether to check the intrinsic variability. *default: True*.

  - **RMS_ratio_sigma: float**

    How significant the anomaly should be in the intrinsic variability checks in sigmas. *default: 1.0*.

  - **if_check_multiple_sites: bool**

    Whether to check if the time window includes data from more than one observational site. *default: True*.

  - **if_check_continuous: bool**

    Whether to check if the max-SNR point has the same residual sign as its surrounding data points (all positive or all negative). *default: False*.

  - **continuous_check_length: int**

    Only valid when `if_check_continuous = True`. How many equal-sign surrounding points are required. *default: 3*.

  - **if_check_chi2_domination: bool**

    Whether to check if the signal is dominated by a small fraction of the largest $\chi^2$ data points. *default: True*.

  - **if_check_fwhm_domination: bool**

    Whether to check if the signal is dominated by a small fraction of the worst (largest) seeing data points. *default: False*.

  - **if_check_sky_domination: bool**

    Whether to check if the signal is dominated by a small fraction of the worst (largest) sky background data points. *default: False*.

  - **drop_frac: float**

    Fraction of the points that `if_check_chi2_domination`, `if_check_fwhm_domination`, and `if_check_sky_domination` check. *default: 0.2*.

  - **chi2_drop_frac: float**

    Fraction of the points that `if_check_chi2_domination` checks. If provided, overwrites `drop_frac` value. *default: drop_frac*.

  - **fwhm_drop_frac: float**

    Fraction of the points that `if_check_fwhm_domination` checks. If provided, overwrites `drop_frac` value. *default: drop_frac*.

  - **sky_drop_frac: float**

    Fraction of the points that `if_check_sky_domination` checks. If provided, overwrites `drop_frac` value. *default: drop_frac*.

  - **if_check_smooth_poly: bool**

    Whether to check if the signal is smooth using a polynomial fit. *default: True*.

  - **poly_consistent_thres: float**

    Maximum allowed relative $\chi^2$ improvement from the polynomial fit. *default: 0.3*.

  - **dchi2_ref_low**

    Minimum allowed $\Delta\chi^2$ used when `mode='sensitivity'`, compared with the $\chi^2_{\rm window}$. *default: 50*.

  - **verbose: bool**

    Whether to print verbose output. *default: False*.