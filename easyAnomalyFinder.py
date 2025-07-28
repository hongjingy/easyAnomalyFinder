# Copyright (c) [2025] [杨弘靖/YANG Hongjing]
# Please cite the following paper when using this code:
# 
# "Systematic Reanalysis of KMTNet Microlensing Events. II. Two New Planets in Giant-source Events", Yang et al., The Astronomical Journal, 2025, DOI:10.3847/1538-3881/adc73e, https://ui.adsabs.harvard.edu/abs/2025AJ....169..295Y


import numpy as np
from scipy.stats import chi2 as chi2_distribution
from scipy.stats import chi as chi_distribution
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
import os

class BaseClass():
    def _set_value(self, kwargs, key, default_value):
        ''' function to set self.key with kwargs or default value '''
        value = kwargs[key] if (key in kwargs) else default_value
        setattr(self, key, value)
        return value
    
class AnomalyFinder(BaseClass):
    def __init__(self, 
                 data_kwargs,
                 eventname='noname'):
        '''
        data_kwargs: dict
            supported keys: date, chi, chi2, flux, ferr, fwhm, sky, amp_model, flux_model, site, chi2_ref.
            only date, flux, ferr, flux_model are necessary;
                chi, chi2 can be internally calculated from flux, ferr, flux_model;
            other keys are optional and related to some specific checks:
                fwhm: used to rescale error by seeing;
                sky: used to rescale error by sky background;
                site: used to check multiple sites data;
                amp_model: currently not used;
                chi2_ref: used in sensitivity mode to compare with each chi2_window.
        eventname: str
            name of the event, not necessary.
        '''
        # data_kwargs = {'date':0, 'chi':1, 'chi2':2, 'flux':3, 'ferr':4, 'fwhm':5, 'sky':6, 'amp_model':7, 'flux_model':8, 'site':9}

        self.verbose = False
        self.eventname = eventname

        self.get_data(**data_kwargs)
        self.mode = None

        self.if_rescale_error_by_fwhm = False
        self.if_rescale_error_by_sky  = False
        self.errfac_min = 1.0
        self.errfac_max = 3.0

        self.time_windows = None
        self.time_step = None
        self.window_npt_thres = None
        self.coverage_min_frac = None
        self.sigma = None
        self.survival_prob = None
        self.chi2_low = None
        self.chi2_thres_dict = {}
        self.dchi2_ref_low = None

        self.if_check_RMS = True
        self.RMS_ratio_sigma = None # -- chi distribution
        self.RMS_ratio_survival_prob = None
        self.chi_thres_dict = {}
        # self.RMS_ratio_thres = None

        self.if_check_multiple_sites = True

        self.if_check_continuous = False
        self.continuous_check_length = None
        self.continuous_sigma_thres = None

        self.if_check_chi2_domination = True
        self.if_check_fwhm_domination = False
        self.if_check_sky_domination = False
        self.drop_frac = None
        self.chi2_drop_frac = None
        self.fwhm_drop_frac = None
        self.sky_drop_frac = None

        self.if_check_smooth_poly = True
        self.poly_consistent_thres = None
        # plot parameters
        self.cstrs_kmta = ['#34F134','#66CD1B','#149214','#79f179']
        self.cstrs_kmtc = ['#FF3B26','#FF754A','#BD2413','#8F332A']
        self.cstrs_kmts = ['#004FF9','#3C7AFF','#76A1FF','#154ABC']
        self.cstrs = []
        self.has_init_plot = False

    def get_data(self, **kwargs):
        self.date = self._set_value(kwargs, 'date', None)
        self.flux = self._set_value(kwargs, 'flux', None)
        self.ferr = self._set_value(kwargs, 'ferr', None)
        self.flux_model = self._set_value(kwargs, 'flux_model', None)
        if any([getattr(self, attr) is None for attr in ['date','flux','ferr','flux_model']]):
            raise ValueError('date, flux, ferr, and flux_model must be input!')
        
        self.chi  = self._set_value(kwargs, 'chi', (self.flux-self.flux_model)/self.ferr)
        self.res_sign = np.sign(self.chi).astype('int')
        self.chi2 = self._set_value(kwargs, 'chi2', (self.chi)**2)
        self.fwhm = self._set_value(kwargs, 'fwhm', -1*np.ones_like(self.date))
        self.sky  = self._set_value(kwargs,  'sky', 1.0*np.ones_like(self.date))
        # self.amp_model  = chi2data[:,7]
        self.sitefield = self._set_value(kwargs,  'site', np.ones_like(self.date, dtype='str'))
        # self.site = np.array([s[:4] for s in self.sitefield])
        self.site = np.copy(self.sitefield)
        self.obsnames = self._sort_obsnames()
        self.chi2_ref = self._set_value(kwargs, 'chi2_ref', np.zeros_like(self.date))


    def _sort_obsnames(self,):
        obsnames = np.sort(np.unique(self.sitefield))
        ### try to sort sitefield as [kmta01, kmta41, kmtc01, kmtc41, kmts01, kmts41] ... ###
        try:
            obsnames_fields = np.sort(np.unique([ob[-2:] for ob in obsnames]))
            result = []
            for fi in obsnames_fields:
                for s in ['A','S','C']:
                    try:
                        ind_obs = np.where(np.char.lower(obsnames)==('kmt'+s+fi).lower())[0][0]
                        result.append(obsnames[ind_obs]) 
                    except IndexError:
                        continue
        except:
            result = obsnames
        if len(result) != len(obsnames):
            result = obsnames
        return result

    def set_up(self, **kwargs):
        ''' set up easyAnomalyFinder '''
        self._set_value(kwargs, 'verbose', False)
        self._set_value(kwargs, 'mode', 'detection')

        # self._set_value(kwargs, 'time_windows', np.logspace(np.log10(0.2),np.log10(200),30))
        # self._set_value(kwargs, 'time_windows', np.logspace(np.log10(0.02),np.log10(2000),51))
        self._set_value(kwargs, 'time_windows', np.unique(np.hstack((np.logspace(np.log10(0.02),np.log10(2),16),np.logspace(np.log10(2),np.log10(2000),21)))))
        self._set_value(kwargs, 'window_npt_thres', 3)
        self._set_value(kwargs, 'coverage_min_frac', 0.5)
        self._set_value(kwargs, 'time_step', 1/5)

        ### rescale errors? ###
        self._set_value(kwargs, 'errfac_min', 1.0)
        self._set_value(kwargs, 'errfac_max', 3.0)
        self._set_value(kwargs, 'if_rescale_error_by_fwhm', False)
        if self.if_rescale_error_by_fwhm:
            self.rescale_error_by_fwhm(verbose=self.verbose)
        self._set_value(kwargs, 'if_rescale_error_by_sky', False)
        if self.if_rescale_error_by_sky:
            self.rescale_error_by_sky(verbose=self.verbose)
        
        from scipy.stats import norm as gaussian_distribution
        self._set_value(kwargs, 'sigma', 4)
        self.survival_prob = 2*gaussian_distribution.sf(self.sigma)
        self._set_value(kwargs, 'chi2_low', 80.0)

        self._set_value(kwargs, 'if_check_RMS', True)
        if self.if_check_RMS:
            self.RMS = self.cal_RMS((self.flux-self.flux_model)/np.sqrt(self.flux),sigma_clip=10,iterate=False)
        self._set_value(kwargs, 'RMS_ratio_sigma', 1.0)
        self.RMS_ratio_survival_prob = 2*gaussian_distribution.sf(self.RMS_ratio_sigma)
        # self._set_value(kwargs, 'RMS_ratio_thres', 1.0)

        self._set_value(kwargs, 'if_check_multiple_sites', True)

        self._set_value(kwargs, 'if_check_continuous', False)
        self._set_value(kwargs, 'continuous_check_length', 3)
        self._set_value(kwargs, 'continuous_sigma_thres', 1)
        
        self._set_value(kwargs, 'if_check_chi2_domination', True)
        self._set_value(kwargs, 'if_check_fwhm_domination', False)
        self._set_value(kwargs, 'if_check_sky_domination', False)
        self._set_value(kwargs, 'drop_frac', 0.2)
        self._set_value(kwargs, 'chi2_drop_frac', self.drop_frac)
        self._set_value(kwargs, 'fwhm_drop_frac', self.drop_frac)
        self._set_value(kwargs, 'sky_drop_frac', self.drop_frac)
            
        self._set_value(kwargs, 'if_check_smooth_poly', True)
        self._set_value(kwargs, 'poly_consistent_thres', 0.3)

        self._set_value(kwargs, 'dchi2_ref_low', self.chi2_low)
        #(chi2_poly-window_npt)/(chi2_window-window_npt)<poly_consistent_thres will survive
        

    def cal_poly_correlation(self, x, y, deg=6):
        '''
        Calculate significance of the correlation, using polynomial fits.
        All polynomial coefficients equal to 0 means no correlation.
        '''
        poly_c,poly_cov = np.polyfit(x,y,deg=deg,cov='unscaled')
        poly_cov_inv = np.linalg.inv(poly_cov)
        significance = mahalanobis(poly_c,np.zeros_like(poly_c),poly_cov_inv)
        return significance


    def rescale_error_by_fwhm(self, dfwhm=0.2, verbose=False):
        if verbose:
            print('Rescale errors by FWHM')
        for iob,ob in enumerate(self.obsnames):
            obs_filter = (self.sitefield==ob)
            fwhm_obs = self.fwhm[obs_filter]
            chi2_obs = self.chi2[obs_filter]
            fwhm_range = [np.min(fwhm_obs)-dfwhm/2, np.max(fwhm_obs)+dfwhm/2]
            chi_obs = self.chi[obs_filter]
            chi2_obs = self.chi2[obs_filter]
            ferr_obs = self.ferr[obs_filter]

            # # check the fwhm correlation
            # significance_fwhm_corr = self.cal_poly_correlation(fwhm_obs,chi_obs,deg=6)
            # print('  Seeing corr. of %s: %.3f sigma'%(ob, significance_fwhm_corr))

            for ifwhm in np.arange(*fwhm_range,dfwhm):
                fwhm_filter = (fwhm_obs>ifwhm) & (fwhm_obs<=ifwhm+dfwhm)
                npt = np.sum(fwhm_filter)
                if npt<=0: continue
                errfac = np.sqrt(np.sum(chi2_obs[fwhm_filter])/npt)
                errfac = min(errfac,self.errfac_max)
                errfac = max(errfac,self.errfac_min)

                ### rescale ferr, chi2, res, ... ###
                chi_obs[fwhm_filter]  /= (errfac)
                chi2_obs[fwhm_filter] /= (errfac*errfac)
                ferr_obs[fwhm_filter] *= (errfac)
                if verbose and (errfac!=1.0):
                    print('  %s - fwhm range: (%.2f,%.2f], npt: %4d, errfac: %.3f'%(ob,ifwhm,ifwhm+dfwhm,npt,errfac))
            self.chi[obs_filter] = chi_obs
            self.chi2[obs_filter] = chi2_obs
            self.ferr[obs_filter] = ferr_obs

            # # check the fwhm correlation
            # significance_fwhm_corr = self.cal_poly_correlation(fwhm_obs,chi_obs,deg=6)
            # print('  Seeing corr. of %s: %.3f sigma'%(ob, significance_fwhm_corr))


    def rescale_error_by_sky(self, dsky=0.2, verbose=False):
        if verbose:
            print('Rescale errors by sky background')
        for iob,ob in enumerate(self.obsnames):
            obs_filter = (self.sitefield==ob)
            sky_obs  = np.log10(self.sky[obs_filter])
            chi2_obs = self.chi2[obs_filter]
            sky_range = [np.min(sky_obs)-dsky/2, np.max(sky_obs)+dsky/2]

            chi_obs = self.chi[obs_filter]
            chi2_obs = self.chi2[obs_filter]
            ferr_obs = self.ferr[obs_filter]
            for isky in np.arange(*sky_range,dsky):
                sky_filter = (sky_obs>isky) & (sky_obs<=isky+dsky)
                npt = np.sum(sky_filter)
                if npt<=0: continue
                errfac = np.sqrt(np.sum(chi2_obs[sky_filter])/npt)
                errfac = min(errfac,self.errfac_max)
                errfac = max(errfac,self.errfac_min)

                ### rescale ferr, chi2, res, ... ###
                chi_obs[sky_filter]  /= (errfac)
                chi2_obs[sky_filter] /= (errfac*errfac)
                ferr_obs[sky_filter] *= (errfac)
                if verbose and (errfac!=1.0):
                    print('  %s - log sky range: (%.2f,%.2f], npt: %4d, errfac: %.3f'%(ob,isky,isky+dsky,npt,errfac))
            self.chi[obs_filter] = chi_obs
            self.chi2[obs_filter] = chi2_obs
            self.ferr[obs_filter] = ferr_obs
        

    def find_chi2_thres(self, dof, survival_prob=None):
        ''' find chi2 thresholds, use Hash table to speed up '''
        if survival_prob is None:
            survival_prob = self.survival_prob
        try:
            chi2_thres = self.chi2_thres_dict[dof]
        except KeyError:
            chi2_thres = chi2_distribution.isf(survival_prob,df=dof)
            self.chi2_thres_dict[dof] = chi2_thres
        return chi2_thres

    def find_chi_thres(self, dof, survival_prob=None):
        ''' find chi thresholds (for RMS), use Hash table to speed up '''
        if survival_prob is None:
            survival_prob = self.RMS_ratio_survival_prob
        try:
            chi_thres = self.chi_thres_dict[dof]
        except KeyError:
            chi_thres = chi_distribution.isf(survival_prob,df=dof)
            self.chi_thres_dict[dof] = chi_thres
        return chi_thres
    
    def continuous_check(self, sign_array):
        ''' check if the given sign array are all the same '''
        if abs(np.sum(sign_array)) == len(sign_array):
            return True
        else:
            return False

    def poly_fit(self,idate,iflux,iferr,order=5,retuen_callable=False):
        ''' fit polynomial '''
        date_min,date_max = idate[0],idate[-1]
        date_norm = (idate-date_min)/(date_max-date_min)*2-1 # normalize to [-1,1]
        poly_coeff = np.polyfit(date_norm,iflux,deg=order,w=1/iferr)
        flux_poly_model = np.poly1d(poly_coeff)(date_norm)
        if retuen_callable:
            poly_fit_func = lambda d: np.poly1d(poly_coeff)((d-date_min)/(date_max-date_min)*2-1)
            return poly_coeff,flux_poly_model,poly_fit_func
        return poly_coeff,flux_poly_model

    def is_window_valid(self, window_npt):
        ''' check if the window has enough points '''
        return (window_npt >= self.window_npt_thres)

    def is_chi2_pass_threshold(self, ichi2_window, window_npt=None):
        ''' check if dchi2 passes the threshold '''
        chi2_window = np.sum(ichi2_window)
        if window_npt is None:
            window_npt = len(ichi2_window)
        chi2_thres = self.find_chi2_thres(window_npt,survival_prob=self.survival_prob)
        return (chi2_window>=chi2_thres) & ((chi2_window-window_npt)>=self.chi2_low)

    def cal_med_std(self,array):
        med = np.nanmedian(array) 
        std = np.nanpercentile(array,(15.87,84.13))
        std = 0.5*(std[1]-std[0])
        return med,std

    def cal_mean_std(self,array):
        mean = np.nanmean(array)
        std = np.nanstd(array)
        return mean,std
    
    def clip_outlier(self,array,med,std,sigma_clip=4):
        good = (np.abs(array - med) <= sigma_clip*std)
        good[np.isnan(good)] = False
        return good,array[good],np.sum(good)

    def clip_outlier_iter(self,array,med_ini=None,std_ini=None,sigma_clip=4,maxiter=100):
        ngood = len(array)
        array_new = np.copy(array)
        good = np.ones_like(array,dtype='bool')
        if (med_ini is None) or (std_ini is None):
            med, std = self.cal_med_std(array=array)
        for i in range(maxiter):
            good,_,_ = self.clip_outlier(array_new,med,std,sigma_clip=sigma_clip)
            if np.sum(good) ==  ngood:
                break
            med, std = self.cal_med_std(array=array[good])
            ngood = np.sum(good)
        return array[good]

    def cal_RMS(self,iflux_res,iterate=True,sigma_clip=5):
        if iterate:
            iflux_res_clean = self.clip_outlier_iter(iflux_res,sigma_clip=sigma_clip,maxiter=1000)
            med, std = self.cal_med_std(array=iflux_res_clean)
            RMS = np.sqrt(np.nanmean((iflux_res_clean-med)**2))
        else:
            mean, std = self.cal_mean_std(array=iflux_res)
            RMS = np.sqrt(np.nanmean((iflux_res-mean)**2))
        return RMS

    def is_significant_RMS(self, RMS_ratio, window_npt):
        chi_thres = self.find_chi_thres(window_npt,survival_prob=self.RMS_ratio_survival_prob)
        return (np.sqrt(window_npt) * RMS_ratio > chi_thres)

    def is_multiple_sites(self, isite_window):
        return (len(np.unique(isite_window))>1)
    
    def is_duty_full(self,t_window,idate_window):
        longest_time_gap = np.max(idate_window[1:]-idate_window[:-1])
        valid = (1.0-longest_time_gap/t_window) >= self.coverage_min_frac
        return valid
    
    def is_continuous_significant(self, ires_window, n_continuous=None, thres=None, window_npt=None):
        # check if there are continuous points with significant chi2 with the same sign
        if window_npt is None:
            window_npt = len(ires_window)
        n_continuous = self.continuous_check_length if n_continuous is None else n_continuous
        thres = self.continuous_sigma_thres if thres is None else thres
        continuous = False

        abs_ires = np.abs(ires_window)
        
        for i in range(window_npt-n_continuous+1):
            segment = ires_window[i:i+n_continuous]
            # check if the absolute residuals are all above the threshold
            if np.all(abs_ires[i:i+n_continuous] > thres):
                # check if the signs are all the same
                if np.all(segment > 0) or np.all(segment < 0):
                    continuous = True
                    break
        return continuous

    def is_continuous(self, ichi2_window, ires_sign_window, window_npt=None):
        # The max-SNR point should have the same sign with surrounding points
        if window_npt is None:
            window_npt = len(ichi2_window)
        continuous_check_index_start = np.argmax(ichi2_window)-self.continuous_check_length+1
        continuous = False
        for i in range(continuous_check_index_start,continuous_check_index_start+self.continuous_check_length):
            if i<0 or i+self.continuous_check_length>window_npt:
                continue
            continuous_check_indices = ires_sign_window[i:i+self.continuous_check_length]
            continuous = continuous or self.continuous_check(continuous_check_indices)
            if continuous:
                break
        return continuous

    def is_dominate_by_largest_chi2(self, ichi2_window, window_npt=None):
        if window_npt is None:
            window_npt = len(ichi2_window)
        ndrop = int(np.ceil(window_npt*self.chi2_drop_frac))
        nremain = window_npt-ndrop
        ichi2_window_remain_ind = np.argpartition(ichi2_window, nremain)[:nremain]
        ichi2_window_dropped = ichi2_window[ichi2_window_remain_ind]
        chi2_window_dropped = np.sum(ichi2_window_dropped)
        chi2_thres_dropped = self.find_chi2_thres(nremain)
        return (chi2_window_dropped>=chi2_thres_dropped)

    def is_dominate_by_worst_fwhm(self, ichi2_window, ifwhm_window, window_npt=None):
        if window_npt is None:
            window_npt = len(ichi2_window)
        ndrop = int(np.ceil(window_npt*self.fwhm_drop_frac))
        nremain = window_npt-ndrop
        ichi2_window_remain_ind = np.argpartition(ifwhm_window, nremain)[:nremain]
        ichi2_window_dropped = ichi2_window[ichi2_window_remain_ind]
        chi2_window_dropped = np.sum(ichi2_window_dropped)
        chi2_thres_dropped = self.find_chi2_thres(nremain)
        return (chi2_window_dropped>=chi2_thres_dropped)

    def is_dominate_by_highest_sky(self, ichi2_window, isky_window, window_npt=None):
        if window_npt is None:
            window_npt = len(ichi2_window)
        ndrop = int(np.ceil(window_npt*self.sky_drop_frac))
        nremain = window_npt-ndrop
        ichi2_window_remain_ind = np.argpartition(isky_window, -nremain)[-nremain:]
        ichi2_window_dropped = ichi2_window[ichi2_window_remain_ind]
        chi2_window_dropped = np.sum(ichi2_window_dropped)
        chi2_thres_dropped = self.find_chi2_thres(nremain)
        return (chi2_window_dropped>=chi2_thres_dropped)

    def is_consistent_poly(self, idate_window, iflux_res_window, iferr_window, chi2_window, order=None, window_npt=None, return_chi2_poly=False):
        if window_npt is None:
            window_npt = len(idate_window)
        # polynomial fit
        if order is None:
            poly_order = max(2,min(window_npt//6,8))
            # poly_order = max(2,min(window_npt//3,10))
        else:
            poly_order = order
        self.__poly_order = poly_order
        if window_npt<=5:
            if return_chi2_poly:
                return True, 0.0
            return True
        poly_coeff,flux_poly_model = self.poly_fit(idate_window,iflux_res_window,iferr_window,order=poly_order)
        chi2_poly = np.sum( ((flux_poly_model-iflux_res_window)/iferr_window)**2 )
        chi2_poly_frac = (chi2_poly-window_npt)/(chi2_window-window_npt)
        if return_chi2_poly:
            return (chi2_poly_frac<=self.poly_consistent_thres), chi2_poly
        return (chi2_poly_frac<=self.poly_consistent_thres)

    def find_anomaly(self, verbose=True, sort_anomaly_list=True):
        for arg in [self.mode,self.window_npt_thres,self.sigma,self.survival_prob,self.time_windows,self.time_step,self.chi2_drop_frac,self.fwhm_drop_frac,self.sky_drop_frac,self.poly_consistent_thres]:
            if arg is None:
                print('AnomalyFinder has not been set up, please run AnomalyFinder.set_up() first.')
                return None

        nsearch,nvalid,ncover,nanom = 0,0,0,0
        n_pass_chi2,n_pass_RMS,n_pass_mutisite,n_pass_continuous,n_pass_dropchi2,n_pass_dropfwhm,n_pass_dropsky,n_pass_poly = 0,0,0,0,0,0,0,0
        # if verbose:
        #     print('Event RMS: %.6f'%self.RMS)

        anomaly_list = []
        if self.mode =='all':
            afall_list = []
            fmt = (['%.6f %10.6f %6d %6d %6d %12.6f %12.6f']+
                   (['%1d'] if self.if_check_multiple_sites else []) +
                   (['%1d'] if self.if_check_continuous else []) +
                   (['%1d'] if self.if_check_chi2_domination else []) + 
                   (['%1d'] if self.if_check_fwhm_domination else []) +
                   (['%1d'] if self.if_check_sky_domination else []) + 
                   ['%12.6f %1d'])
            fmt = ' '.join(fmt)
            # t_start, t_window, window_npt, chi2_window, RMS_ratio, b_is_multiple_sites, b_is_continuous, b_is_dominate_by_largest_chi2, b_is_dominate_by_worst_fwhm, b_is_dominate_by_highest_sky, chi2_poly
        for t_window in self.time_windows:
            for t_start in np.arange(self.date[0],self.date[-1],t_window*self.time_step):
                nsearch += 1 # number of searched time windows

                ### check if the window is valid (have enouht data points) ###
                index_start,index_end = np.searchsorted(self.date,[t_start,t_start+t_window])
                window_npt = index_end-index_start
                if not(self.is_window_valid(window_npt)):
                    # if self.mode == 'all':
                    #     l = ([t_start, t_window, index_start, index_end, window_npt, 0.0, 1.0] + 
                    #          ([False] if self.if_check_multiple_sites else []) +
                    #          ([False] if self.if_check_continuous else []) +
                    #          ([False] if self.if_check_chi2_domination else []) + 
                    #          ([False] if self.if_check_fwhm_domination else []) +
                    #          ([False] if self.if_check_sky_domination else []) + 
                    #          [np.inf, False])
                    #     if verbose: print(fmt%tuple(l))
                    #     afall_list.append(l)
                    continue
                nvalid += 1

                ### Read the indices ###
                window_indices = np.arange(index_start,index_end)
                idate_window = self.date[window_indices]
                ires_window  = self.chi[window_indices]
                ires_sign_window = self.res_sign[window_indices]
                ichi2_window = self.chi2[window_indices]
                iflux_window = self.flux[window_indices]
                iferr_window = self.ferr[window_indices]
                ifwhm_window = self.fwhm[window_indices]
                isky_window  = self.sky[window_indices]
                # iamp_model_window = self.amp_model[window_indices]
                iflux_model_window = self.flux_model[window_indices]
                isite_window = self.site[window_indices]
                ichi2ref_window = self.chi2_ref[window_indices]

                ### check duty ###
                # # print(t_start,t_window,self.is_duty_full(t_window, idate_window))
                if not(self.is_duty_full(t_window, idate_window)) and (self.mode!='all'):
                    continue
                ncover += 1

                chi2_window = np.sum(ichi2_window)
                ## if in sensitivity mode, also compare chi2 with the reference model ##
                dchi2_ref = chi2_window - np.sum(ichi2ref_window)
                if (self.mode == 'sensitivity') and (dchi2_ref<self.dchi2_ref_low):
                    continue
                ### compare chi2_window ###
                if not(self.is_chi2_pass_threshold(ichi2_window, window_npt)) and (self.mode!='all'):
                    continue
                n_pass_chi2 += 1

                ### check apparent magnification, dF/F ###
                iflux_res_window = (iflux_window-iflux_model_window)
                idflux_window = iflux_res_window/iflux_model_window
                dflux_max = idflux_window[np.argmax(np.abs(idflux_window))]

                ### check variability ###
                RMS_ratio = -np.inf
                if self.if_check_RMS:
                    RMS_window = self.cal_RMS(iflux_res_window/np.sqrt(iflux_window),iterate=False)
                    RMS_ratio = RMS_window/self.RMS
                    if not(self.is_significant_RMS(RMS_ratio,window_npt=window_npt)) and (self.mode!='all'):
                        continue
                    n_pass_RMS += 1
                    # print(window_npt,RMS_ratio,self.find_chi_thres(window_npt)/np.sqrt(window_npt))

                ### check if have multiple sites data ###
                if self.if_check_multiple_sites:
                    b_is_multiple_sites = self.is_multiple_sites(isite_window)
                    if not(b_is_multiple_sites) and (self.mode!='all'):
                        continue
                    n_pass_mutisite += 1

                ### check if the sign of residuals are continuous and exceed a threshold ###
                if self.if_check_continuous:
                    b_is_continuous = self.is_continuous_significant(ires_window)
                    if not(b_is_continuous) and (self.mode!='all'):
                        continue
                    n_pass_continuous += 1

                ### check if dominated by a few largest chi2 points ###
                if self.if_check_chi2_domination:
                    b_is_dominate_by_largest_chi2 = self.is_dominate_by_largest_chi2(ichi2_window, window_npt)
                    if not(b_is_dominate_by_largest_chi2) and (self.mode!='all'):
                        continue
                    n_pass_dropchi2 += 1

                ### check if dominated by a few worst seeing points ###
                if self.if_check_fwhm_domination:
                    b_is_dominate_by_worst_fwhm = self.is_dominate_by_worst_fwhm(ichi2_window, ifwhm_window, window_npt)
                    if not(b_is_dominate_by_worst_fwhm) and (self.mode!='all'):
                        continue
                    n_pass_dropfwhm += 1

                ### check if dominated by a few highest skybg points ###
                if self.if_check_sky_domination:
                    b_is_dominate_by_highest_sky = self.is_dominate_by_highest_sky(ichi2_window, isky_window,window_npt)
                    if not(b_is_dominate_by_highest_sky) and (self.mode!='all'):
                        continue
                    n_pass_dropsky += 1

                ### check if smooth (by fitting polynomial) ###
                # print(self.is_consistent_poly(idate_window, iflux_window, iferr_window, chi2_window, return_chi2_poly=True))
                if self.if_check_smooth_poly:
                    consistent_poly_result = self.is_consistent_poly(idate_window, iflux_res_window, iferr_window, chi2_window, return_chi2_poly=True)
                    consistent_poly, chi2_poly = consistent_poly_result
                    if not(consistent_poly) and (self.mode!='all'):
                        continue
                    n_pass_poly += 1
                else:
                    chi2_poly = chi2_window
                    self.__poly_order = 0

                z_over_sigma = np.sum(ires_window)/np.sqrt(window_npt)

                nanom += 1
                if (self.mode=='all'):
                    l = ([t_start, t_window, 
                          index_start, index_end,
                          window_npt, chi2_window, 
                          RMS_ratio] +
                         ([b_is_multiple_sites] if self.if_check_multiple_sites else []) +
                         ([b_is_continuous] if self.if_check_continuous else []) +
                         ([b_is_dominate_by_largest_chi2] if self.if_check_chi2_domination else []) + 
                         ([b_is_dominate_by_worst_fwhm] if self.if_check_fwhm_domination else []) +
                         ([b_is_dominate_by_highest_sky] if self.if_check_sky_domination else []) + 
                         [chi2_poly, consistent_poly])
                    if verbose: print(fmt%tuple(l))
                    afall_list.append(l)
                    continue
                anomaly_list.append([t_start,t_window,
                                     index_start,index_end,
                                     window_npt,chi2_window-window_npt,
                                     chi2_window/window_npt,
                                     chi2_poly,self.__poly_order,
                                     z_over_sigma,
                                     dflux_max,RMS_ratio])
                if self.mode == 'sensitivity':
                    self.anomaly_list = anomaly_list
                    return anomaly_list
        if (self.mode=='all'):
            afall_list = np.array(afall_list,dtype='O')
            return afall_list

        anomaly_list = np.array(anomaly_list,dtype='O')
        if sort_anomaly_list and nanom>1:
            anomaly_list = anomaly_list[np.argsort(anomaly_list[:,5])[::-1]]
            nanom_grp = nanom
            nanom_grp_tmp = 0
            while nanom_grp != nanom_grp_tmp:
                anomaly_list = self.group_anomaly(anomaly_list=anomaly_list)
                nanom_grp,nanom_grp_tmp = len(anomaly_list),nanom_grp
            anomaly_list = anomaly_list[np.argsort(anomaly_list[:,5])[::-1]]
        self.anomaly_list = anomaly_list
        nanom_grp = len(anomaly_list)

        if verbose:
            print('All windows: %24d'%nsearch)
            print('Windows have >=3 pt: %16d'%nvalid)
            print('Enough coverage: %20d'%ncover)
            print('Passed chi2 threshold: %14d'%n_pass_chi2)
            if self.if_check_RMS:
                print('Passed variability check: %11d'%n_pass_RMS)
            if self.if_check_multiple_sites:
                print('Multiple sites: %21d'%n_pass_mutisite)
            if self.if_check_continuous:
                print('Continuous significant points: %6d'%n_pass_continuous)
            if self.if_check_chi2_domination:
                print('Not dominated by largest chi2: %6d'%n_pass_dropchi2)
            if self.if_check_fwhm_domination:
                print('Not dominated by worst seeing: %6d'%n_pass_dropfwhm)
            if self.if_check_sky_domination:
                print('Not dominated by highest sky: %7d'%n_pass_dropsky)
            if self.if_check_smooth_poly:
                print('Smooth (using poly-fit): %12d'%n_pass_poly)
            print('All anomaly: %24d'%nanom)
            print('Grouped anomaly: %20d'%nanom_grp)
            if nanom>0:
                print('\nt_start t_window ind_start ind_end window_npt chi2_window-window_npt chi2_window/window_npt chi2_poly poly_order z/σz max(dF/F) RMS_ratio')
                for i,anom in enumerate(anomaly_list.tolist()):
                    # print('%4d '%i,end='',flush=True)
                    # print(anom)
                    print('%4d %.6f %10.6f %6d %6d %4d %12.6f %10.6f %12.6f %1d %10.6f %10.6f %10.6f'%(i,*anom))
        return anomaly_list

    def group_anomaly(self, anomaly_list=None):
        if anomaly_list is None:
            anomaly_list = self.anomaly_list
        _, uni_ind = np.unique(anomaly_list[:,2:4].astype('int'),axis=0,return_index=True)
        anomaly_list = anomaly_list[np.argsort(uni_ind,axis=0)]

        anomaly_npt = anomaly_list[:,4]
        arg_sortbyn = np.argsort(anomaly_npt)
        anomaly_list_sortbyn = anomaly_list[arg_sortbyn]
        anomaly_list_sortbyn_ind0 = anomaly_list_sortbyn[:,2]
        anomaly_list_sortbyn_ind1 = anomaly_list_sortbyn[:,3]
        anomaly_list_sortbyn_dchi2 = anomaly_list_sortbyn[:,5]
        anomaly_list_sortbyn_dchi2_pp = anomaly_list_sortbyn[:,6]
        anomaly_list_grp_ind = []
        anomaly_list_ind = list(range(len(anomaly_list_sortbyn)))
        ### make groups ###
        for i,anom in enumerate(anomaly_list_sortbyn):
            assigned = any(i in grp for grp in anomaly_list_grp_ind)
            if assigned:
                continue

            ind0, ind1, window_npt, dchi2_window, dchi2_window_pp = anom[2:7]
            for igrp,grp in enumerate(anomaly_list_grp_ind):
                grp_ind = grp[0]
                isincluded = (ind0<=anomaly_list_sortbyn_ind0[grp_ind]) & (ind1>=anomaly_list_sortbyn_ind1[grp_ind])
                if isincluded:
                    anomaly_list_grp_ind[igrp].append(i)
                    assigned = True
                    break

            # if not assigned, create a new group
            if not assigned:
                anomaly_list_grp_ind.append([i])

        ### preserve both the maximum dchi2 and maximum one in each group ###
        anomaly_list_grp = []
        for igrp,grp in enumerate(anomaly_list_grp_ind):
            grp_filter = np.array(grp)
            ind_max_dchi2 = np.argmax(anomaly_list_sortbyn_dchi2[grp_filter])
            ind_max_dchi2_pp = np.argmax(anomaly_list_sortbyn_dchi2_pp[grp_filter])

            anomaly_list_grp.append(anomaly_list_sortbyn[grp_filter][ind_max_dchi2])
            # if ind_max_dchi2 != ind_max_dchi2_pp:
            #     anomaly_list_grp.append(anomaly_list_sortbyn[grp_filter][ind_max_dchi2_pp])
        anomaly_list_grp = np.array(anomaly_list_grp)
        return anomaly_list_grp

    ### functions to plot the anomaly ###
    def __init_plot(self,):
        from matplotlib import gridspec
        from scipy.interpolate import interp1d
        self.interp1d = interp1d
        plt.rcParams['font.family']='serif'
        plt.rcParams['mathtext.fontset']='stix'
        plt.rcParams['lines.linewidth'] = 1
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['axes.labelsize'] = 16
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.right'] = True
        plt.rcParams['axes.spines.bottom']=True
        plt.rcParams['axes.spines.top']=True
        plt.rcParams['axes.spines.left']=True
        plt.rcParams['axes.spines.right']=True
        plt.rcParams['xtick.bottom']=True
        plt.rcParams['xtick.labelbottom']=True
        plt.rcParams['ytick.left']=True
        plt.rcParams['ytick.labelleft']=True
        plt.rcParams['font.size'] = 18.
        plt.rcParams['xtick.major.size'] = 10.
        plt.rcParams['xtick.minor.size'] = plt.rcParams['xtick.major.size'] / 2.
        plt.rcParams['ytick.major.size'] = 10.
        plt.rcParams['ytick.minor.size'] = plt.rcParams['ytick.major.size'] / 2.

        if not self.cstrs:
            self.get_kmt_obs_color(self.obsnames)

        self.fig = plt.figure(figsize=(7,9))
        gs0 = gridspec.GridSpec(2,1,height_ratios=[1.5,7],hspace=0.08)
        gs_full = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=gs0[0,0],
                                              height_ratios=[1,0.5],hspace=0.04)
        gs = gridspec.GridSpecFromSubplotSpec(5,1,subplot_spec=gs0[1,0],
                                              height_ratios=[3,1,1,1,1],hspace=0.04)

        self.ax0 = plt.subplot(gs_full[0,0]) # full light curve
        self.ax1 = plt.subplot(gs_full[1,0],sharex=self.ax0) # full light curve
        self.ax2 = plt.subplot(gs[0,0])
        self.ax3 = plt.subplot(gs[1,0],sharex=self.ax2)
        self.ax4 = plt.subplot(gs[2,0],sharex=self.ax2)
        self.ax5 = plt.subplot(gs[3,0],sharex=self.ax2)
        self.ax6 = plt.subplot(gs[4,0],sharex=self.ax2)
        self.axes = [self.ax0,self.ax1,self.ax2,self.ax3,self.ax4,self.ax5,self.ax6]

        for iob,ob in enumerate(self.obsnames):
            obs_filter = (self.sitefield==ob)
            self.ax0.errorbar(self.date[obs_filter],
                              self.flux[obs_filter],
                              self.ferr[obs_filter],
                              c=self.cstrs[iob],
                              linestyle='none',marker='o',ms=2,mfc='none',capsize=1,
                              zorder=1,
                              label=ob.upper())
            self.ax1.errorbar(self.date[obs_filter],
                              self.flux[obs_filter]-self.flux_model[obs_filter],
                              self.ferr[obs_filter],
                              c=self.cstrs[iob],
                              linestyle='none',marker='o',ms=2,mfc='none',capsize=1,
                              zorder=1,
                              label=ob.upper())
        self.ax0.plot(self.date,self.flux_model,c='k')
        self.ax1.axhline(c='k')
        plt.setp( self.ax0.get_xticklabels(), visible=False)
        self.ax0.legend(fontsize=7,ncol=int((len(self.obsnames)-1)/3)+1,
                        bbox_to_anchor=(0.15, 1.04, 0.7, 0.2), loc="lower left",
                        mode="expand", borderaxespad=0)
        # self.ax0.set_title(self.eventname,fontsize=10,
                        #    x=0.5,y=1.55)
        self.has_init_plot = True

    def get_kmt_obs_color(self,obsnames):
        #global cstrs_kmta,cstrs_kmtc,cstrs_kmts
        if len(obsnames)==1:
            self.cstrs = ['#dd0000']
            return
        for iob,ob in enumerate(obsnames):
            if ob[:4].lower() == 'kmta':
                self.cstrs.append(self.cstrs_kmta.pop(0))
            elif ob[:4].lower() == 'kmtc':
                self.cstrs.append(self.cstrs_kmtc.pop(0))
            elif ob[:4].lower() == 'kmts':
                self.cstrs.append(self.cstrs_kmts.pop(0))
            elif ob.lower() == 'ogle':
                self.cstrs.append('#bb00bb')
            elif ob.lower() == 'moa':
                self.cstrs.append('#888888')
            else:
                print('%s not recognized, use random color.'%(ob))
                self.cstrs.append( hex(np.random.randint(2**(8*3))).replace('0x','#') )

    def plot_anomaly(self,ianom,saveto=None,anom_info=None):
        if anom_info is None:
            anom = self.anomaly_list[ianom]
        else:
            anom = anom_info
            # t_start,t_window,poly_ind0,poly_ind1,window_npt,dchi2_window,dchi2_window_pp,chi2_poly,poly_order,z_over_sigma,dflux_max,RMS_ratio
        if not(self.has_init_plot):
            self.__init_plot()
        fig = self.fig

        ### read anomaly parameters ###
        t_start,t_window,poly_ind0,poly_ind1,window_npt,dchi2_window,dchi2_window_pp,chi2_poly,poly_order,z_over_sigma,dflux_max,RMS_ratio = anom

        ### find plot range ###
        t_plot_range = max(t_window,1.2)
        ind0,ind1 = np.searchsorted(self.date,[t_start-t_plot_range,t_start+t_window+t_plot_range])
        date_anom = self.date[ind0:ind1]
        flux_anom = self.flux[ind0:ind1]
        ferr_anom = self.ferr[ind0:ind1]
        flux_model_anom = self.flux_model[ind0:ind1]
        chi2_anom = self.chi2[ind0:ind1]
        res_anom = self.chi[ind0:ind1]
        fwhm_anom = self.fwhm[ind0:ind1]
        sky_anom  = np.log10(self.sky[ind0:ind1])
        site_anom = self.sitefield[ind0:ind1]
        mag_anom = 18-2.5*np.log10(flux_anom)
        merr_anom = 2.5/np.log(10)/flux_anom * ferr_anom
        mag_model_anom = 18-2.5*np.log10(flux_model_anom)

        dchi2_window = np.sum(self.chi2[poly_ind0:poly_ind1])-(poly_ind1-poly_ind0)

        flux_min,flux_max = np.min(flux_anom-1*ferr_anom),np.max(flux_anom+1*ferr_anom)
        flux_median = np.average(flux_anom,weights=1/ferr_anom**2)
        dflux = flux_max-flux_min

        ### find polynomial model ###
        date_poly = self.date[poly_ind0:poly_ind1]
        flux_poly = self.flux[poly_ind0:poly_ind1]
        ferr_poly = self.ferr[poly_ind0:poly_ind1]
        flux_model_poly = self.flux_model[poly_ind0:poly_ind1]
        flux_res_poly = flux_poly-flux_model_poly
        poly_coeff,flux_poly_model,poly_fit_func = self.poly_fit(date_poly,
            flux_res_poly,ferr_poly,order=poly_order,retuen_callable=True)
        chi2_poly = np.sum( ((flux_poly_model-flux_res_poly)/ferr_poly)**2 )
        try:
            fmodel_func = self.interp1d(date_poly,flux_model_poly,
                                        kind='cubic',fill_value="extrapolate")
        except ValueError:
            fmodel_func = self.interp1d(date_poly,flux_model_poly,
                                        kind='linear',fill_value="extrapolate")


        ### clean previous anomaly figures ###
        for ax in [self.ax2,self.ax3,self.ax4,self.ax5,self.ax6]:
            ax.clear()
        if hasattr(self, 'highlight0') or hasattr(self, 'highlight1'):
            self.highlight0.remove()
            self.highlight1.remove()

        ### plot the anomaly region ###
        self.ax2.plot(self.date,self.flux_model,c='k')
        for iob,ob in enumerate(self.obsnames):
            obs_filter = (site_anom==ob)
            self.ax2.errorbar(date_anom[obs_filter],
                              flux_anom[obs_filter],
                              ferr_anom[obs_filter],
                              linestyle='none',c=self.cstrs[iob],
                              marker='o',ms=2,mfc='none',capsize=1,zorder=1)
            self.ax3.errorbar(date_anom[obs_filter],
                              flux_anom[obs_filter]-flux_model_anom[obs_filter],
                              ferr_anom[obs_filter],
                              linestyle='none',c=self.cstrs[iob],
                              marker='o',ms=2,mfc='none',capsize=1,zorder=1)
            self.ax4.scatter(date_anom[obs_filter],(fwhm_anom[obs_filter]),
                             marker='o',s=2,c=self.cstrs[iob])
            self.ax5.scatter(date_anom[obs_filter],(sky_anom[obs_filter]),
                             marker='o',s=2,c=self.cstrs[iob])
            self.ax6.scatter(date_anom[obs_filter],res_anom[obs_filter],
                             marker='o',s=2,c=self.cstrs[iob])
        poly_tmodel = np.linspace(date_poly[0],date_poly[-1],100)
        self.ax2.plot(poly_tmodel,poly_fit_func(poly_tmodel)+fmodel_func(poly_tmodel),c='k',lw=3,alpha=0.3)
        self.ax3.axhline(c='k',linestyle='-')
        self.ax3.plot(poly_tmodel,poly_fit_func(poly_tmodel),c='k',lw=3,alpha=0.3)
        self.ax6.axhline(c='k',linestyle='-')

        lc_yrange = np.array([np.nanpercentile(flux_anom-ferr_anom,1),np.nanpercentile(flux_anom+ferr_anom,99)])
        lc_yrange = 1.0*(lc_yrange[1]-lc_yrange[0])*np.array([-1,1])+lc_yrange
        res_yrange = np.array([np.nanpercentile(flux_anom-flux_model_anom-ferr_anom,1),np.nanpercentile(flux_anom-flux_model_anom+ferr_anom,99)])
        res_yrange = 0.5*(res_yrange[1]-res_yrange[0])*np.array([-1,1])+res_yrange

        ### plot highlights ###
        self.highlight0 = self.ax0.axvspan(date_anom[0],date_anom[-1],color='m',alpha=0.3)
        self.highlight1 = self.ax1.axvspan(date_anom[0],date_anom[-1],color='m',alpha=0.3)
        for ax in [self.ax2,self.ax3,self.ax4,self.ax5,self.ax6]:
            ax.axvspan(t_start,t_start+t_window,
                       fc='m',ec='none',alpha=0.1,
                       zorder=-1)

        ### set plot ranges ###
        for ax in [self.ax2,self.ax3,self.ax4,self.ax5,self.ax6]:
            ax.set_xlim(t_start-t_plot_range,t_start+t_window+t_plot_range)
        # self.ax2.set_ylim(flux_min-0.2*dflux,flux_max+0.2*dflux)
        self.ax2.set_ylim(lc_yrange.tolist())
        self.ax3.set_ylim(res_yrange.tolist())
        # self.ax5.set_yscale('log')

        ### add text ###
        textstr = (r'$\Delta\chi^2_{\rm window}=%.2f(%.2f)$'%(dchi2_window,max(self.find_chi2_thres(window_npt)-window_npt,self.chi2_low))
                  +', ' + r'$N_{\rm window}=%d$'%(window_npt)
                  +', ' + r'$Z/\sigma_Z=%.3f$'%(z_over_sigma)
                  +', ' + r'$|\Delta F/F|=%.3f$'%(dflux_max)
                  +', ' + r'$q_{\rm RMS}=%.3f~(%.3f)$'%(RMS_ratio,self.find_chi_thres(window_npt)/np.sqrt(window_npt)))
        # self.ax2.text(0.96,0.90,textstr,fontsize=10,
        #               transform=self.ax2.transAxes,ha='right',va='top',
        #               bbox=dict(boxstyle='round', facecolor='w', alpha=1))
        self.ax0.set_title(self.eventname+'\n'+textstr,fontsize=10,
                           x=0.5,y=1.55)

        ### set ticks and tick labels ###
        for ax in [self.ax2,self.ax3,self.ax4,self.ax5]:
            plt.setp( ax.get_xticklabels(), visible=False)
        for ax in self.axes:
            ax.minorticks_on()
        self.ax0.set_ylabel('Flux',fontsize=10)
        self.ax2.set_ylabel('Flux',fontsize=10)
        self.ax3.set_ylabel('Residual',fontsize=10)
        # self.ax3.set_ylabel(r'$\sqrt{\chi^2}$',fontsize=12)
        self.ax4.set_ylabel('Seeing (")',fontsize=10)
        self.ax5.set_ylabel('log bkg',fontsize=10)
        self.ax6.set_ylabel(r'$\chi=\Delta F/\sigma_{F}$',fontsize=12)
        self.ax6.set_xlabel('HJD-2450000',fontsize=10)

        fig.align_ylabels()
        plt.subplots_adjust(bottom=0.05)

        ### save/show the figure ###
        if saveto is not None:
            dirs = os.path.split(saveto)[0]
            if dirs:
                os.makedirs(dirs,exist_ok=True)
            plt.savefig(saveto,dpi=120)
        else:
            plt.show()

    def plot_main_lc(self, saveto=None):
        from matplotlib import gridspec
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        if not self.cstrs:
            self.get_kmt_obs_color(self.obsnames)

        mag = 18-2.5*np.log10(self.flux)
        merr = np.abs(2.5/np.log(10)/self.flux * self.ferr)
        mag_model = 18-2.5*np.log10(self.flux_model)
        mag_lim = [min(22,np.nanpercentile(mag+merr,99)),max(13,np.nanpercentile(mag-merr,0.2))]
        mag_lim_range = mag_lim[0]-mag_lim[1]
        mag_lim = [mag_lim[0]+0.2*mag_lim_range,mag_lim[1]-0.2*mag_lim_range]
        res_lim = [min(5,np.nanpercentile(mag+merr-mag_model,99)),max(-5,np.nanpercentile(mag-merr-mag_model,0.2))]
        res_lim_range = res_lim[0]-res_lim[1]
        res_lim = [res_lim[0]+0.2*res_lim_range,res_lim[1]-0.3*res_lim_range]

        fig = plt.figure(figsize=(10,7))
        gs = gridspec.GridSpec(5,1,height_ratios=[3,1,1,1,1],hspace=0.1)
        ax1 = plt.subplot(gs[0,0])
        if saveto is None:
            ax2 = plt.subplot(gs[1,0],sharex=ax1)
            ax3 = plt.subplot(gs[2,0],sharex=ax1)
            ax4 = plt.subplot(gs[3,0],sharex=ax1)
            ax5 = plt.subplot(gs[4,0],sharex=ax1)
        else:
            ax2 = plt.subplot(gs[1,0])
            ax3 = plt.subplot(gs[2,0])
            ax4 = plt.subplot(gs[3,0])
            ax5 = plt.subplot(gs[4,0])

        for iob,ob in enumerate(self.obsnames):
            obs_filter = (self.sitefield==ob)
            ax1.errorbar(self.date[obs_filter],mag[obs_filter],merr[obs_filter],c=self.cstrs[iob],
                         linestyle='none',marker='o',ms=2,mfc='none',capsize=1,zorder=1,label=ob.upper())
            ax2.errorbar(self.date[obs_filter],mag[obs_filter]-mag_model[obs_filter],
                         merr[obs_filter],c=self.cstrs[iob],
                         linestyle='none',marker='o',ms=2,mfc='none',capsize=1,zorder=1,label=ob.upper())
            ax3.scatter(self.date[obs_filter],self.chi[obs_filter],
                        marker='o',s=2,c=self.cstrs[iob])
            ax4.scatter(self.date[obs_filter],self.fwhm[obs_filter],
                        marker='o',s=2,c=self.cstrs[iob])
            ax5.scatter(self.date[obs_filter],self.sky[obs_filter],
                        marker='o',s=2,c=self.cstrs[iob])
        ax1.legend(fontsize=9,ncol=3)
        ax1.plot(self.date,mag_model,c='k')
        ax2.axhline(0,c='k')
        ax3.axhline(0,c='k')
        ax1.set_ylim(*mag_lim)
        ax2.set_ylim(*res_lim)
        ax5.set_yscale('log')

        ax1.set_ylabel(r'$I$ mag')
        ax2.set_ylabel('Residual')
        ax3.set_ylabel(r'$\chi=\Delta F/\sigma$')
        ax4.set_ylabel('FWHM')
        ax5.set_ylabel('sky bkg')
        ax5.set_xlabel('HJD-2450000')
        if saveto is not None:
            for ax in [ax1,ax2,ax3,ax4]:
                ax.xaxis.set_ticklabels([])

        for ax in [ax1,ax2,ax3,ax4,ax5]:
            ax.minorticks_on()
        fig.align_ylabels()

        ### save/show the figure ###
        if saveto is not None:
            dirs = os.path.split(saveto)[0]
            if dirs:
                os.makedirs(dirs,exist_ok=True)
            plt.savefig(saveto,dpi=120)
            plt.clf()
        else:
            plt.show()
            plt.clf()






if __name__=='__main__':
    import sys

    event = 'kb180900'
    for i,arg in enumerate(sys.argv):
        if arg.lower() in ['-e','-event']:
            event = sys.argv[i+1].lower()
    print(' '.join(sys.argv))
    residual_file = 'data_for_test/{0}/{0}_residual.dat'.format(event)
    save_dir = 'data_for_test/anom/{0}'.format(event)
    os.makedirs(save_dir,exist_ok=True)

    ### select the working mode
    # 'detection': find anomalies; 
    # 'sensitivity': return when find first anomaly; 
    # 'all': return parameters for all time windows;
    AFmode = 'detection' 
    
    ### Set up AnomalyFinder: source files, methods, thresholds, ...
    # read data
    residual_data = np.genfromtxt(residual_file,
                                  dtype=[('date','f8'),('chi','f8'),('chi2','f8'),
                                         ('flux','f8'),('ferr','f8'),('fwhm','f8'),
                                         ('sky','f8'),('amp_model','f8'),('flux_model','f8'),
                                         ('site','U10')])
    
    # Actually, only date, flux, ferr, flux_model are necessary:
    #     chi, chi2 can be internally calculated from flux, ferr, flux_model;
    # other keys are optional and related to some specific criterion:
    #     fwhm: used to rescale error by seeing;
    #     sky: used to rescale error by sky background;
    #     site: used to check multiple sites data;
    #     amp_model: currently not used;
    #     chi2_ref: used in sensitivity mode to compare with each chi2_window.
    data_kwargs = {'date':residual_data['date'], 
                   'chi' :residual_data['chi'], 
                   'chi2':residual_data['chi2'],
                   'flux':residual_data['flux'], 
                   'ferr':residual_data['ferr'], 
                   'fwhm':residual_data['fwhm'],
                   'sky' :residual_data['sky'], 
                   'flux_model':residual_data['flux_model'],
                   'site':residual_data['site'], 
                   'chi2_ref':np.zeros_like(residual_data['date'])}
    AF = AnomalyFinder(data_kwargs=data_kwargs,eventname=event)

    # set up AF options
    AF_options = {'sigma': 4.0, 'chi2_low':80.0,
                  'mode':AFmode,
                  'dchi2_ref_low':50.0,
                  'coverage_min_frac': 0.5,
                  'if_rescale_error_by_fwhm':True,
                  'chi2_drop_frac':0.15,
                  'if_check_RMS':True,
                  'RMS_ratio_sigma':1.0,
                  'if_check_multiple_sites':True,
                  'if_check_continuous':True,
                  'continuous_check_length': 3,
                  'continuous_sigma_thres': 1,
                  'if_check_chi2_domination':True,
                  'if_check_smooth_poly':True,
                  'poly_consistent_thres':0.3,
                  'verbose':False,
                  }
    AF.set_up(**AF_options)

    ### Plot the entire light curve ###
    AF.plot_main_lc(saveto=save_dir+'/main_lc.png')
    
    ### Run AnomalyFinder ###
    anomaly_list = AF.find_anomaly(verbose=True)

    ### Save the anomaly list & plot the anomaly figure(s) ###
    header = 't_start t_window ind_start ind_end window_npt chi2_window-window_npt chi2_window/window_npt chi2_poly poly_order z/σz max(dF/F) RMS_ratio'
    if len(anomaly_list)>0:
        np.savetxt(save_dir+'/anomaly_list.dat',anomaly_list,fmt='%.6f %10.6f %6d %6d %4d %12.6f %12.6f %12.6f %1d %12.6f %12.6f %12.6f',header=header)
    for ianom,anom in enumerate(anomaly_list):
        AF.plot_anomaly(ianom=ianom,saveto=save_dir+'/anom_%03d.png'%(ianom))
