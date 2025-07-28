import h5py
from pybird.module import *

def get_dict_from_hdf5(group, none_flag=float("nan")):
    """Recursively read data from an HDF5 group into a Python dictionary.
    
    Args:
        group (h5py.Group): HDF5 group to read from.
        none_flag (float, optional): Value used to represent None in the HDF5 file.
            Defaults to float("nan").
            
    Returns:
        dict: Dictionary containing all datasets and subgroups from the HDF5 group.
    """
    d = {}
    for key in group.keys():
        if isinstance(group[key], h5py.Group):
            d[key] = get_dict_from_hdf5(group[key], none_flag=none_flag)
        elif isinstance(group[key], h5py.Dataset):
            data = group[key][()]
            if array_equal(data, none_flag):
                d[key] = None
            elif data.dtype == np.object_ and isinstance(data[0], bytes):
                d[key] = data.astype(str).tolist()
            else:
                d[key] = data
    return d

def save_dict_to_hdf5(group, data, none_flag=float("nan")):
    """Recursively save a Python dictionary to an HDF5 group.
    
    Args:
        group (h5py.Group): HDF5 group to write to.
        data (dict): Dictionary to save to HDF5.
        none_flag (float, optional): Value to use when saving None values.
            Defaults to float("nan").
    """
    def save_recursive(subgroup, subdata):
        for key, value in subdata.items():
            if isinstance(value, dict):
                save_recursive(subgroup.create_group(key), value)
            elif value is None:
                subgroup.create_dataset(key, data=none_flag)
            else:
                subgroup.create_dataset(key, data=value)

    save_recursive(group, data)

class ReadWrite(object):
    """A class for reading, writing, and formatting cosmological data files.
    
    This class handles I/O operations for cosmological power spectra and correlation
    functions, including reading HDF5 files, formatting data for analysis, and writing
    results to various output formats.
    
    Attributes:
        None (all data is passed through method parameters)
    
    Methods:
        read(): Read data from HDF5 file and format for analysis.
        check(): Validate data against configuration parameters.
        config(): Prepare configuration dictionary for correlator.
        format(): Format raw data for analysis.
        write(): Write results to output files.
        write_common(): Write common cosmological parameters to output.
        write_pk(): Write power spectrum data to output.
        write_cf(): Write correlation function data to output.
        write_bao_rec(): Write BAO reconstruction data to output.
        set_header(): Create header for output files.
        get_wedge_rotation_matrix(): Get matrix for multipole to wedge conversion.
    """
    def __init__(self):
        pass

    def read(self, c, verbose=True):
        data = os.path.join(c['data_path'], c['data_file'])
        if not os.path.isfile(data): raise Exception("%s not found" % data)
        elif verbose: print ('reading data file: %s' % data)
        try: 
            with h5py.File(data, 'r') as hf: d = get_dict_from_hdf5(hf)
        except: 
            d = load(data, allow_pickle='TRUE').item()
        self.check(c, d, verbose=verbose)
        fd_sky = self.format(c, d, verbose=verbose) # skylist of formatted data dict for Likelihood
        fc_sky = self.config(c, fd_sky)             # skylist of formatted config dict for Correlator
        return d, fc_sky, fd_sky

    def check(self, c, d, verbose=True):

        options_in_config = ['with_bao_rec', "with_ap", "with_survey_mask", "with_binning", "with_wedge", "with_redshift_bin", "with_stoch", "with_nnlo_counterterm", "with_loop_prior", "with_emu"]
        for keys in options_in_config:
            if not keys in c: c[keys] = False
        options_in_config_default_true = ['with_resum']
        for keys in options_in_config_default_true:
            if not keys in c: c[keys] = True

        for sky, cut in c['sky'].items():
            if verbose: print ('-----------------------')
            if sky not in d: raise Exception("no sky: %s in data" % sky)
            elif verbose: print ('sky: %s' % sky)
            if c['output'] not in d[sky]: raise Exception("no output: %s in data" % c["output"])
            elif verbose: print ('output: %s' % c['output'])
            if c['with_wedge']:
                if c['multipole'] != 3: c["multipole"] = 3
                if c['wedge_type'] not in ['PA-w1-w2', 'Q0-w1-w2']: raise Exception("no wedge_type %s")
                if verbose: print ('wedge (rotation of multipole): %s | read wedge instead of multipole in the following' % c['wedge_type'])
            if c['multipole'] > d[sky][c['output']]['multipole']: raise Exception("no %s multipoles in data, but %s" % (c["multipole"], d[sky][c['output']]['multipole']))
            elif verbose: print ('multipole: %s' % c['multipole'])
            if len(cut['min']) != c['multipole']: raise Exception("%s multipoles but %s min bounds provided" % (c['multipole'], len(cut['min'])))
            elif verbose: print ('min bound (per multipole): %s' % cut['min'])
            if len(cut['max']) != c['multipole']: raise Exception("%s multipoles but %s max bounds provided" % (c['multipole'], len(cut['max'])))
            elif verbose: print ('max bound (per multipole): %s' % cut['max'])
            if c['with_bao_rec']:
                if 'bao_rec' not in d[sky]: raise Exception("no bao_rec in data")
                if verbose: print ('bao rec: on')
            if c['with_ap']:
                if 'fid' not in d[sky]: raise Exception('no fiducial cosmology in data, cannot apply AP')
                if verbose: print ('coordinate (AP) distortion: on')
            if c['with_survey_mask'] and c['output'] != 'bPk': 
                    if verbose: print('survey mask only for bPk, not %s, disabling it' % c['output'])
                    c['with_survey_mask'] = False
            if c['with_survey_mask']:
                if 'survey_mask' not in d[sky][c['output']]: raise Exception('no survey mask in data, cannot apply mask')
                if verbose: print ('survey mask: on')
            if c['with_binning'] and c['with_survey_mask']: 
                    if verbose: print('survey mask matrix is already binned, disabling binning')
                    c['with_binning'] = False
            if c['with_binning']:
                if 'binsize' not in d[sky][c['output']]: raise Exception('no binning information in data, cannot apply binning')
                if verbose: print ('binning: on')
            if c['with_redshift_bin']:
                if 'redshift' not in d[sky]: raise Exception('no redshift information in data, cannot apply redshift selection')
                if verbose: print ('redshift selection: on')
            if c['with_loop_prior']:
                if 'c_sys_max' not in c['sky'][sky]: 
                    print ("loop prior: no upper bound estimate on the size of constant (k^0) systematics, setting it to default = 1/nbar")
                    c['sky'][sky]['c_sys_max'] = 1.
                if verbose: print ('loop prior: on')
            if verbose: print ('-----------------------')
        return

    def config(self, c, fd_sky):
        options_for_correlator = ["output", "multipole", "km", "kr", "nd", "with_emu", "with_resum", "fftaccboost",
                                  "eft_basis", "with_stoch", "with_nnlo_counterterm", "with_time", "with_exact_time",
                                  "with_ap", "with_survey_mask", "with_binning", "with_wedge", "with_redshift_bin"]

        fc_sky = [] # skylist of formatted config dict for Correlator

        for sky, fd in zip(c['sky'].keys(), fd_sky):
            fc = {}
            ### Added to allow redshift dependent nbar
            if 'nd' in c['sky'][sky].keys():
                fc['nd'] = c['sky'][sky]['nd']
            for option in options_for_correlator: 
                if option in c: 
                    fc[option] = c[option]
            fc['z'] = fd['z']
            fc['xdata'] = array(fd['x'])
            if 'Pk' in c['output']: 
                fc['kmax'] = float(max(np.array([k[-1] for k in fd['x_arr']])) + 0.05) # we take some margin
            if c['with_ap']: fc.update({'H_fid': fd['fid']['H'], 'D_fid': fd['fid']['D']})
            if c['with_survey_mask']:
                fc.update({'survey_mask_arr_p': fd['survey_mask_arr_p'], 'survey_mask_mat_kp': fd['survey_mask_mat_kp']})
                fc['kmax'] = float(max(array([k[-1] for k in fd['x_arr']])) + 0.1) # we give margin since mask mat_kp has a p-support of +/- 0.1 around k at ~ 0.1% precision
            if c['with_binning']: fc['binsize'] = float(fd['binsize'])
            if c['with_wedge']: fc['wedge_mat_wl'] = fd['wedge_mat_wl']
            if c['with_redshift_bin']: fc.update({'redshift_bin_zz': fd['redshift_bin_zz'], 'redshift_bin_nz': fd['redshift_bin_nz']})
            fc_sky.append(fc)

        return fc_sky

    def format(self, c, data, verbose=True):

        fd_sky = [] # skylist of formatted data dict

        for sky, cut in c['sky'].items():
            d = data[sky]
            dd = d[c['output']]

            xmask = [where((dd['x'] >= cut['min'][i]) & (dd['x'] <= cut['max'][i]))[0] for i in range(c["multipole"])]
            cmask = concatenate([xmask_i + i*len(dd['x']) for i, xmask_i in enumerate(xmask)]) 

            x_arr = array([dd['x'] for i in range(c['multipole'])])
            y_arr = array([dd['l%s'%(2*i)] for i in range(c['multipole'])])
            cov = dd['cov']

            if c['with_wedge']:
                mat = self.get_wedge_rotation_matrix(c['wedge_type'])
                y_arr = einsum('al,lk->ak', mat, y_arr)
                cov_resh = cov.reshape((3, cov.shape[0] // 3, 3, cov.shape[1] // 3))
                cov = einsum('al,bm,lkmj->akbj', mat, mat, cov_resh).reshape(cov.shape)

            y_err = sqrt(diag(cov)).reshape(3,-1)

            fdata = { 'z': d['z']['eff'], 
                      'mask_arr': xmask,                                               # mask for theory model for plotting
                      'x_arr': [x_arr[i, xmask_i] for i, xmask_i in enumerate(xmask)], # k in [h/Mpc] or s in [Mpc/h]
                      'y_arr': [y_arr[i, xmask_i] for i, xmask_i in enumerate(xmask)], # pk in [Mpc/h]^3 or cf for plotting
                      'y_err': [y_err[i, xmask_i] for i, xmask_i in enumerate(xmask)], # diagonal error bars   for plotting
                      'mask' : cmask,                                                  # mask for theory model for analysis (concatenated)
                      'x': dd['x']       ,                                             # k in [h/Mpc] or s in [Mpc/h] (unmasked) over which the theory model is evaluated
                      'y': y_arr.reshape(-1)[cmask],                                   # pk in [Mpc/h]^3 or cf for analysis (concatenated)
                    }

            if c['with_bao_rec']: 
                fdata['bao_rec_fid'] = d['bao_rec']['fid']
                fdata['y'] = concatenate((fdata['y'], array([d['bao_rec']['alpha']['par'], d['bao_rec']['alpha']['per']])))
                cmask = concatenate((cmask, array([-2, -1]))) 
                cross_fs_alpha = d['bao_rec']['cov']['cross-%s' % c['output']]
                cov_alpha = d['bao_rec']['cov']['alpha']
                cov = block([[cov, cross_fs_alpha], [cross_fs_alpha.T, cov_alpha]])

            fdata['p'] = linalg.inv(cov[ix_(cmask, cmask)])                 # precision matrix for analysis
            if dd['nsims'] > 0: 
                fdata['p'] *= (dd['nsims'] - len(cmask) - 2) / (dd['nsims'] - 1.) # Hartlap factor correction
                if verbose: print('%s: Hartlap factor correction on precision matrix estimated from %s mocks for %s bins' % (sky, dd['nsims'], len(cmask)))

            if c['with_ap']: fdata['fid'] = d['fid']
            if c["with_survey_mask"]: fdata.update({'survey_mask_arr_p': dd['survey_mask']['arr_p'], 
                                                    'survey_mask_mat_kp': dd['survey_mask']['mat_kp'][:c['multipole'],:c['multipole']]})
            if c['with_binning']: fdata['binsize'] = dd['binsize'] 
            if c['with_wedge']: fdata['wedge_mat_wl'] = mat # multipole-to-wedge rotation matrix (see above for definition)
            if c['with_redshift_bin']: fdata.update({'redshift_bin_z': d['redshift']['zz'], 'redshift_bin_nz': d['redshift']['nz']})
            fd_sky.append(fdata)
        return fd_sky

    def write(self, c, fd_sky, out=None):

        fit = True
        if out is None: out, fit = fd_sky, False

        if c['write']['fake'] and fit:
            # if c["with_wedge"]: raise Exception("Fake file generation not compatible with wedges, only with multipoles. ")
            fake_d = {}
            _c = deepcopy(c); _c['with_wedge'] = False 
            d, _, d_sky = self.read(_c, verbose=False)
            for i_sky, (sky, o) in enumerate(zip(c['sky'].keys(), out)):
                fake_d[sky] = {}
                self.write_common(fake_d[sky],
                    d[sky]['z']['min'], d[sky]['z']['max'], d[sky]['z']['eff'], 
                    d[sky]['fid']['Omega_m'], d[sky]['fid']['H'], d[sky]['fid']['D'])
                if c['output'] == 'bPk':
                    if c['with_survey_mask']: survey_mask_arr_p, survey_mask_mat_kp = d[sky]['bPk']['survey_mask']['arr_p'], d[sky]['bPk']['survey_mask']['mat_kp']
                    else: survey_mask_arr_p, survey_mask_mat_kp = None, None
                    if c['with_binning']: binsize = d[sky]['bPk']['binsize']
                    else: binsize = None
                    if c['with_wedge']: y_arr_unmasked = einsum('al,lk->ak', linalg.inv(self.get_wedge_rotation_matrix(c['wedge_type'])), o['y_arr_unmasked']) # rotating back the wedges to multipoles that are what is saved
                    else: y_arr_unmasked = 1. * o['y_arr_unmasked']
                    self.write_pk(fake_d[sky], c['multipole'], o['x_unmasked'], y_arr_unmasked, d[sky]['bPk']['cov'], d[sky]['bPk']['nsims'], survey_mask_arr_p=survey_mask_arr_p, survey_mask_mat_kp=survey_mask_mat_kp, binsize=binsize)
                if c['output'] == 'bCf':
                    if c['with_binning']: binsize = d[sky]['bCf']['binsize']
                    else: binsize = None
                    self.write_cf(fake_d[sky], c['multipole'], o['x_unmasked'], o['y_arr_unmasked'], d[sky]['bCf']['cov'], d[sky]['bCf']['nsims'], binsize=binsize)
                if c['with_bao_rec']:
                    if c['output'] == 'bPk': cov_cross_pk = d[sky]['bao_rec']['cov']['cross-bPk']
                    else: cov_cross_pk = None
                    if c['output'] == 'bCf': cov_cross_cf = d[sky]['bao_rec']['cov']['cross-bCf']
                    else: cov_cross_cf = None
                    self.write_bao_rec(fake_d[sky], d[sky]['bao_rec']['fid']['rd'], d[sky]['bao_rec']['fid']['H'], d[sky]['bao_rec']['fid']['D'], o['alpha'][0], o['alpha'][1], d[sky]['bao_rec']['cov']['alpha'], cov_cross_pk=cov_cross_pk, cov_cross_cf=cov_cross_cf)
            with h5py.File(os.path.join(c['write']['out_path'], 'fake_%s.h5') % c['write']['out_name'], 'w') as hf: save_dict_to_hdf5(hf, fake_d)
            # save(os.path.join(c['data_path'], 'fake_%s.npy') % c['write']['out_name'], fake_d) 
            print ('fake data from best fit saved to %s.' % c['write']['out_path'])
        for fdata, o, sky in zip(fd_sky, out, c['sky']):
            if c['write']['save']:
                header = self.set_header(o)
                for i, l in enumerate(range(0,2*c['multipole'],2)):
                    to_save = vstack([ fdata['x_arr'][i], fdata['y_arr'][i], fdata['y_err'][i] ])
                    if c['output'] == 'bPk': header += "k [h/Mpc], P_data_l%s [Mpc/h]^3, sigma_data_l%s [Mpc/h]^3" % (l, l)
                    elif c['output'] == 'bCf': header += 's [Mpc/h], C_data_l%s, sigma_data_l%s' % (l, l)
                    fmt = "%.4f %.6e %.6e"
                    if fit: 
                        to_save =  vstack([ to_save, o['y_arr'][i] ]) 
                        if c['output'] == 'bPk': header += ", P_theo_l%s [Mpc/h]^3" % l
                        elif c['output'] == 'bCf': header += ", C_theo_l%s" % l
                        fmt += " %.6e"
                    savetxt(os.path.join(c['write']['out_path'], 'fit_%s_%s_l%s.dat') % (c['write']['out_name'], sky, l), to_save.T, header=header, fmt=fmt)
                print ('data files with best fit saved to %s.' % c['write']['out_path'])
            if c['write']['plot']: 
                import matplotlib.pyplot as plt
                plt.figure()
                if c['output'] == 'bPk':
                    n, xlabel, ylabel = 1, r'$k \ [h/{\rm Mpc}]$', r'$k \ P_\ell(k) \ [{\rm Mpc}/h]^2$'
                elif c['output'] == 'bCf':
                    n, xlabel, ylabel = 2, r'$s \ [{\rm Mpc}/h]$', r'$s^2 \xi_\ell(s) \ [{\rm Mpc}/h]^2$'
                if c['with_wedge']: 
                    if c['wedge_type'] == 'PA-w1-w2': label=[r'$P\!\!\!/$', r'$w_1$', r'$w_2$']
                    elif c['wedge_type'] == 'Q0-w1-w2': label=[r'$Q_0$', r'$w_1$', '$w_2$']
                else: label = [r'$\ell=%s$' % (2*i) for i in range(c['multipole'])]
                for i in range(c['multipole']): 
                    plt.errorbar(fdata['x_arr'][i], fdata['x_arr'][i]**n*fdata['y_arr'][i], 
                                yerr=fdata['x_arr'][i]**n*fdata['y_err'][i],
                                fmt='.', label=label[i])
                    if fit: plt.plot(fdata['x_arr'][i], fdata['x_arr'][i]**n*o['y_arr'][i], 'k')
                plt.title(sky)
                plt.legend()
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                if c['write']['save']: 
                    plt.savefig(os.path.join(c['write']['out_path'], 'plot_%s_%s.pdf') % (c['write']['out_name'], sky), bbox_inches='tight')
                    print ('data plot with best fit saved to %s.' % c['write']['out_path'])
                if c['write']['show']: plt.show()
        return

    def write_common(self, d_sky, zmin, zmax, zeff, Omega_m_fid, H_fid, D_fid):
        d_sky.update({ 
           'z': {'min': zmin, 'max': zmax, 'eff': zeff}, 
           'fid': {'Omega_m': Omega_m_fid, 'H': H_fid,  'D': D_fid}, 
        })
        return 
    
    def write_pk(self, d_sky, mult_pk, kk, pk, cov_pk, nsims_cov_pk=0, survey_mask_arr_p=None, survey_mask_mat_kp=None, binsize=None):
        d_sky['bPk'] = {'multipole': mult_pk, 'x': kk, 
                   'cov': cov_pk, 'nsims': nsims_cov_pk, 
                   'survey_mask': {'arr_p': survey_mask_arr_p, 'mat_kp': survey_mask_mat_kp},
                   'binsize': binsize}
        for i, l in enumerate(range(0,2*mult_pk,2)): d_sky['bPk']['l%s' % l] = pk[i]
        return 
    
    def write_cf(self, d_sky, mult_cf, ss, cf, cov_cf, nsims_cov_cf=0, binsize=None):
        d_sky['bCf'] = {'multipole': mult_cf, 'x': ss, 
                   'cov': cov_cf, 'nsims': nsims_cov_cf,
                   'binsize': binsize }
        for i, l in enumerate(range(0,2*mult_cf,2)): d_sky['bCf']['l%s' % l] = cf[i]
        return 
    
    def write_bao_rec(self, d_sky, bao_rec_rd_fid, bao_rec_H_fid, bao_rec_D_fid, alpha_par, alpha_per, cov_alpha, cov_cross_pk=None, cov_cross_cf=None):
        d_sky['bao_rec'] = {'fid': {'rd': bao_rec_rd_fid, 'H': bao_rec_H_fid, 'D': bao_rec_D_fid} ,
                    'alpha': {'par': alpha_par, 'per': alpha_per}, 
                    'cov': {'alpha': cov_alpha, 'cross-bPk': cov_cross_pk, 'cross-bCf': cov_cross_cf}}
        return 

    def set_header(self, out): 
        header = "fit | chi2 = %.2f | parameters: " % out['chi2']
        if 'cosmo' in out: 
            for key, value in out['cosmo'].items(): header += "%s: %.4e, " % (key, value)
        header += "\n"
        if 'eft_parameters' in out:
            for key, value in out['eft_parameters'].items(): header += "%s: %.4f, " % (key, value)
            header += "\n"
        return header

    def get_wedge_rotation_matrix(self, wedge_type='PA-w1-w2'):
        if wedge_type == 'PA-w1-w2': mat = array([[1., -3./7., 11./56.], [1., -3/8., 15/128.], [1., 3/8., -15./128.]])
        elif wedge_type == 'Q0-w1-w2': mat = array([[1., -1./2., 3./8.], [1., -3/8., 15/128.], [1., 3/8., -15./128.]])
        return mat
