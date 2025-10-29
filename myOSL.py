# -*- coding: utf-8 -*-
"""
Module for the myOSL whole body dosimeter dose calculation algorithm.

Created on 05.03.2021
@Author:  A. Pitzschke, IRA, CHUV (andreas.pitzschke@chuv.ch)

05.07.24 : major clean-up for RSD-method and response fitting
"""

##############################################################################
# HEADER
##############################################################################

__all__ = [
    'DoseCalcConfig',
    'DosimeterCharacteristics',
    'DoseLSQOptimizer',
    'compute_dose_from_readout',
    'generate_readout_from_dose',
    'benchmark_algorithm_robustness'
]
__version__ = '0.1'
__author__ = 'Andreas Pitzschke'
__email__ = 'andreas.pitzschke@chuv.ch'
__copyright__ = ''

# Import libraries
import os
import numpy as np
from numpy.typing import NDArray
from scipy import interpolate
from scipy.optimize import minimize
from scipy import stats
from scipy.optimize import lsq_linear
import copy
from tqdm import trange
from colorama import Fore
import pprint
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time


##############################################################################
# HELPER FUNCTIONS
##############################################################################


def benchmark_algorithm_robustness(dose_in=None, cfg_alg=None, cfg_noise: dict = None):
    """
    Benchmarks the dose calculation algorithm simulating (Monte Carlo) detector noise.

    Syntax:
    ------
       df, df_quart = check_algorithm_robustness(dose=None, cfg=None, noise=None)

    Args:
    ----
        - dose (dict or object from DoseCalculator):
            see also help(generate_readout_from_dose)
        - cfg (dict or object from DoseCalcConfig)
            see also help(DoseCalcConfig)
        - noise (dict for noise settings)

    Output:
    ------
        df (Pandas dataframe): data of benchmark
        df_quart (Pandas dataframe): quartiles of benchmark data

    -------------------------------------------------------------------------
    @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                University hospital Lausanne, Switzerland, 2025
    """
    # start timer
    start_time = time.perf_counter()

    # init. MC
    if cfg_noise is None:
        cfg_noise = dict({
            'seed':             13395,      # seed of numpy rng
            'niter':            30,         # take a large number, this one is only for testing
            'distribution':     'normal',
            # according to http://www.isgmax.com/Articles_Papers/Selecting%20and%20Applying%20Error%20Distributions.pdf
            'scale':            0.15,       # 3*sigma from https://doi.org/10.1016/j.radmeas.2024.107346
            'normalize':        True,       # renormalize total dose+noise to 1 mSv
            'save_to_csv':      False,      # export panda data frames
            'plot_results':     True
        })

    # check for configuration
    if not isinstance(cfg_alg, (dict, DoseCalcConfig)):
        print('No configuration provided. Fallback to defaults.')
        cfg_alg = DoseCalcConfig(selection='mixed')

    else:
        if isinstance(cfg_alg, dict):
            cfg_alg = DoseCalcConfig(config=cfg_alg)

    # disable verbosity of algorithm
    cfg_alg.verbose = 0

    # check for reference dose values
    if dose_in is None:

        # dictionary
        help(benchmark_algorithm_robustness)
        print("No reference dose provided. Fallback to example.")
        dose = dict({
            'photons': dict({
                'Hp(10)':       1.00,
                'Hp(0.07)':     1.00,
                'energy':       0.662,
            }),
            'betas': dict({
                'Hp(0.07)':     [0.00, 0.00],
                'energy':       [0.8, 0.25],
            })
        })

    elif not isinstance(dose_in, (dict, DoseCalculator, DoseLSQOptimizer)):

        # transfer from dose object
        dose = dict({
            'photons': dict({
                'Hp(10)':   dose_in.Hp10.get('value'),
                'Hp(0.07)': dose_in.Hp007_ph.get('value'),
                'energy':   dose_in.effective_photon_energy.get('photons').get('value'),
            }),
            'betas': dict({
                'Hp(0.07)': [dose_in.Hp007_beta_highE.get('value'), dose_in.Hp007_beta_lowE.get('value')],
                'energy':   [dose_in.Hp007_beta_highE.get('energy'), dose_in.Hp007_beta_lowE.get('energy')]
            })
        })

    else:
        dose = dose_in.copy()

    # get calibration information and reference data
    dm_characteristics = DosimeterCharacteristics(config=cfg_alg)

    # define seed of rng
    np.random.seed(cfg_noise.get('seed'))

    # init. some constants
    if cfg_alg.icru_version == 57:
        E_Cs = 0.662
    elif cfg_alg.icru_version == 93:
        E_Cs = 0.639
    else:
        raise ValueError('ICRU version must be 57 or 93')

    # 1) iterate over photon energies
    i_iter = 0
    data = list()
    for index in trange(
            len(dm_characteristics.response_photons.get('energy')),
            position=0, leave=True, desc='Progress',
            bar_format="{l_bar}%s{bar}%s{r_bar} [time left: {remaining}]" %
                       (Fore.GREEN, Fore.RESET)
    ):

        # get current photon energy, update settings and compute readout signal
        energy_ref = dm_characteristics.response_photons.get('energy')[index]
        dose_iter = copy.deepcopy(dose)
        dose_iter.get('photons').update({'energy': dm_characteristics.response_photons.get('energy')[index]})
        readout0 = generate_readout_from_dose(dose_in=dose_iter, cfg=cfg_alg)

        # 2) iterate over repetitions
        for _ in range(int(cfg_noise.get('niter'))):

            # compute noise from probability distribution
            if cfg_noise.get('distribution') == 'normal':
                sgm = np.random.normal(
                    loc=0., scale=cfg_noise.get('scale'), size=4
                )

            elif cfg_noise.get('distribution') == 'uniform':
                sgm = np.random.uniform(low=-1, high=1, size=4)

            elif cfg_noise.get('distribution') == 'triangular':
                sgm = np.random.triangular(left=-1, mode=0, right=1, size=4)

            else:
                raise ValueError('Unknown probability function specified')

            # apply noise to readout
            readout1 = dict()
            for k, key in enumerate(readout0.keys()):
                # update new readout dict. by applying z = mu + sigma * err
                z = readout0.get(key).get('value') + sgm[k] * cfg_noise.get('scale')
                if z < 0:
                    z = 0.0
                readout1.update({key: {'value': fastround(z, decimals=4)}})

            # recompute dose from noisy data
            out = object()
            if cfg_alg.optimization.get('lsq'):
                # LSQ optimization enabled
                _, out = compute_dose_from_readout(rdout=readout1, cfg=cfg_alg)
            else:
                # LSQ optimization disabled
                out = compute_dose_from_readout(rdout=readout1, cfg=cfg_alg)

            # get dose conversion factors
            index = np.abs(out.lookup.response_photons.get('energy') - energy_ref).argmin()
            hpK10 = out.lookup.hpK.get('hpK10')[index]
            hpK007 = out.lookup.hpK.get('hpK007')[index]

            if dose_iter.get('photons').get('Hp(0.07)') is not None:
                Hp007ph_ref = dose_iter.get('photons').get('Hp(0.07)')
                Hp10_ref = Hp007ph_ref / hpK007 * hpK10
            elif dose_iter.get('photons').get('Hp(10)') is not None:
                Hp10_ref = dose_iter.get('photons').get('Hp(10)')
                Hp007ph_ref = Hp10_ref / hpK10 * hpK007
            else:
                Hp10_ref, Hp007ph_ref = None, None

            # compute noise corrected reference beta dose values
            Hp007b_ref = np.sum(dose_iter.get('betas').get('Hp(0.07)'))

            # store in data frame
            data.append(dict({
                'energy_ref':   energy_ref,
                'Hp10_ref':     fastround(Hp10_ref, 3),
                'Hp007ph_ref':  fastround(Hp007ph_ref, 3),
                'Hp007b_ref':   fastround(Hp007b_ref, 3),
                'Hp007_ref':    fastround(Hp007ph_ref + Hp007b_ref, 3),
                'H1_ref':       fastround(readout1.get('R1').get('value'), 3),
                'H2_ref':       fastround(readout1.get('R2').get('value'), 3),
                'H3_ref':       fastround(readout1.get('R3').get('value'), 3),
                'H4_ref':       fastround(readout1.get('R4').get('value'), 3),
                'E':            out.effective_photon_energy.get('value'),
                'E_uc':         out.effective_photon_energy.get('uncert'),
                'channels':     out.effective_photon_energy.get('channels'),
                'Ka':           out.Kerma.get('value'),
                'Hp10':         out.Hp10.get('value'),
                'Hp10_uc':      out.Hp10.get('uncert'),
                'Hp007':        out.Hp007.get('value'),
                'Hp007_uc':     out.Hp007.get('uncert'),
                'Hp007ph':      out.Hp007_ph.get('value'),
                'Hp007ph_uc':   out.Hp007_ph.get('uncert'),
                'Hp007b':       out.Hp007_beta_highE.get('value') + out.Hp007_beta_lowE.get('value'),
                'Hp007b_uc':    fastround(
                    np.sqrt(
                        out.Hp007_beta_highE.get('uncert') ** 2
                        + out.Hp007_beta_highE.get('uncert') ** 2
                    ), 3),
                'Hp007b_highE':     out.Hp007_beta_highE.get('value'),
                'Hp007b_highE_uc':  out.Hp007_beta_highE.get('uncert'),
                'Hp007b_lowE':      out.Hp007_beta_lowE.get('value'),
                'Hp007b_lowE_uc':   out.Hp007_beta_lowE.get('uncert'),
            })
            )

            # update counter
            i_iter = i_iter + 1

    # stop timer and display execution details
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(
        """
        Execution time of loop:
        {:6.0f} dose calculations
        {:6.3f} seconds in total
        {:6.3f} calc./sec"""
        .format(i_iter, elapsed, i_iter / elapsed)
    )

    # transform into DataFrame
    df = pd.DataFrame(data)

    # compute relative responses
    df['R(E)'] = fastround(df['E'] / df['energy_ref'], 3)
    df['R(Hp10)'] = fastround(df['Hp10'].div(df['Hp10_ref']).replace(np.inf, 0), 3)
    df['R(Hp007ph)'] = fastround(df['Hp007ph'].div(df['Hp007ph_ref']).replace(np.inf, 0), 3)
    df['R(Hp007b)'] = fastround(df['Hp007b'].div(df['Hp007b_ref']).replace(np.inf, 0), 3)
    df['R(Hp007)'] = fastround(df['Hp007'].div(df['Hp007_ref']).replace(np.inf, 0), 3)

    # whisker plot of R(Eeff)
    if cfg_noise.get('plot_results'):

        if cfg_alg.icru_version == 57:
            E_Cs = 0.662
        elif cfg_alg.icru_version == 93:
            E_Cs = 0.639

        # figure for photon energies
        fig, ax = plt.subplots(1, 1, sharex=True, tight_layout=True)
        fig.set_size_inches(18.5, 10.5, forward=True)
        chart = sns.boxplot(
            data=df, x='energy_ref', y='R(E)', ax=ax,
            medianprops=dict(color="red", alpha=0.7),
            notch=True)
        ax.set(
            xlabel=r'$E_{eff}$ [MeV]',
            ylabel=r'$E_{alg}\,/\,E_{eff}$',
            yscale='log', ylim=[1e-2, 1e2])
        # chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
        ax.grid(True, which="both", ls="--", c='gray')

        # figure with dose values
        fig, ax = plt.subplots(2, 2, sharex=True, tight_layout=True)
        fig.set_size_inches(18.5, 10.5, forward=True)

        # subfigure 1: R(Hp10) vs Eref
        chart = sns.boxplot(
            data=df, x='energy_ref', y='R(Hp10)', ax=ax[0, 0],
            medianprops=dict(color="red", alpha=0.7),
            notch=True)
        ax[0, 0].set(
            xlabel=r'$E_{eff}$ [MeV]',
            ylabel=r'$R(H{_p}(10))$',
            yscale='linear')
        # chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
        ax[0, 0].grid(True, which="both", ls="--", c='gray')

        # subfigure 2: R(Hp007, tot) vs Eref
        chart = sns.boxplot(
            data=df, x='energy_ref', y='R(Hp007)', ax=ax[0, 1],
            medianprops=dict(color="red", alpha=0.7),
            notch=True)
        ax[0, 1].set(
            xlabel=r'$E_{eff}$ [MeV]',
            ylabel=r'$R(H{_{p,tot}}(0.07))$',
            yscale='linear')
        # chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
        ax[0, 1].grid(True, which="both", ls="--", c='gray')

        # subfigure 3: R(Hp007, photons) vs Eref
        chart = sns.boxplot(
            data=df, x='energy_ref', y='R(Hp007ph)', ax=ax[1, 0],
            medianprops=dict(color="red", alpha=0.7),
            notch=True)
        ax[1, 0].set(
            xlabel=r'$E_{eff}$ [MeV]',
            ylabel=r'$R(H{_p}(0.07, \nu))$',
            yscale='linear')
        chart.set_xticks(chart.get_xticks())
        chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
        ax[1, 0].grid(True, which="both", ls="--", c='gray')

        # subfigure 4: R(Hp007, betas) vs Eref
        chart = sns.boxplot(
            data=df, x='energy_ref', y='R(Hp007b)', ax=ax[1, 1],
            medianprops=dict(color="red", alpha=0.7),
            notch=True)
        ax[1, 1].set(
            xlabel=r'$E_{eff}$ [MeV]',
            ylabel=r'$R(H{_p}(0.07, \beta))$',
            yscale='linear')
        chart.set_xticks(chart.get_xticks())
        chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
        ax[1, 1].grid(True, which="both", ls="--", c='gray')

        # add limits (reference irradiation conditions)
        list_ax = [[0, 0], [1, 0], [0, 1], [1, 1]]
        for ll in list_ax:
            x = [min(chart.get_xticks()), max(chart.get_xticks())]
            ax[ll[0], ll[1]].plot(
                x, [0.7, 0.7],
                marker='', linestyle='--', color='C01'
            )
            ax[ll[0], ll[1]].plot(
                x, [1.3, 1.3],
                marker='', linestyle='--', color='C01'
            )
            x_Cs = dm_characteristics.response_photons.get(
                'energy').index(E_Cs)
            ax[ll[0], ll[1]].plot(
                [x_Cs - 0.5, x_Cs + 0.5], [0.9, 0.9],
                marker='', linestyle='--', color='C01'
            )
            ax[ll[0], ll[1]].plot(
                [x_Cs - 0.5, x_Cs + 0.5], [1.1, 1.1],
                marker='', linestyle='--', color='C01'
            )

        plt.show()

    # evaluating precision of calculated energy and dose quantities:
    # Hp(10)
    chisq_Hp10 = np.sum((df['R(Hp10)'] - 1) ** 2) / cfg_noise.get('niter')
    chisq_Hp10_Cs = np.sum((df['R(Hp10)'].loc[df['energy_ref'] == E_Cs] - 1) ** 2) / cfg_noise.get('niter')

    # Hp(0.07)
    chisq_Hp007 = np.sum((df['R(Hp007)'] - 1) ** 2) / cfg_noise.get('niter')
    chisq_Hp007_Cs = np.sum((df['R(Hp007)'].loc[df['energy_ref'] == E_Cs] - 1) ** 2) / cfg_noise.get('niter')

    # photon energy
    chisq_E = np.sum((df['R(E)'] - 1) ** 2) / cfg_noise.get('niter')
    chisq_E_Cs = np.sum((df['R(E)'].loc[df['energy_ref'] == E_Cs] - 1) ** 2) / cfg_noise.get('niter')

    # display results
    print(80 * '-')
    print('{:30}\t:\t{}'.format('Calculation method for Eeff',
          cfg_alg.calc_method.get('energy')))
    print('{:30}\t:\t{}'.format('Calculation method for dose',
          cfg_alg.calc_method.get('dose')))
    print('{:30}\t:\t{}'.format('Least-square optimization',
          cfg_alg.optimization.get('lsq')))
    print(80 * '-')
    # print('{:30}\t:\t{:3.2e}'.format('ChiSq(Hp10(all))', chisq_Hp10))
    # print('{:30}\t:\t{:3.2e}'.format('ChiSq(Hp10(137Cs))', chisq_Hp10_Cs))
    print('{:30}\t:\t{:3.2e}'.format('SD(R(Hp10(all)))', np.sqrt(chisq_Hp10)))
    print('{:30}\t:\t{:3.2e}'.format('SD(R(Hp10(137Cs)))', np.sqrt(chisq_Hp10_Cs)))
    print(80 * '-')
    # print('{:30}\t:\t{:3.2e}'.format('ChiSq(Hp007(all))', chisq_Hp007))
    # print('{:30}\t:\t{:3.2e}'.format('ChiSq(Hp007(137Cs))', chisq_Hp007_Cs))
    print('{:30}\t:\t{:3.2e}'.format('SD(R(Hp007(all)))', np.sqrt(chisq_Hp007)))
    print('{:30}\t:\t{:3.2e}'.format('SD(R(Hp007(137Cs)))', np.sqrt(chisq_Hp007_Cs)))
    print(80 * '-')
    # print('{:30}\t:\t{:3.2e}'.format('ChiSq(E)', chisq_E))
    print('{:30}\t:\t{:3.2e}'.format('SD(R(E))', np.sqrt(chisq_E)))
    print('{:30}\t:\t{:3.2e}'.format('SD(R(E(137Cs)))', np.sqrt(chisq_E_Cs)))

    # compute quartiles and more
    df_quart = df[['energy_ref', 'R(E)', 'R(Hp10)', 'R(Hp007)']]
    df_quart10 = df_quart.groupby(by='energy_ref').quantile(q=0.10).reset_index()
    df_quart25 = df_quart.groupby(by='energy_ref').quantile(q=0.25).reset_index()
    df_quart50 = df_quart.groupby(by='energy_ref').quantile(q=0.50).reset_index()
    df_quart75 = df_quart.groupby(by='energy_ref').quantile(q=0.75).reset_index()
    df_quart90 = df_quart.groupby(by='energy_ref').quantile(q=0.90).reset_index()
    df_quart = pd.concat(
        [df_quart10, df_quart25, df_quart50, df_quart75, df_quart90],
        keys=['quart10', 'quart25', 'quart50', 'quart75', 'quart90'],
    )

    # naive physicist req. for everything: 0.9 <= R <= 1.1 :D
    df_perc10 = df[['energy_ref', 'R(E)', 'R(Hp10)', 'R(Hp007)']]
    df_perc10 = df_perc10.groupby(by='energy_ref').agg(
        E_cnt=('R(E)', 'count'),
        E_in=('R(E)', lambda x: ((x >= 0.9) & (x <= 1.1)).sum()),
        Hp10_cnt=('R(Hp10)', 'count'),
        Hp10_in=('R(Hp10)', lambda x: ((x >= 0.9) & (x <= 1.1)).sum()),
        Hp007_cnt=('R(Hp007)', 'count'),
        Hp007_in=('R(Hp007)', lambda x: ((x >= 0.9) & (x <= 1.1)).sum())
    )

    # US req. (2020): 0.8 <= R <= 1.2
    # https://www.nrc.gov/reading-rm/doc-collections/cfr/part034/part034-0047.html
    df_perc20 = df[['energy_ref', 'R(E)', 'R(Hp10)', 'R(Hp007)']]
    df_perc20 = df_perc20.groupby(by='energy_ref').agg(
        E_cnt=('R(E)', 'count'),
        E_in=('R(E)', lambda x: ((x >= 0.8) & (x <= 1.2)).sum()),
        Hp10_cnt=('R(Hp10)', 'count'),
        Hp10_in=('R(Hp10)', lambda x: ((x >= 0.8) & (x <= 1.2)).sum()),
        Hp007_cnt=('R(Hp007)', 'count'),
        Hp007_in=('R(Hp007)', lambda x: ((x >= 0.8) & (x <= 1.2)).sum())
    )

    # IEC 62387 (2020) requirements: 0.71 <= R <= 1.67
    df_iec = df[['energy_ref', 'R(E)', 'R(Hp10)', 'R(Hp007)']]
    df_iec = df_iec.groupby(by='energy_ref').agg(
        E_cnt=('R(E)', 'count'),
        E_in=('R(E)', lambda x: ((x >= 0.71) & (x <= 1.67)).sum()),
        Hp10_cnt=('R(Hp10)', 'count'),
        Hp10_in=('R(Hp10)', lambda x: ((x >= 0.71) & (x <= 1.67)).sum()),
        Hp007_cnt=('R(Hp007)', 'count'),
        Hp007_in=('R(Hp007)', lambda x: ((x >= 0.71) & (x <= 1.67)).sum())
    )

    # Swiss ordinance on dosimetry (2018) requirements : 0.7 <= R <= 1.3
    df_ch = df[['energy_ref', 'R(E)', 'R(Hp10)', 'R(Hp007)']]
    df_ch = df_ch.groupby(by='energy_ref').agg(
        E_cnt=('R(E)', 'count'),
        E_in=('R(E)', lambda x: ((x >= 0.7) & (x <= 1.3)).sum()),
        Hp10_cnt=('R(Hp10)', 'count'),
        Hp10_in=('R(Hp10)', lambda x: ((x >= 0.7) & (x <= 1.3)).sum()),
        Hp007_cnt=('R(Hp007)', 'count'),
        Hp007_in=('R(Hp007)', lambda x: ((x >= 0.7) & (x <= 1.3)).sum())
    )

    # concatenate
    df_req = pd.concat(
        [df_perc10, df_perc20, df_iec, df_ch],
        keys=['10perc', '20perc', 'iec', 'ch'],
    )
    df_req['E_perc'] = 1e2 * df_req['E_in'] / df_req['E_cnt']
    df_req['Hp10_perc'] = 1e2 * df_req['Hp10_in'] / df_req['Hp10_cnt']
    df_req['Hp007_perc'] = 1e2 * df_req['Hp007_in'] / df_req['Hp007_cnt']

    # save to Excel file
    if cfg_noise.get('save_to_csv'):
        cwd = os.getcwd()
        cwd = os.path.join(cwd, 'DATA')

        # all data
        fnm_all = 'ICRU-{}_method_{}_all.csv'.format(cfg_alg.icru_version, cfg_alg.calc_method['energy'])
        df.round(3).to_csv(os.path.join(cwd, fnm_all))

        # quantiles
        fnm_quart = 'ICRU-{}_method_{}_quartiles.csv'.format(cfg_alg.icru_version, cfg_alg.calc_method['energy'])
        df_quart.round(3).to_csv(os.path.join(cwd, fnm_quart))

        # regulatory requirements
        fnm_req = 'ICRU-{}_method_{}_regulatory.csv'.format(cfg_alg.icru_version, cfg_alg.calc_method['energy'])
        df_req.round(3).to_csv(os.path.join(cwd, fnm_req))

    return df, df_quart, df_req


##############################################################################

def generate_readout_from_dose(dose_in=None, cfg=None) -> dict:
    """
    Compute the readout for a given dose.

    Syntax:
    ------
       res = generate_readout_from_dose(dose=None, cfg=None)

    Args:
    ----
        - dose (dict): contains data for the dose according to the keys:
            - 'photons': subdict. with the keys:
                - 'Hp(10)': organ dose in mSv (float)
                - 'Hp(0.07)':skin dose in mSv (float)
                - ... or 'Ka' : air kerma in mGy (float)
                - 'energy': effective energy in MeV (float)
            - 'betas': subdict. with the keys:
                - 'Hp(0.07)':skin dose in mSv (list)
                - 'energy': effective energy in MeV (list)
        - or as object from DoseCalculator

        - cfg (dict or object from DoseCalcConfig)

    Example for dose dictionary (Hp(0.07) is used as operational quantity):
        dose = dict({
            'photons': dict({
                'Hp(0.07)': 1.00,
                'energy':   0.662,
                }),
            'betas': dict({
                'Hp(0.07)': [1.00],
                'energy':   ['90Sr/90Y'],
                })
            })

    @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                University hospital Lausanne, Switzerland, 2025
    """
    # init.
    energy_nuclide = {
        0.8:    '90Sr/90Y',
        0.25:   '85Kr'
    }

    # parse input
    if dose_in is None:
        # define default input
        help(generate_readout_from_dose)
        print("No dose provided. Fallback to example.")
        dose_in = dict({
            'photons': dict({
                'Hp(0.07)':     1.00,
                'energy':       0.662,
            }),
            'betas': dict({
                'Hp(0.07)':     [1.00],
                'energy':       [energy_nuclide[0.8]],
            })
        })
        print(dose_in)

    # check format of input
    if not isinstance(dose_in, (dict, DoseCalculator, DoseLSQOptimizer)):
        raise ValueError('No compatible dose format.')

    # simplify data input: copy data from object into dictionary
    if isinstance(dose_in, (DoseCalculator, DoseLSQOptimizer)):
        dose = dict({
            'photons': dict({
                # 'Ka':       dose_in.Kerma.get('value'),
                'Hp(10)':   dose_in.Hp10.get('value'),
                'Hp(0.07)': dose_in.Hp007_ph.get('value'),
                'energy':   dose_in.effective_photon_energy.get('value'),
            }),
            'betas': dict({
                'Hp(0.07)': [dose_in.Hp007_beta_highE.get('value'), dose_in.Hp007_beta_lowE.get('value')],
                'energy':   [dose_in.Hp007_beta_highE.get('energy'), dose_in.Hp007_beta_lowE.get('energy')]
            })
        })
    else:
        dose = copy.deepcopy(dose_in)

    # check for configuration
    if isinstance(cfg, dict):

        # transform into object
        cfg = DoseCalcConfig(config=cfg)

    if not isinstance(cfg, DoseCalcConfig):

        # fallback
        print("No configuration provided. Fallback to defaults.")
        cfg = DoseCalcConfig(selection='mixed')

    # select all channels
    channels = ('R1', 'R2', 'R3', 'R4')

    # construct objects
    dm_characteristics = DosimeterCharacteristics(config=cfg)
    lut = LookupTable(config=cfg, dm_characteristics=dm_characteristics)

    # seek indices for calculated and reference energy
    E_ph = dose.get('photons').get('energy')
    index_E = np.abs(lut.response_photons.get('energy') - E_ph).argmin()

    # retrieve dose conversion coefficients
    dose.update({
        'hpK': {
                'Hp007':    float(lut.hpK.get('hpK007')[index_E]),
                'Hp10':     float(lut.hpK.get('hpK10')[index_E])
                }
        })

    # compute air kerma
    if dose.get('photons').get('Hp(0.07)') is not None:
        # use Hp(0.07)
        dose.get('photons').update({
            'Ka': float(fastround(
                dose.get('photons').get('Hp(0.07)') / dose['hpK']['Hp007'],
                decimals=4
                ))
        })
    elif dose.get('photons').get('Hp(10)') is not None:
        # use Hp(10)
        dose.get('photons').update({
            'Ka': float(fastround(
                dose.get('photons').get('Hp(10)') / dose['hpK']['Hp10'],
                decimals=4
                ))
        })
    else:
        raise ValueError('No valid dose quantity provided')

    # (re-)compute Hp(0.07) for photons
    dose.get('photons').update({
        'Hp007': float(fastround(dose.get('photons').get('Ka') * dose['hpK']['Hp007'], decimals=4))
    })

    # (re-)compute Hp(10) for photons
    dose.get('photons').update({
        'Hp10': float(fastround(dose.get('photons').get('Ka') * dose['hpK']['Hp10'], decimals=4))
    })

    # get lookup table normalization
    cf_norm = system_calibration_normalization(lookup=lut)

    # compute detector signals for photon contribution: Hi = Ka*hpK(d)*Ri/coeff
    H_ph = dict()
    for key in channels:
        H_ph.update({
            key: (
                    dose.get('photons').get('Ka')                               # air kerma
                    / cf_norm.get('photons')                                    # system calibration correction
                    * dose['hpK'].get(lut.normalization, 1.0)                   # kerma to dose equivalent conv. factor
                    * lut.response_photons[lut.normalization][key][index_E]     # response in lookup table
                    )
        })

    # cumulated beta contribution to Hp(0.07)
    H_b = dict()
    for key in channels:
        H_b_i = 0.0
        if dose.get('betas').get('energy') is not None:
            for idx, ergy in enumerate(dose.get('betas').get('energy')):
                H_b_i += (
                    dose.get('betas').get('Hp(0.07)')[idx]
                    / cf_norm.get('betas')
                    * lut.response_betas[lut.normalization][key][idx]    # lookup table response
                )
        H_b.update({key: H_b_i})

    # combine photon and beta doses
    H_tot = dict()
    for key in H_ph.keys():
        H_tot.update({
            key: dict({
                'value': float(fastround(H_ph.get(key) + H_b.get(key), 3))
            })
        })

    return H_tot


##############################################################################


def system_calibration_normalization(lookup: object = None, quantity: str = None) -> dict:
    """
    Yield normalization coefficients for readout and dose calculation.

    Syntax:
    ------
       cf_norm = system_calibration_normalization(lookup, quantity)

    Args:
    ----
        - lookup (object): lookup table from class 'LookupTable'
        - measuring quantity 'kerma', 'Hp007' or 'Hp10' (value by default: lookup.config.normalization)

    Output:
    ------
        dictionary with system normalization coefficient: needs to be multiplied
        with the readout to obtain the correct dose normalization, e.g.
        kerma = readout * system_calibration_normalization(lookup)

    @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                University hospital Lausanne, Switzerland, 2025
    """
    # check consistency of lookup table
    if not isinstance(lookup, LookupTable):
        raise ValueError('Config file must be of class DoseCalcConfig')

    if quantity is None:
        quantity = lookup.config.normalization
    else:
        if quantity not in ('kerma', 'Hp007', 'Hp10'):
            raise ValueError("Measuring quantity mut be: 'kerma', 'Hp007' or 'Hp10'")

    # define energy of 137Cs
    if lookup.config.icru_version == 57:
        E_Cs = 0.662
    elif lookup.config.icru_version == 93:
        E_Cs = 0.639
    else:
        raise ValueError('ICRU version must be 57 or 93.')

    # if E is None:
    #     print('No effective photon energy provided. Fallback to 137Cs energy.')
    # E = E_Cs

    # seek indices for calculated and reference energy
    # index_E = np.abs(lookup.response_photons.get('energy') - E).argmin()
    index_Cs = np.abs(lookup.response_photons.get('energy') - E_Cs).argmin()

    if quantity == 'kerma':
        # air kerma (NOTE: system calibration is in terms of Hp(10, 137Cs))
        cf_norm = {
            # 'photons':  1.0 / lookup.hpK.get('hpK10')[index_Cs],
            # 'betas':    1.0 / lookup.hpK.get('hpK10')[index_Cs],
            'photons':  float(1.0 / lookup.hpK.get('hpK10')[index_Cs]),
            'betas':    float(1.0 / lookup.hpK.get('hpK10')[index_Cs]),
        }
    elif quantity == 'Hp007':
        # skin dose
        cf_norm = {
            # 'photons':  1.0 / lookup.hpK.get('hpK007')[index_E],
            # 'betas':    1.0 / (lookup.hpK.get('hpK007')[index_Cs] / lookup.hpK.get('hpK10')[index_Cs]),
            'photons':  float(lookup.hpK.get('hpK007')[index_Cs] / lookup.hpK.get('hpK10')[index_Cs]),
            'betas':    float(lookup.hpK.get('hpK007')[index_Cs] / lookup.hpK.get('hpK10')[index_Cs]),
        }
    elif quantity == 'Hp10':
        # organ dose
        cf_norm = {
            # 'photons':  1.0 / lookup.hpK.get('hpK10')[index_E],
            # 'betas':    1.0 / (lookup.hpK.get('hpK007')[index_Cs] / lookup.hpK.get('hpK10')[index_Cs]),
            'photons':  1.0,
            'betas':    1.0,
        }
    else:
        raise ValueError('Unknown normalization of lookup table')

    return cf_norm


##############################################################################

def compute_dose_from_readout(rdout: dict = None, cfg: object = None) -> object:
    """
    Calculate dose from dosimeter readout.

    Syntax:
    ------
       res = compute_dose(rdout=None, cfg=None)

    Args:
    ----
        - rdout (dict): contains the data from the four elements with the
            keys R1...R4 and sub-dictionaries containing the elements in mSv:
                - 'value' (float)
                - 'uncert' (float)
                - 'background' (float)
        - cfg (dict) : dose calculation configuration (see help of DoseCalcConfig)

    Example for readout:
    -------------------
        rdout = dict({
            'R1': {'value': 1.0, 'uncert': 0.05, 'background' : 0.1},
            'R2': {'value': 1.0, 'uncert': 0.05, 'background' : 0.1},
            'R3': {'value': 1.0, 'uncert': 0.05, 'background' : 0.1},
            'R4': {'value': 1.0, 'uncert': 0.05, 'background' : 0.1}
            })

    Output:
    ------
        dose (object)

    @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                University hospital Lausanne, Switzerland, 2025
    """
    # parse input
    if rdout is None:
        help(compute_dose_from_readout)
        print("No readout provided. Fallback to example.")
        rdout = dict({
            'R1': {'value': 1.05, 'background': 0.1, 'uncert': 0.05},
            'R2': {'value': 1.00, 'background': 0.1, 'uncert': 0.05},
            'R3': {'value': 1.00, 'background': 0.1, 'uncert': 0.05},
            'R4': {'value': 2.00, 'background': 0.1, 'uncert': 0.05}
        })
        print(rdout)

    if not (
            isinstance(cfg, dict)
            or isinstance(cfg, DoseCalcConfig)
    ):
        print("No configuration provided. Fallback to defaults.")
        cfg = DoseCalcConfig(selection='mixed')
    else:
        if isinstance(cfg, dict):
            cfg = DoseCalcConfig(config=cfg)

    # define effective photon energy threshold in MeV when to add the open window
    E_threshold = dict({
        'stage 2': 50e-3,
        'stage 3': 35e-3
    })

    # construct objects
    dm_characteristics = DosimeterCharacteristics(config=cfg)
    lut = LookupTable(config=cfg, dm_characteristics=dm_characteristics)
    dose = DoseCalculator(readout=rdout, lookup=lut, config=cfg)

    # compute photons energies for channel combinations
    dose.__compute_energy_of_channels__()

    # check whether photon effective energy can be computed
    valid = []
    for key in dose.ratio_measured:
        valid.append(dose.ratio_measured.get(key).get('validity'))

    if any(valid):
        # take user selected channels
        dose.compute_effective_energy()
    else:
        # fallback: take all available channels
        dose.compute_effective_energy(
            channels=['R1', 'R2', 'R3', 'R4'],
        )

    # 1) straight-forward computation of doses
    dose.compute_dose()

    # 2) optimization for low photon energy
    # --> explicitely activate beta window (also sensitive to beta radiation!)
    if (
            dose.effective_photon_energy.get(
                'value') <= E_threshold.get('stage 2')
            and 'R4' not in dose.config.channels.get('energy')
            and dose.config.optimization.get('add_beta_channel') is True
    ):

        # enable all channels
        if dose.config.verbose > 0:
            print('\n stage 2: Photon energy < {} MeV --> rerun with full filter set \n'.format(
                E_threshold.get('stage 2')))
        dose.compute_effective_energy(channels=['R1', 'R2', 'R3', 'R4'])

        # recompute dose
        dose.compute_dose()

    # 3) beta or low energy photon radiation detected
    # --> correct filter ratios by subtracting the possible beta contribution from the readout,
    # recompute photon energies and photon and beta dose equivalents
    if (
            bool(dose.Hp007_beta_highE.get('value'))
            or bool(dose.Hp007_beta_lowE.get('value'))
    ):

        # recompute ratios and their energies
        if dose.config.verbose > 0:
            print(
                '\n Beta radiation possibly present --> recomputation of energies and doses\n')
        dose.__compute_energy_of_channels__()
        dose.compute_effective_energy()

        # add beta window
        if (
                dose.effective_photon_energy.get('value') < E_threshold.get('stage 3')
                and 'R4' not in dose.config.channels.get('energy')
        ):
            if dose.config.verbose > 0:
                print('\n stage 2: Photon energy < {} MeV --> rerun with full filter set \n'.format(
                    E_threshold.get('stage 3')
                    ))
            # compute once again effective photon energy, this time with cleaned-up readout
            dose.compute_effective_energy(channels=['R1', 'R2', 'R3', 'R4'])

        # recompute dose
        dose.compute_dose()

    # 4) least-square optimization on effective photon energy and photon and beta doses (depreciated)
    if dose.config.optimization.get('lsq', False):
        dose_lsq = DoseLSQOptimizer(dose)
        dose_lsq.optimize()

        if not dose_lsq.lsq_results.success:
            print('LSQ-fitting did not converge. Do rollback of object.')

        return dose, dose_lsq
    else:
        # No least-square optimization done -> return None (recommended)
        return dose


##############################################################################

def apply_detection_limit(H: float, DL: float = 10e-3) -> float:
    """
    Compute expected dose, DL/sqrt(2), when below detection limit (DL).

    Syntax:
    ------
       apply_detection_limit(H, DL)

    Args:
    ----
        - H (float): dose value
        - DL (float): detection limit

    Output:
    ------
        expected dose value (float)


    Reference:
    ---------
        Richard W. Hornung & Laurence D. Reed (1990) "Estimation of Average
        Concentration in the Presence of Nondetectable Values", Applied Occupational and
        Environmental Hygiene, 5:1, 46-51, DOI: 10.1080/1047322X.1990.10389587

    @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                University hospital Lausanne, Switzerland, 2025
    """
    return float(max(H, DL / np.sqrt(2)))

##############################################################################


def energy_weighting_factor(
        val: NDArray[np.float64], uc: NDArray[np.float64], method: str = 'loglin'
) -> NDArray[np.float64]:
    """
    Compute weighting factors of filter energies.

    Syntax:
    ------
       w = energy_weighting_factor(val, uc, method='loglin')

    Args:
    ----
        - val (NumPy array): vector of energies
        - uc (NumPy array): vector of energy uncertainties

    Output:
    ------
        - w (NumPy array): to unity normalized weighting factors

    @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                University hospital Lausanne, Switzerland, 2025
    """
    if method == 'standard':

        # standard approach to compute weights = 1/sigma^2
        w = uc ** (-2)

    elif method == 'loglin':

        # relative sigma/value normalized to log10-base
        w = (uc / val / np.log(10)) ** (-2)

    else:
        raise ValueError('Unknown method to calculate weights.')

    w_sum = np.sum(w)
    if w_sum > 0:
        w /= w_sum
    else:
        print('Sum of weights equal zero. Return equal weights')
        w = np.ones_like(uc)
        w /= np.sum(w)

    return w

##############################################################################


def find_nearest(array: list, value: float, side: str = 'left') -> int:
    """
    Find index of the vector element being closest to a given value.

    Syntax:
    ------
       idx_nearest = find_nearest(array, value, side='left')

    Args:
    ----
        - array (list or numpy.array): data vector
        - value (single value): reference value
        - side (string 'left' or 'right'): chose between left- or right-handed
          closest value
    Output:
    ------
        Index (int) for array/list

    @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                University hospital Lausanne, Switzerland, 2025
    """
    array = np.array(array)
    idx_sorted = np.argsort(array)
    sorted_array = np.array(array[idx_sorted])
    idx = np.searchsorted(sorted_array, value, side=side)
    if idx >= len(array):
        idx_nearest = idx_sorted[len(array) - 1]
    elif idx == 0:
        idx_nearest = idx_sorted[0]
    else:
        if abs(value - sorted_array[idx - 1]) < abs(value - sorted_array[idx]):
            idx_nearest = idx_sorted[idx - 1]
        else:
            idx_nearest = idx_sorted[idx]

    return idx_nearest

##############################################################################


def weighted_median(data: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Calculate weighted median.

    Syntax:
    ------
      w_median = weighted_median(data, weights)

    Args:
    ----
      data (list or numpy.array): data
      weights (list or numpy.array): weights

    @Author:  Jack Peterson (jack@tinybike.net)
    """
    data, weights = np.array(data).squeeze(), np.array(weights).squeeze()
    s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
    midpoint = 0.5 * sum(s_weights)
    if any(weights > midpoint):
        w_median = (data[weights == np.max(weights)])[0]
    else:
        cs_weights = np.cumsum(s_weights)
        idx = np.where(cs_weights <= midpoint)[0][-1]
        if cs_weights[idx] == midpoint:
            w_median = np.mean(s_data[idx:idx + 2])
        else:
            w_median = s_data[idx + 1]
    return w_median

##############################################################################


def scoring_mean(data: NDArray[np.float64], weights: NDArray[np.float64]) -> float:
    """
    Calculate the mean according to the scoring algorithm.

    Syntax:
    ------
      res = scoring_mean(data, weights)

    Args:
    ----
      data (numpy.array): data
      weights (numpy.array): weights

    Output:
    ------
        average (float)

    Reference:
    ---------
        R. Fagin, E. L. Wimmers in Theoretical Computer Science 239 (2000),
        https://doi.org/10.1016/S0304-3975(99)00224-8

    @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                University hospital Lausanne, Switzerland, 2025
    """
    val = 0.0

    for i in range(len(weights)):

        # for i<m
        # f += i*(w_{i} - w_{i+1})*f(x_{1},..., x_{i+1}) (with i>=1)
        if i < len(weights) - 1:
            val_i = (i + 1) * (weights[i] - weights[i + 1]) * np.average(data[:i + 2], weights=weights[:i + 2])
            val += val_i

        # for i=m
        # f += i*(w_{m})*f(x_{1},..., x_{m}) (with i>=1)
        else:
            val_i = (i + 1) * weights[i] * np.average(data, weights=weights)
            val += val_i

    return float(val)

##############################################################################


def fastround(arr: NDArray[np.float64], decimals: int = 4) -> NDArray[np.float64]:
    """
    Fast rounding (essentially NumPy function).

    Syntax:
    ------
        a_round = fastround(arr, decimals=4)

    Args:
    ----
        - a (NumPy array): vector of numbers to be round
        - decimals (int): round precision

    Output:
    ------
        rounded value(s) (NumPy array)

    @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                University hospital Lausanne, Switzerland, 2025
    """
    # factor = 10**(decimals + 1)
    # return np.floor(arr * (decimals+1)) / (decimals+1)                          # 1.93 μs ± 23.6 ns per loop
    # return np.int64(np.array(arr)*(10**decimals))/(10.**decimals)               # 2.36 μs ± 51.5 ns per loop
    return np.round(arr, decimals)  # 2.09 μs ± 87.7 ns per loop

##############################################################################


def SD_from_CI(val: float, val_UB: float, val_LB: float, dist: str = 'half-cosine') -> float:
    """
    Compute std. dev. from confidence interval depending on the underlying distribution function.

    Syntax:
    ------
       stdev = uncert_from_conf_interval(val, val_UB, val_LB, dist='half-cosine')

    Args:
    ----
        - val (float): estimated value
        - val_UB (float): upper boundary of estimated value
        - val_LB (float): lower boundary of estimated value
        - dist (str): distribution function
            - 'cosine'
            - 'half-cosine' (default)
            - 'quadratic'
            or with appendix '_LB' or '_UB' for one-sided confidence interval

    Reference:
    ---------
      H. Castrup's talk on
      "Selecting and applying error distribution functions"
      presented at the Measurement Science Conference, Anaheim, 2004
      http://www.isgmax.com/Articles_Papers/Selecting%20and%20Applying%20Error%20Distributions.pdf

    @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                University hospital Lausanne, Switzerland, 2025
    """
    # adapt values to symmetry
    if '_LB' in dist:
        # from lower boundary to middle value times 2
        val2, val1 = val, val_LB
        fct = 2.0

    elif '_UB' in dist:
        # from upper boundary to middle value times 2
        val2, val1 = val_UB, val
        fct = 1.0

    else:
        # from upper to lower boundary
        val2, val1 = val_UB, val_LB
        fct = 1.0

    # calculate uncertainty
    if dist in ('half-cosine', 'half-cosine_LB', 'half-cosine_UB'):

        # half-cosine distribution
        SD = 0.435236 * fct * np.abs(val2 - val1)

    elif dist in ('cosine', 'cosine_LB', 'cosine_UB'):

        # cosine distribution
        SD = 0.361512 * fct * np.abs(val2 - val1)

    elif dist in ('quadratic', 'quadratic_LB', 'quadratic_UB'):

        # quadratic distribution
        SD = 0.447214 * fct * np.abs(val2 - val1)

    else:
        raise ValueError('Unknown distribution function.')

    return float(SD)

##############################################################################


def get_hpK(icru_version: int = 57) -> dict:
    """
    Provide hpK coefficients for photon effective energy depending on ICRU version.

    The default air kerma to dose equivalent conversion coefficients are
    based on ISO 4037-3:2019 (ICRU 57 and publications of Ankerhold in
    2000 and 2007)
    The ICRU 95 coefficients are from the publication of Behrens et al. from
    2022, Journal of Radiological Protection:
    https://doi.org/10.1088/1361-6498/abc860

    Syntax:
    ------
        get_hpK(icru_version = 57)

    Arg.:
    ----
        icru_version (int): publication 57 or 93

    Output:
    ------
        (dict) hpK values

    @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                University hospital Lausanne, Switzerland, 2025
    """
    # init.
    beam = ('N-15', 'N-20', 'N-25', 'N-30', 'N-40',
            'N-60', 'N-80', 'N-100', 'N-120', 'N-150',
            'N-200', 'N-250', 'N-300', 'N-350', 'N-400',
            '137Cs', '60Co', 'R-C', 'R-F')

    # select air kerma to dose equivalent coefficients
    if icru_version == 57:
        energy = (0.012, 0.017, 0.020, 0.026,
                  0.033, 0.048, 0.065, 0.083, 0.100,
                  0.118, 0.161, 0.207, 0.248, 0.288,
                  0.328, 0.662, 1.253, 4.400, 6.130)
        hpK10 = (0.060, 0.333, 0.583, 0.819,
                 1.210, 1.660, 1.890, 1.880, 1.800,
                 1.720, 1.560, 1.480, 1.420, 1.380,
                 1.350, 1.210, 1.150, 1.110, 1.100)
        hpK007 = (0.970, 0.990, 1.040, 1.110,
                  1.280, 1.560, 1.720, 1.720, 1.660,
                  1.600, 1.490, 1.420, 1.380, 1.340,
                  1.320, 1.210, 1.170, 1.110, 1.100)
        # hpK(3) for a cylinder phantom (ISO 4037-3:2019)
        # hpK3 =      (0.420, 0.670, 0.880, 1.040,
        #              1.280, 1.540, 1.660, 1.630, 1.580,
        #              1.520, 1.420, 1.360, 1.320, 1.290,
        #              1.270, 1.180, 1.140, 1.100, 1.090)
        #
        # hpK(3) for a slab phantom
        #   (Behrens, Radiation Prot. Dosimetry 2011, doi:10.1093/rpd/ncq459)
        hpK3 = (0.420, 0.660, 0.880, 1.040,
                1.290, 1.630, 1.800, 1.810, 1.740,
                1.660, 1.530, 1.460, 1.410, 1.370,
                1.330, 1.220, 1.160, 1.120, 1.120)

    elif icru_version == 93:
        # Behrens & Otto, J. Radiol. Prot. 42 (2022)
        energy = (0.0124, 0.0163, 0.0203, 0.0246, 0.0333,
                  0.0479, 0.0652, 0.0833, 0.1004, 0.1182,
                  0.1648, 0.2073, 0.2484, 0.6390, 1.1979)
        hpK10 = (0.0219, 0.0611, 0.1298, 0.2357, 0.5222,
                 1.0139, 1.3639, 1.4256, 1.3933, 1.3369,
                 1.2148, 1.1514, 1.1164, 1.0158, 0.9971)
        hpK007 = (0.9876, 1.0292, 1.0862, 1.1622, 1.3445,
                  1.6175, 1.7693, 1.7410, 1.6752, 1.6088,
                  1.4927, 1.4280, 1.3787, 1.2136, 1.1606)
        hpK3 = (0.2723, 0.5549, 0.7805, 0.9523, 1.2040,
                1.4564, 1.5685, 1.5560, 1.5032, 1.4486,
                1.3645, 1.3184, 1.2907, 1.1712, 1.1276)

    else:
        raise ValueError('Invalid ICRU version')

    # define output dictionary
    return {
        'beam': beam,
        'energy': energy,
        'hpK10': hpK10,
        'hpK007': hpK007,
        'hpK3': hpK3,
    }


##############################################################################
# Class DoseCalcConfig
##############################################################################


class DoseCalcConfig():
    """
    Class providing a configuration for the dose calculation.

    Syntax:
    ------
        config = DoseCalcConfig(selection = 'mixed')

    Args:
    ----
        - 'config' (object): output from call: 'config = DoseCalcConfig()'
        - 'selection' (str): choice of config according to:
            - 'mixed' for mixed radiation field (photons and betas) or
            - 'photons' for a pure photon field or the usage of the open window
              for low energy photons or
            - 'betas' for Sr90 and 85Kr only (not yet supported)

    @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                University hospital Lausanne, Switzerland, 2025
    """

    def __init__(self, **kwargs):

        # default class variables
        self.verbose = 1  # display only most important information
        self.plot = False  # plot results, if applicable
        list_selection = ('default', 'mixed', 'photons', 'betas')
        selection = 'mixed'  # run for mixed radiation field by default
        config = None

        # parse function arguments
        if len(kwargs) > 0:
            for key in kwargs:

                # preset of radiation field
                if key == 'selection':
                    if isinstance(kwargs[key], str):
                        if kwargs[key] in list_selection:
                            selection = kwargs[key]
                        else:
                            raise ValueError(
                                "Selection must be part of: {}".format(list_selection))
                    else:
                        raise ValueError("Selection must be a string.")

                # preset or user settings for function call
                if key in ('config', 'cfg'):
                    if isinstance(kwargs[key], (dict, DoseCalcConfig)):
                        config = kwargs[key]
                    else:
                        raise ValueError("Config needs to be a dictionary.")

        else:
            # No input provided -> display help
            help(DoseCalcConfig)
            return

        # version of hpK conversion factors
        # ICRU 57 (current standard of ISO 4037-3) or 93
        self.icru_version = 57

        # energy-axis for interpolation of response and hpK
        self.energy_axis = dict({
            # density of energy resolution 'full' or 'calibration' (N-series, 137Cs and 60Co)
            'selection': 'full',
            # vector in MeV or number of axis elements (100 by default)
            'elements': 100,
        })

        # response normalization: choices are 'Hp10', 'Hp007' or 'kerma'
        self.normalization = 'Hp007'  # best: 'Hp007' (default)

        # interp. methods: 'nearest', 'linear' (default), 'cubic'
        self.interp_method = dict({
            'response': {'interpolate': True, 'method': 'linear'},
            'ratio':    {'interpolate': True, 'method': 'linear'},
        })

        # methods for photon energy calculation:
        self.calc_method = dict({
            'energy':   'scoring',  # weighted, scoring, logic or logic2 or logic2-scoring, harmonic, geometric
            'dose':     'weighted',  # 'weighted'
        })

        # criteria for beta dose calculation
        self.betas = dict({
            '90Sr/90Y': {
                'seek': True,
                'calc_dose': True,
                'include_dose': True
            },
            '85Kr': {
                'seek': True,
                'calc_dose': True,
                'include_dose': False
            }
        })

        # fiter ratios used by ratio method
        self.ratios = ('R1/R4', 'R3/R4', 'R2/R4', 'R2/R1', 'R3/R1', 'R2/R3')

        # optimizations True/False
        self.optimization = dict({
            # (recommended) if required, add channel specific to beta/low energy photons
            'add_beta_channel':         True,
            # solving a linear system of equations for improving Eeff
            'energy_postprocessing':    True,
            # solving a SOLE for improving the dose values (True/False or threshold Eeff ~ 0.15 MeV)
            # note: at Hp(10)=1 mSv and Hp(0.07, 90Sr)=1 mSv, select Eeff threshold or Hp(0.07, 137Cs) is off by ~15%
            'dose_postprocessing':      0.15,
            # (NOT recommended) final results polishing with help of least-square-fitting -> bad convergence
            'lsq':                      False,
        })

        if selection in ('default', 'mixed'):
            # mixed field by default
            self.channels = dict({
                # take all except R4 (betas possible)
                'energy':   ['R1', 'R2', 'R3'],
                'dose':     ['R1', 'R3'],  # R1 req., R2 & R3 opt., avoid R4
            })

        elif selection in ('photons'):
            # Specific settings for a pure photon field
            self.channels = dict({
                'energy':   ['R1', 'R2', 'R3', 'R4'],  # take all (no betas)
                'dose':     ['R1', 'R3', 'R4'],
            })

        elif selection in ('betas', '90Sr/90Y'):
            # specific settings for a beta radiation field
            self.channels = dict({
                'energy':   ['R1', 'R4'],  # take only betas sensitive channels
                'dose':     ['R4'],
            })

        elif selection in ('85Kr'):
            # specific settings for a 85Kr radiation field
            self.channels = dict({
                'energy':   ['R4'],
                'dose':     ['R4'],
            })

        else:
            raise ValueError("Radiation field selection not supported.")

        # if necessary, update with user input
        if isinstance(config, dict):
            for key, val in config.items():
                if isinstance(key, (list, tuple)):
                    setattr(self, str(key), [x if isinstance(x, dict) else x for x in val])
                else:
                    setattr(self, key, val if isinstance(val, dict) else val)

        return

##############################################################################
# Class DosimeterCharacteristics
##############################################################################


class DosimeterCharacteristics(DoseCalcConfig):
    """
    Class providing the dosimeter properties as response and filter ratios.

    Syntax:
    ------
        dm_characteristics = DosimeterCharacteristics(config = None)

    Args:
    ----
        config: return object from call: 'config = DoseCalcConfig()'

    @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                University hospital Lausanne, Switzerland, 2025
    """

    def __init__(self, config=None, **kwargs):

        # super().__init__(**kwargs)
        if not isinstance(config, DoseCalcConfig):
            print("Default configuration for a mixed radiation field is used.")
            config = DoseCalcConfig(selection='mixed')

        # default class variables
        self.response_photons = dict()
        self.response_betas = dict()
        self.ratio_photons = dict()
        self.ratio_betas = dict()
        self.ratio_lincomb_betas = dict()
        self.ratio_lincomb_photons = dict()
        self.label = dict()
        self.config = config

        # detection limit of each filter in mSv
        # (according to https://doi.org/10.1016/j.radmeas.2024.107346)
        self.detection_limit = {
            'R1': 15e-3,
            'R2': 15e-3,
            'R3': 15e-3,
            'R4': 15e-3
        }

        # init. dosimeter response
        self.__dosimeter_response__()

        # compute ratios
        self.__compute_ratios__(config)

        return

    ##############################################################################

    def __dosimeter_response__(self):
        """
        Provide filter response of dosimeter (for ICRU 57 and 93).

        Syntax:
        ------
            self.__dosimeter_response__(self)

        @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                    University hospital Lausanne, Switzerland, 2025
        """
        # check available dosimeter versions
        supported_icru_versions = (57, 93)
        version = self.config.icru_version
        if version not in supported_icru_versions:
            raise ValueError('Unsupported ICRU version!')

        if version == 57:

            # element response in terms of air kerma
            self.response_photons = {
                'beam': ('N-15', 'N-20', 'N-25', 'N-30', 'N-40',
                         'N-60', 'N-80', 'N-100', 'N-120', 'N-150',
                         'N-200', 'N-250', 'N-300', '137Cs', '60Co'),
                'theta': (0., 0., 0., 0., 0.,
                          0., 0., 0., 0., 0.,
                          0., 0., 0., 0., 0.),
                'energy': (0.012, 0.017, 0.020, 0.026, 0.033,
                           0.048, 0.065, 0.083, 0.100, 0.118,
                           0.161, 0.207, 0.248, 0.662, 1.253),
                'channels': ('R1', 'R2', 'R3', 'R4'),
                'kerma': {
                    'R1': (0.100, 0.267, 0.465, 0.622, 0.828,
                           1.037, 1.126, 1.110, 1.100, 1.099,
                           1.092, 1.058, 1.034, 1.000, 0.966),
                    'R2': (0.001, 0.005, 0.015, 0.028, 0.062,
                           0.116, 0.173, 0.313, 0.496, 0.696,
                           0.969, 1.069, 1.077, 1.000, 0.900),
                    'R3': (0.002, 0.008, 0.021, 0.039, 0.095,
                           0.357, 0.744, 0.979, 1.089, 1.143,
                           1.170, 1.139, 1.093, 1.000, 0.941),
                    'R4': (0.410, 0.527, 0.646, 0.734, 0.873,
                           1.032, 1.102, 1.090, 1.073, 1.090,
                           1.078, 1.063, 1.039, 1.000, 0.962),
                },
                'Hp10': {
                    'R1': (2.019, 0.970, 0.966, 0.918, 0.828,
                           0.756, 0.721, 0.715, 0.740, 0.773,
                           0.847, 0.865, 0.881, 1.000, 1.016),
                    'R2': (0.026, 0.019, 0.032, 0.041, 0.062,
                           0.085, 0.111, 0.202, 0.334, 0.490,
                           0.752, 0.874, 0.918, 1.000, 0.947),
                    'R3': (0.042, 0.029, 0.044, 0.057, 0.095,
                           0.260, 0.477, 0.630, 0.732, 0.804,
                           0.908, 0.931, 0.931, 1.000, 0.990),
                    'R4': (8.268, 1.915, 1.341, 1.084, 0.873,
                           0.752, 0.705, 0.702, 0.721, 0.767,
                           0.836, 0.869, 0.885, 1.000, 1.012),
                },
                'Hp007': {
                    'R1': (0.125, 0.326, 0.542, 0.677, 0.783,
                           0.804, 0.792, 0.781, 0.802, 0.831,
                           0.887, 0.902, 0.906, 1.000, 0.999),
                    'R2': (0.002, 0.007, 0.018, 0.031, 0.059,
                           0.090, 0.122, 0.220, 0.362, 0.526,
                           0.787, 0.911, 0.944, 1.000, 0.930),
                    'R3': (0.003, 0.010, 0.025, 0.042, 0.089,
                           0.277, 0.524, 0.689, 0.794, 0.864,
                           0.950, 0.970, 0.958, 1.000, 0.973),
                    'R4': (0.511, 0.644, 0.752, 0.800, 0.825,
                           0.800, 0.775, 0.767, 0.782, 0.825,
                           0.876, 0.905, 0.911, 1.000, 0.995),
                },
                'label': {
                    'R1': 'R1 (1.35mm PTFE)',
                    'R2': 'R2 (1.2mm Sn)',
                    'R3': 'R3 (0.5mm Cu)',
                    'R4': 'R4 (0.4mm ABS)'
                }
            }

            # element response Hp(0.07) for 90Sr/90Y normalized to response
            # to 137Cs in terms of Hp(10) etc.
            self.response_betas = {
                'beam': ('90Sr/90Y', '85Kr'),
                'theta': (0., 0.),
                'energy': (0.8, 0.25),
                'channels': ('R1', 'R2', 'R3', 'R4'),
                'kerma': {
                    'R1': (0.080, 0.001),
                    'R2': (0.004, 0.001),
                    'R3': (0.006, 0.001),
                    'R4': (1.245, 0.122)
                },
                'Hp007': {
                    'R1': (0.066, 0.001),
                    'R2': (0.003, 0.001),
                    'R3': (0.005, 0.001),
                    'R4': (1.029, 0.100)
                },
                'Hp10': {
                    'R1': (0.066, 0.001),
                    'R2': (0.003, 0.001),
                    'R3': (0.005, 0.001),
                    'R4': (1.029, 0.100)
                },
                'label': {
                    'R1': 'R1 (1.35mm PTFE)',
                    'R2': 'R2 (1.2mm Sn)',
                    'R3': 'R3 (0.5mm Cu)',
                    'R4': 'R4 (0.4mm ABS)'
                }
            }

        elif version == 93:
            # new definitions according to publication:
            # https://doi.org/10.1088/1361-6498/abc860
            # photon energy in MeV

            self.response_photons = {
                'beam': ('N-15', 'N-20', 'N-25', 'N-30', 'N-40',
                         'N-60', 'N-80', 'N-100', 'N-120', 'N-150',
                         'N-200', 'N-250', 'N-300', '137Cs', '60Co'),
                'theta': (0., 0., 0., 0., 0.,
                          0., 0., 0., 0., 0.,
                          0., 0., 0., 0., 0.),
                'energy': (0.0124, 0.0163, 0.0203, 0.0246, 0.0333,
                           0.0479, 0.0652, 0.0833, 0.1004, 0.1182,
                           0.1648, 0.2073, 0.2484, 0.6390, 1.1979),
                'channels': ('R1', 'R2', 'R3', 'R4'),
                'kerma': {
                    'R1': (0.100, 0.267, 0.465, 0.621, 0.828,
                           1.037, 1.126, 1.110, 1.100, 1.099,
                           1.092, 1.058, 1.034, 1.000, 0.966),
                    'R2': (0.001, 0.005, 0.015, 0.028, 0.062,
                           0.116, 0.173, 0.313, 0.497, 0.696,
                           0.969, 1.069, 1.077, 1.000, 0.900),
                    'R3': (0.002, 0.008, 0.021, 0.039, 0.095,
                           0.357, 0.744, 0.980, 1.090, 1.143,
                           1.170, 1.138, 1.093, 1.000, 0.941),
                    'R4': (0.410, 0.527, 0.646, 0.733, 0.873,
                           1.032, 1.102, 1.090, 1.073, 1.090,
                           1.078, 1.063, 1.039, 1.000, 0.962)
                },
                'Hp10': {
                    'R1': (4.643, 4.437, 3.643, 2.678, 1.610,
                           1.039, 0.839, 0.791, 0.802, 0.835,
                           0.913, 0.934, 0.941, 1.000, 0.984),
                    'R2': (0.061, 0.089, 0.119, 0.121, 0.121,
                           0.117, 0.129, 0.223, 0.362, 0.529,
                           0.810, 0.943, 0.980, 1.00, 0.917),
                    'R3': (0.097, 0.130, 0.165, 0.168, 0.184,
                           0.357, 0.554, 0.698, 0.794, 0.868,
                           0.979, 1.004, 0.994, 1.000, 0.958),
                    'R4': (19.016, 8.761, 5.058, 3.161, 1.697,
                           1.034, 0.820, 0.777, 0.782, 0.829,
                           0.902, 0.937, 0.945, 1.000, 0.980),
                },
                'Hp007': {
                    'R1': (0.123, 0.315, 0.520, 0.649, 0.747,
                           0.778, 0.773, 0.774, 0.797, 0.829,
                           0.888, 0.899, 0.910, 1.000, 1.010),
                    'R2': (0.002, 0.006, 0.017, 0.029, 0.056,
                           0.087, 0.119, 0.218, 0.360, 0.525,
                           0.788, 0.909, 0.948, 1.000, 0.941),
                    'R3': (0.003, 0.009, 0.024, 0.041, 0.085,
                           0.268, 0.511, 0.683, 0.789, 0.862,
                           0.952, 0.967, 0.962, 1.000, 0.984),
                    'R4': (0.504, 0.621, 0.722, 0.766, 0.788,
                           0.774, 0.756, 0.760, 0.777, 0.823,
                           0.877, 0.903, 0.914, 1.000, 1.006),
                },
                'label': {
                    'R1': 'R1 (1.35mm PTFE)',
                    'R2': 'R2 (1.2mm Sn)',
                    'R3': 'R3 (0.5mm Cu)',
                    'R4': 'R4 (0.4mm ABS)'
                }
            }

            # element response Hp(0.07) for 90Sr/90Y normalized to response
            # to 137Cs in terms of Hp(10) etc.
            self.response_betas = {
                'beam': ('90Sr/90Y', '85Kr'),
                'theta': (0., 0.),
                'energy': (0.8, 0.25),
                'channels': ('R1', 'R2', 'R3', 'R4'),
                'kerma': {
                    'R1': (0.066, 0.001),
                    'R2': (0.003, 0.001),
                    'R3': (0.005, 0.001),
                    'R4': (1.029, 0.100)
                },
                'Hp007': {
                    'R1': (0.066, 0.001),
                    'R2': (0.003, 0.001),
                    'R3': (0.005, 0.001),
                    'R4': (1.029, 0.100)
                },
                'Hp10': {
                    'R1': (0.066, 0.001),
                    'R2': (0.003, 0.001),
                    'R3': (0.005, 0.001),
                    'R4': (1.029, 0.100)
                },
                'label': {
                    'R1': 'R1 (1.35mm PTFE)',
                    'R2': 'R2 (1.2mm Sn)',
                    'R3': 'R3 (0.5mm Cu)',
                    'R4': 'R4 (0.4mm ABS)'
                }
            }

        else:
            raise ValueError('Unsupported ICRU version')

        return

    ##############################################################################

    def __compute_ratios__(self, config, ratios=None):
        """
        Compute the filter ratios of the dosimeter.

        Syntax:
        ------
            self.__compute_ratios__(self, **kwargs)

        Args:
        ----
            ratios (tuple or list): filter ratios to be computed,
                    e.g. ('R1/R4', 'R3/R4', ...)

        @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                    University hospital Lausanne, Switzerland, 2025
        """
        # init.
        if ratios is None:
            ratios = config.ratios
        elif ~isinstance(ratios, tuple):
            raise ValueError("Input 'ratios' needs to be a tuple.")

        norm = 'kerma'

        # if necessary, display ratios to be computed
        if config.verbose > 1:
            print('Compute filter ratios: {}'.format(ratios))

        # compute filter ratios
        for key in ratios:

            # photons
            Ri = np.array(self.response_photons.get(norm).get(key[0:2]))
            Rj = np.array(self.response_photons.get(norm).get(key[3:5]))
            self.ratio_photons.update({key: fastround(Ri / Rj, decimals=4)})

            # betas
            Ri_b = np.array(self.response_betas.get(norm).get(key[0:2]))
            Rj_b = np.array(self.response_betas.get(norm).get(key[3:5]))
            self.ratio_betas.update({key: fastround(Ri_b / Rj_b, decimals=4)})

        # compute linear combination of channels for betas only
        R1_b = np.array(self.response_betas.get(norm).get('R1'))
        R4_b = np.array(self.response_betas.get(norm).get('R4'))

        # store linear combinations (used for determining betas) to object
        self.ratio_lincomb_betas.update({
            '(R4-R1)/R4': fastround((R4_b - R1_b) / R4_b, 4),
            '(R4-R1)/R1': fastround((R4_b - R1_b) / R1_b, 4)
        })

        # do the same for photons, but store it into the beta branch
        # it's not required for the calculation of the photon effective energy
        R1_ph = np.array(self.response_photons.get(norm).get('R1'))
        R4_ph = np.array(self.response_photons.get(norm).get('R4'))

        self.ratio_lincomb_photons.update({
            '(R4-R1)/R4': fastround((R4_ph - R1_ph) / R4_ph, 4),
            '(R4-R1)/R1': fastround((R4_ph - R1_ph) / R1_ph, 4)
        })

        return


##############################################################################
# Class LookupTable
##############################################################################


class LookupTable(DosimeterCharacteristics):
    """
    Class building the lookup table for subsequent dose calculation.

    @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                University hospital Lausanne, Switzerland, 2025
    """

    # initialize class
    def __init__(self, **kwargs):

        # photons
        self.response_photons = dict()
        self.response_photons_deriv = dict()
        self.normalization = None
        self.ratio_photons = dict()
        self.ratio_photons_deriv = dict()
        self.hpK = dict()

        # betas
        self.response_betas = dict()
        self.ratio_betas = dict()

        # misc.
        cfg = object()

        # function arguments
        for key in kwargs:
            if key == 'config':
                cfg = kwargs[key]

        # transform any user config dictionary into its object
        if isinstance(cfg, dict):
            cfg = DoseCalcConfig(config=cfg)

        # if necessary, fall back to defaults
        elif not isinstance(cfg, DoseCalcConfig):
            print(
                """
                No configuration given for building the lookup-table.
                Fallback to defaults."""
            )
            cfg = DoseCalcConfig(selection='default')

        # copy normalization information
        self.normalization = cfg.normalization

        # retrieve dosimeter characteristics
        dm_characteristics = DosimeterCharacteristics(config=cfg)

        # preserve inputs
        self.config = cfg
        self.dm_characteristics = dm_characteristics

        # photons: interpolate filter response on finer energy grid
        self.__define_energy_axis__()
        self.__interpolate_hpK__()
        self.__interpolation__()

        return

    ##############################################################################

    def __define_energy_axis__(self):
        """
        Define a dense energy axis if the axis is not specified by the user.

        Syntax:
            self.__define_energy_axis__()

        @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                    University hospital Lausanne, Switzerland, 2025
        """
        # init.
        E_calib = np.array(self.dm_characteristics.response_photons.get('energy')).round(decimals=4)

        # define energy axis
        if self.config.energy_axis.get('selection') == 'calibration':

            # use discrete energy values from calibration data,
            # i.e. N-beams, 137Cs and 60Co
            energy = np.array(
                self.dm_characteristics.response_photons.get('energy'))

        elif self.config.energy_axis.get('selection') == 'full':

            # user choice: either list or integer
            elements = self.config.energy_axis.get('elements')

            if isinstance(elements, (tuple, list)):

                # if a vector is provided by the user
                E_fine = np.array(elements)

            elif isinstance(elements, int):

                # if the number of discrete energy values is given
                E_fine = 10 ** np.linspace(
                    start=np.log10(E_calib.min()),
                    stop=np.log10(E_calib.max()),
                    num=elements
                ).round(decimals=4)
            else:

                # default
                E_fine = 10 ** np.linspace(
                    start=np.log10(E_calib.min()),
                    stop=np.log10(E_calib.max()),
                    num=100
                ).round(decimals=4)

            # build axis and avoid doubles
            energy = fastround(np.concatenate((E_calib, E_fine), axis=None), decimals=4)
            energy = np.unique(energy)

        else:
            raise ValueError(
                "Unknown energy axis selected for building lookup table")

        # store to class
        self.response_photons.update({'energy': energy})

        return

    ##############################################################################

    def __interpolate_hpK__(self):
        """
        Interpolates hpK coefficents.

        Interpolation is set to 'linear' as hpK=F(log10(E)) for a pre-defined
        energy axis provided by the class itself.
        The default air kerma to dose equivalent conversion coefficients,
        Hp(d)/Ka at 0°, are based on ISO 4037-3:2019 (ICRU 57 and publications
        of Ankerhold in 2000 and 2007).

        Syntax:
        ------
            self.__interpolate_hpK__()

        @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                    University hospital Lausanne, Switzerland, 2025
        """
        # load hpK coefficients
        hpK = get_hpK(self.config.icru_version)

        # axis on which we want to interpolate hpK
        xi = np.log10(self.response_photons.get('energy'))

        for key in ('hpK10', 'hpK007', 'hpK3'):

            # get data for interpolation
            x = np.log10(hpK.get('energy'))
            y = np.array(hpK.get(key))

            # do interpolation (k=1: linear, k=3: cubic spline)
            if self.config.normalization in ('Hp007', 'Hp10', 'Hp3', 'kerma'):
                tck = interpolate.splrep(x, y, k=1)
                yi = interpolate.splev(xi, tck)
            elif self.config.normalization in ('test'):
                # for testing: does not work
                tck = interpolate.splrep(x, np.log10(y), k=1)
                yi = interpolate.splev(xi, tck)
            else:
                raise ValueError("Only 'kerma', 'Hp007' or 'Hp10' are supported.")

            yi = fastround(yi, decimals=4)
            yi[np.where(yi < 0)] = 0.0

            # store to lookup table
            self.hpK.update({key: yi})

        # store also energy axis
        self.hpK.update({'energy': self.response_photons.get('energy')})

        return

    ##############################################################################

    def __interpolation__(self):
        """
        Interpolation of detector response and response ratios to get smooth function.

        Syntax:
        ------
            self.__interpolation__()

        @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                    University hospital Lausanne, Switzerland, 2025
        """
        # init. interpolation method
        method = self.config.interp_method

        # check methods
        for key in method.keys():
            if method.get(key).get('interpolate') is True:
                if method.get(key).get('method') is None:
                    method.get(key).update({'method': 'linear'})

        # copy dictionary structure to lookup table
        # self.response_photons = self.dm_characteristics.response_photons
        # self.ratios_photons = self.dm_characteristics.ratios_photons

        # shortcut to choosen normalization
        norm = self.config.normalization

        ################################
        # Photons
        ################################

        # loop over response, ratio
        for key_norm in ('Hp10', 'Hp007', 'kerma'):
            for key_dat in method.keys():

                # define x-axis: photon energy
                x_in = self.dm_characteristics.response_photons.get('energy')
                x_out = self.response_photons.get('energy')

                # define y-axis: response or ratio
                if key_dat == 'response':

                    y_dict = self.dm_characteristics.response_photons.get(key_norm)
                    self.response_photons.update({key_norm: dict()})
                    self.response_photons_deriv.update({key_norm: dict()})

                elif key_dat == 'ratio':

                    y_dict = self.dm_characteristics.ratio_photons
                    self.ratio_photons.update({key_norm: dict()})
                    self.ratio_photons_deriv.update({key_norm: dict()})

                else:
                    raise ValueError('Unknown photon data to interpolate')

                # if necessary, display information about interpolation
                if (
                        self.config.verbose > 1
                        and method.get(key_dat).get('interpolate')
                ):

                    print('LUT: interpolate {} with method ´{}´'.format(
                        key_dat,
                        method.get(key_dat).get('method')
                    ))

                # define interpolation method
                if method.get(key_dat).get('interpolate') is True:

                    if method[key_dat].get('method') == 'linear':

                        # linear interpolation
                        method[key_dat].update({'deg_spline': 1})

                    elif method[key_dat].get('method') == 'cubic':

                        # cubic interpolation (may produce wiggles)
                        method[key_dat].update({'deg_spline': 3})

                    elif method[key_dat].get('method') == 'nearest':

                        # nearest value
                        method[key_dat].update({'deg_spline': 1})
                        x_out_nearest = np.ones_like(x_out)
                        for idx, val in enumerate(x_out):
                            idx_nearest = np.abs(x_in - val).argmin()
                            x_out_nearest[idx] = x_in[idx_nearest]
                        x_out = x_out_nearest

                    else:
                        raise ValueError('Unknown interp. method')

                # loop over each channel (filter or ratio)
                for key_ch in y_dict.keys():

                    # get original y-axis
                    y_in = y_dict.get(key_ch)

                    if method.get(key_dat).get('interpolate') is True:

                        # interpolate hpK value depending on normalization choice
                        if self.config.normalization in ('Hp007', 'kerma'):

                            # interpolate in log10-log10 space
                            tck = interpolate.splrep(
                                np.log10(x_in),
                                np.log10(y_in),
                                k=method.get(key_dat).get('deg_spline'),
                                s=method.get(key_dat).get('smoothing')
                            )
                            y_out = 10 ** interpolate.splev(np.log10(x_out), tck)

                        elif self.config.normalization == 'Hp10':

                            # interpolate in log10-linear space
                            tck = interpolate.splrep(
                                np.log10(x_in),
                                y_in,
                                k=method.get(key_dat).get('deg_spline'),
                                s=method.get(key_dat).get('smoothing')
                            )
                            y_out = interpolate.splev(np.log10(x_out), tck)

                        else:
                            raise ValueError("Only supported normalizations: 'kerma', 'Hp007', 'Hp10'.")

                    else:

                        # do nothing and take input as it is
                        y_out = y_in
                        x_out = x_in

                    # get derivative (linear-linear space)
                    tck = interpolate.splrep(
                        x_out, y_out,
                        k=method.get(key_dat).get('deg_spline'),
                        s=method.get(key_dat).get('smoothing')
                    )
                    y_out_deriv = interpolate.splev(x_out, tck, der=1)

                    # save to object
                    y_out = fastround(y_out, decimals=4)
                    y_out_deriv = fastround(y_out_deriv, decimals=4)

                    if key_dat == 'response':
                        self.response_photons[key_norm].update({key_ch: y_out})
                        self.response_photons_deriv[key_norm].update({key_ch: y_out_deriv})

                    elif key_dat == 'ratio':
                        self.ratio_photons[key_norm].update({key_ch: y_out})
                        self.ratio_photons_deriv[key_norm].update({key_ch: y_out})

            ################################
            # Betas
            ################################
            # Note: no interpolation required, since we only distinguish beetween high and low beta energy

            # prepare dictionary
            self.response_betas.update({key_norm: dict()})

            for key_dat in method.keys():

                # init. dictionary
                if key_dat == 'response':
                    self.response_betas.update({'energy': self.dm_characteristics.response_betas.get('energy')})
                    self.response_betas.update({key_norm: dict()})
                    self.response_betas.update({
                        key_norm: self.dm_characteristics.response_betas.get(norm)
                        })

                elif key_dat == 'ratio':
                    self.ratio_betas.update({'energy': self.dm_characteristics.response_betas.get('energy')})
                    self.ratio_betas.update({key_norm: dict()})
                    self.ratio_betas.update({
                        key_norm: self.dm_characteristics.response_betas.get(norm)
                        })

                else:
                    raise ValueError('Unknown beta data to interpolate')

        return

##############################################################################
# Class DoseCalculator
##############################################################################


class DoseCalculator(LookupTable):
    """
    Class for dose calculation.

    @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                University hospital Lausanne, Switzerland, 2025
    """

    def __init__(self, **kwargs):

        # init.
        self._niter = 0
        self.energies_photons = dict()
        self.effective_photon_energy = dict()
        self.energies_beta = dict()
        self.lookup = dict()
        self.ratio_measured = dict()
        self.ratio_photons = dict()
        self.ratio_betas = dict()
        self.readout = dict()
        self.Hp10 = dict()
        self.Hp007 = dict()
        self.Hp007_ph = dict()
        self.Hp007_beta_highE = dict()
        self.Hp007_beta_lowE = dict()
        self.Hp3 = dict()
        self.Kerma = dict()
        self.hpK = dict()
        self.channels_used = dict()

        # parse function arguments
        for key in kwargs:
            if key == 'readout':
                self.readout = copy.deepcopy(kwargs[key])
            if key == 'lookup':
                self.lookup = kwargs[key]
            if key == 'config':
                if not isinstance(kwargs[key], DoseCalcConfig):
                    if isinstance(kwargs[key], dict):
                        self.config = DoseCalcConfig(kwargs[key])
                    else:
                        raise ValueError(
                            'Unknown format of user configuration')
                else:
                    self.config = kwargs[key]

        # init. effective energy
        if self.lookup.config.verbose > 1:
            print('Init. effective photon energy to 662 keV.')

        # set by default to 137Cs
        self.effective_photon_energy.update(
            {
                'value':    0.622,
                'uncert':   float(fastround(SD_from_CI(0.622, 0.25, 1.253, dist='half-cosine'), decimals=4)),
                'ci997':    [0.248, 1.253],
                'unit':     'MeV'
            }
        )
        self.effective_photon_energy.update(
            {
                'value':    None,
                'uncert':   None,
                'ci997':    [None, None],
                'unit':     'MeV'
            }
        )

        # set by default that presence of betas is unknown
        self.energies_beta.update({
            '90Sr/90Y':       'unknown',
            '85Kr':         'unknown'
        })

        # detection limit
        self.detection_limit = self.lookup.dm_characteristics.detection_limit

        # check lookup table
        if not bool(self.lookup):
            raise ValueError(
                """"
                Data source between lookup table and selection
                are not identical.
                """
            )

        # subtract background dose from readout
        self.__background_subtraction__()

        # validate dosimeter readout
        if not bool(self.readout):

            # check each channel for measured value
            for key in self.readout.keys():
                if not self.readout.get(key).get('value'):
                    raise ValueError('No readout available')

        else:

            # if readout contains values, check validity
            self.__readout_validation__()

        return

    ##############################################################################

    def __readout_validation__(self):
        """
        Validate the readout of the 4 filters in checking if its value is larger than the detection limit.

        Syntax:
        ------
            self.__readout_validation__(self)

        Args: ---
        ----

        @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                    University hospital Lausanne, Switzerland, 2025
        """
        # WB dosimeter myOSL 4.0:
        # loop over each filter value in opposite direction, i.e. start with the open window first
        #
        # NOTE:
        #   - normally we should have H4 >= H1 >= H3 (>=H2)
        #   - in routine measurements H2 is often larger than the others due to
        #     the fluorescence contribution of the Sn-filter in the high energy
        #     radiation field of the natural background
        for key in reversed(self.readout.keys()):

            # check if dose value is smaller than the detection limit
            if self.readout.get(key).get('value') < self.detection_limit.get(key):
                self.readout[key].update({'validity': False})
            else:
                self.readout[key].update({'validity': True})

            # check if at least R4 is valid
            if not self.readout['R4'].get('validity'):
                # if the open window is below the DL, then invalidate all others
                self.readout[key].update({'validity': False})

        # check readout uncertainties
        # if non-existent, compute generic uncertainty
        if not self.readout.get('uncert'):
            self.__default_readout_uncertainty__()

        # check readout background dose
        if not self.readout.get('R2').get('background'):
            self.__default_readout_background__()

        return

    ##############################################################################

    def __default_readout_uncertainty__(self):
        """
        Compute the readout uncertainty (equal to 3*sigma).

        Syntax:
        ------
            self.__default_readout_uncertainty__(self)

        @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                    University hospital Lausanne, Switzerland, 2025
        """
        # loop over each filter value
        for key in self.readout.keys():

            # compute uncertainty based on polynom from linearity tests
            poly = np.polynomial.Polynomial([2.47, 7.8e-3, 5.22e-6])

            # get Hi and compute log10 - avoid zero values
            H = self.readout.get(key).get('value')
            if H < self.detection_limit.get(key):
                H = self.detection_limit.get(key) / np.sqrt(2)
            sd = poly(H) * 1e-3  # convert Micro to MilliSievert

            # check for too small values and replace, if necessary
            # DL = (k_p + k_q) * u(H=0 mSv)
            # -> u(H) ~ DL/(2*k) with k = k_p = k_q = 2.576 (p=q=0.997)
            if sd < self.detection_limit.get(key) / 5.152:
                sd = self.detection_limit.get(key) / 5.152

            # update object data
            self.readout[key].update({'SD': float(fastround(sd, decimals=4))})
            self.readout[key].update({'uncert': float(fastround(3*sd, decimals=4))})

        return

    ##############################################################################

    def __default_readout_background__(self):
        """
        Set readout background to 0 mSv by default.

        This function is called when no natural background is provided by the user.

        Syntax:
        ------
            self.__default_readout_background__(self)

        @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                    University hospital Lausanne, Switzerland, 2025
        """
        # function is called when no natural background is provided

        # loop over each filter value
        for key in self.readout.keys():
            self.readout[key].update({'background': 0.000})

        return

    ##############################################################################

    def __background_subtraction__(self):
        """
        Subtract background from the readout.

        Syntax:
        ------
            self.__background_subtraction__(self)

        @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                    University hospital Lausanne, Switzerland, 2025
        """
        # loop over each filter value
        for key in self.readout.keys():

            # update object data
            H = self.readout.get(key).get('value')
            H_bckgrd = self.readout.get(key).get('background')
            if H_bckgrd is not None:
                H_net = H - H_bckgrd
            else:
                H_net = H

            # avoid negative doses
            if H_net < 0:
                H_net = 0.0

            # update readout
            self.readout[key].update({
                'value':        float(fastround(H_net, 4)),
                'background':   H_bckgrd
            })

        return

    ##############################################################################

    def __compute_energy_of_channels__(self):
        """
        Compute channel energy.

        Syntax:
        ------
           dose = dose.__compute_energy_of_channels__()

        @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                    University hospital Lausanne, Switzerland, 2025
        """
        # NOTE: beta energy detection only works if photon dose is equal to zero,
        # i.e. after subtraction of the natural background, otherwise it will
        # be computed as a delta of the total signal in the open window and the
        # signal attributed to the photon contribution

        # if necessary, display some details
        if self.config.verbose:
            print(80 * '-' + '\nenergies from ratios\n' + 80 * '-')

        # compute photon energies from the readout channel ratios
        self.__compute_photon_ratios_from_readout__()
        self.__compute_photon_energy_from_ratio__()

        # check for betas by computing the specific linear combinations of
        # readout channels
        self.__compute_beta_ratios_from_readout__()
        self.__seek_beta_presence__()

        # display energies, min. and max.
        if self.config.verbose:
            val = [
                self.energies_photons.get(key).get('value') for key in self.energies_photons.keys()
            ]
            print('{:<20s}:\t{:6.3f} MeV'.format('max. E', np.max(val)))
            print('{:<20s}:\t{:6.3f} MeV'.format('min E', np.min(val)))
            for key in self.energies_photons.keys():
                print('{:<20s}:\t{:6.3f} MeV, CI(99.7%) = [{:4.3f} , {:4.3f}] MeV (valid: {}, ratio: {:4.3f})'.format(
                    'E(' + key + ')',
                    self.energies_photons.get(key).get('value'),
                    self.energies_photons.get(key).get('ci997')[0],
                    self.energies_photons.get(key).get('ci997')[1],
                    self.energies_photons.get(key).get('validity'),
                    self.ratio_photons.get(key).get('value')
                ))

        return

    ##############################################################################

    def __compute_photon_ratios_from_readout__(self):
        """
        Compute channel ratios from dosimeter readout.

        Syntax:
        ------
           self.__compute_photon_ratios_from_readout__(self)

        @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                    University hospital Lausanne, Switzerland, 2025
        """
        # correction factor for normalization
        cf_norm = system_calibration_normalization(lookup=self.lookup)

        # hard copy readout from object because we'll modify these values
        # according to the detection limit for computational purposes and
        # want to keep the initial data
        rdout = copy.deepcopy(self.readout)

        # verify detection limit before ratio calculation:
        # 1) If all H's are below DL, ratio = 1 (high energy, no artifical
        #    exposure, nat. background already subtracted from signals)
        # 2) If only one H of the ratio is below DL ratio -> 0 (medium/low
        #    energy, either artifical exposure at low energy or parasitic
        #    signal on the low energy channel)
        for key in rdout.keys():
            rdout[key]['value'] = float(fastround(
                apply_detection_limit(H=rdout.get(key).get('value'), DL=self.detection_limit.get(key)),
                decimals=4
                ))

        # some shortcuts
        lut = self.lookup
        dm_charact = self.lookup.dm_characteristics
        norm = self.config.normalization

        # compute filter ratios, uncertainty and confidence interval
        for ratio in list(lut.ratio_photons.get(norm).keys()):

            # get doses from readout
            ch1 = ratio[0:2]
            ch2 = ratio[3:5]
            key = '{}/{}'.format(ch1, ch2)
            H_i = rdout.get(ch1).get('value')
            H_j = rdout.get(ch2).get('value')

            ###################################################################
            # 1) Ratios for betas only
            ###################################################################

            H_i_beta_highE, H_j_beta_highE = 0.0, 0.0
            H_i_beta_lowE, H_j_beta_lowE = 0.0, 0.0

            if (
                    bool(self.Hp007_beta_highE.get('value'))
                    or bool(self.Hp007_beta_lowE.get('value'))
            ):

                # dose from high energy betas exists
                if self.Hp007_beta_highE.get('value') > 0.0:

                    # high energy beta 90Sr/90Y
                    if not self.energies_beta.get('90Sr/90Y') == 'undetected':

                        # find high energy beta
                        Eb_high = max(dm_charact.response_betas.get('energy'))
                        idx_Eb_high = np.abs(np.array(dm_charact.response_betas.get('energy')) - Eb_high).argmin()

                        # channel i
                        R_i_beta_Ehigh = lut.response_betas[norm][ch1][idx_Eb_high]
                        H_i_beta_highE = R_i_beta_Ehigh * self.Hp007_beta_highE.get('value') / cf_norm.get('betas')

                        # channel j
                        R_j_beta_Ehigh = lut.response_betas[norm][ch2][idx_Eb_high]
                        H_j_beta_highE = R_j_beta_Ehigh * self.Hp007_beta_highE.get('value') / cf_norm.get('betas')

                    # low energy beta 85Kr
                    if not self.energies_beta.get('85Kr') == 'undetected':
                        # find low energy beta
                        Eb_low = min(dm_charact.response_betas.get('energy'))
                        idx_Eb_low = np.abs(np.array(dm_charact.response_betas.get('energy')) - Eb_low).argmin()

                        # channel i
                        R_i_beta_lowE = lut.response_betas[norm][ch1][idx_Eb_low]
                        H_i_beta_lowE = R_i_beta_lowE * self.Hp007_beta_lowE.get('value') / cf_norm.get('betas')

                        # channel j
                        R_j_beta_lowE = lut.response_betas[norm][ch2][idx_Eb_low]
                        H_j_beta_lowE = R_j_beta_lowE * self.Hp007_beta_lowE.get('value') / cf_norm.get('betas')

            ###################################################################
            # 2) Ratios for non-corrected values (mixed rad. field)
            ###################################################################

            # truncate below detection limit
            # note: although a dose > DL was measured, we need to isolate the
            # photon contribution and re-evaluate if we can use specific
            # channels for the computation of the photon energy
            H_i = float(apply_detection_limit(H_i, self.detection_limit.get(ch1)))
            H_j = float(apply_detection_limit(H_j, self.detection_limit.get(ch2)))

            # compute ratio and uncertainty by error propagation
            RR = H_i / H_j
            H_i_sd = rdout.get(ch1).get('SD')
            H_j_sd = rdout.get(ch2).get('SD')
            A = H_i_sd / H_i
            B = H_j_sd / H_j
            RR_sd = RR * np.sqrt(A ** 2 + B ** 2)

            # confidence interval, 3*sigma (99.7%)
            A_UB = H_i + 3 * H_i_sd
            A_LB = H_i - 3 * H_i_sd
            A_LB = np.max([
                A_LB,
                apply_detection_limit(0, self.detection_limit.get(ch1))
                ])

            B_UB = H_j + 3 * H_j_sd
            B_LB = H_j - 3 * H_j_sd
            B_LB = np.max([
                B_LB,
                apply_detection_limit(0, self.detection_limit.get(ch2))
                ])
            RR_ic997 = [A_LB / B_UB, A_UB / B_LB]

            # store to object
            self.ratio_measured.update({
                key: {
                    'value':    fastround(RR, decimals=4),
                    'SD':       fastround(RR_sd, decimals=4),
                    'uncert':   fastround(3*RR_sd, decimals=4),
                    'ci997':    fastround(RR_ic997, decimals=4)
                }

            })

            ###################################################################
            # 3) Ratios from corrected values (pure photon radiation field)
            ###################################################################

            # subtract background and beta contribution
            H_i_net = (H_i - H_i_beta_highE - H_i_beta_lowE)
            H_j_net = (H_j - H_j_beta_highE - H_j_beta_lowE)

            # truncate below detection limit
            H_i_net = apply_detection_limit(
                H_i_net, self.detection_limit.get(ch1))
            H_j_net = apply_detection_limit(
                H_j_net, self.detection_limit.get(ch2))

            # compute ratio and uncertainty by error propagation
            RR = H_i_net / H_j_net
            H_i_sd = rdout.get(ch1).get('SD')
            H_j_sd = rdout.get(ch2).get('SD')
            A = H_i_sd / H_i_net
            B = H_j_sd / H_j_net
            RR_sd = RR * np.sqrt(A ** 2 + B ** 2)

            # confidence interval, 3*sigma (99.7%)
            A_UB = H_i_net + 3 * H_i_sd
            A_LB = H_i_net - 3 * H_i_sd
            A_LB = apply_detection_limit(A_LB, self.detection_limit.get(ch1))

            B_UB = H_j_net + 3 * H_j_sd
            B_LB = H_j_net - 3 * H_j_sd
            B_LB = apply_detection_limit(B_LB, self.detection_limit.get(ch2))
            RR_ic997 = [A_LB / B_UB, A_UB / B_LB]

            ###################################################################
            # 4) store photon part
            ###################################################################

            self.ratio_photons.update({
                key: {
                    'value':    float(fastround(RR, decimals=4)),
                    'SD':       float(fastround(RR_sd, decimals=4)),
                    'uncert':   float(3.0*fastround(RR_sd, decimals=4)),
                    'ci997':    [float(v) for v in fastround(RR_ic997, decimals=4)]
                }

            })

        # set validity of filter ratios
        self.__ratio_validation__()

        return

    ##############################################################################

    def __compute_beta_ratios_from_readout__(self):
        """
        Compute beta channel linear combinations from dosimeter readout.

        Syntax:
        ------
            self.__compute_beta_ratios_from_readout__(self)

        @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                    University hospital Lausanne, Switzerland, 2025
        """
        def subtract_photon_contribution(ch: str, idx_Eeff: int) -> float:
            """Subtract photon dose and uncertainty from channel."""
            R_ph = self.lookup.response_photons[self.config.normalization][ch][idx_Eeff]
            Hp_val = self.Hp007_ph['value']
            Hp_sd = self.Hp007_ph['SD']
            return Hp_val * R_ph, Hp_sd * R_ph

        def compute_ratio(numerator: float, denominator: float, u_n: float, u_d: float) -> float:
            """Compute ratio and confidence interval."""
            R = fastround(numerator / denominator, decimals=4)
            R_UB = fastround((numerator + 3 * u_n) / (denominator - 3 * u_d), decimals=4)
            R_LB = fastround((numerator - 3 * u_n) / (denominator + 3 * u_d), decimals=4)
            sd_R = fastround(SD_from_CI(R, R_LB, R_UB, dist='cosine'), decimals=4)
            return R, sd_R, [R_LB, R_UB]

        rdout = copy.deepcopy(self.readout)
        ch1, ch2 = 'R1', 'R4'

        # shortcuts
        dm_charact = self.lookup.dm_characteristics

        for key in dm_charact.ratio_lincomb_betas:

            # set default
            self.ratio_betas[key] = {
                'value':    None,
                'SD':       None,
                'uncert':   None,
                'ci997':    [None, None],
                'validity': False
            }

            # check validity and proceed
            if not rdout.get(ch2, {}).get('validity'):
                # skip invalid data and proceed to next ratio
                continue

            # photon subtraction
            if isinstance(self.Hp007_ph.get('value'), float):
                Eeff = self.effective_photon_energy.get('value')
                idx = np.abs(np.array(dm_charact.response_photons['energy']) - Eeff).argmin()
                H_ph_ch1, sd_ph_ch1 = subtract_photon_contribution(ch1, idx)
                H_ph_ch2, sd_ph_ch2 = subtract_photon_contribution(ch2, idx)

            else:

                H_ph_ch1, H_ph_ch2, sd_ph_ch1, sd_ph_ch2 = 0.0, 0.0, 0.0, 0.0

            # net doses
            H_ch1 = rdout[ch1]['value'] - H_ph_ch1
            H_ch2 = rdout[ch2]['value'] - H_ph_ch2

            H_ch1 = apply_detection_limit(H_ch1, self.detection_limit[ch1])
            H_ch2 = apply_detection_limit(H_ch2, self.detection_limit[ch2])

            # combined uncertainties
            sd_ch1 = np.sqrt(rdout[ch1]['SD'] ** 2 + sd_ph_ch1 ** 2)
            sd_ch2 = np.sqrt(rdout[ch2]['SD'] ** 2 + sd_ph_ch2 ** 2)

            # compute ratio
            if key == '(R4-R1)/R4':
                validity = rdout['R4']['validity'] and not rdout['R1']['validity']
                R, sd_R, ci = compute_ratio(H_ch2 - H_ch1, H_ch2, sd_ch2, sd_ch2)
            elif key == '(R4-R1)/R1':
                validity = rdout['R4']['validity'] and rdout['R1']['validity']
                R, sd_R, ci = compute_ratio(H_ch2 - H_ch1, H_ch1, sd_ch2, sd_ch1)
            else:
                raise ValueError(f'Unknown ratio key: {key}')

            # write back to object
            self.ratio_betas[key] = {
                'value':    float(fastround(R, decimals=4)),
                'SD':       float(fastround(sd_R, decimals=4)),
                'uncert':   float(fastround(3.0*sd_R, decimals=4)),
                'ci997':    [float(v) for v in fastround(ci, decimals=4)],
                'validity': validity
            }

        return

    ##############################################################################

    def __ratio_validation__(self):
        """
        Validate computed ratios.

        Syntax:
        ------
           self.__ratio_validation__(self)

        @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                    University hospital Lausanne, Switzerland, 2025
        """
        # photons: non-corrected ratios
        for key in self.ratio_measured.keys():
            ch1 = key[0:2]
            ch2 = key[3:5]
            valid = (
                self.readout.get(ch1).get('validity') &
                self.readout.get(ch2).get('validity')
            )

            # store to object
            self.ratio_measured[key].update({
                'validity': valid
            })

        # photons: beta radiation corrected ratios
        for key in self.ratio_measured.keys():
            ch1 = key[0:2]
            ch2 = key[3:5]
            valid = (
                self.readout.get(ch1).get('validity') &
                self.readout.get(ch2).get('validity')
            )

            self.ratio_photons[key].update({'validity': valid})

        return

    ##############################################################################

    def __compute_photon_energy_from_ratio__(self):
        """
        Compute photon energy from ratio.

        Syntax:
        ------
           self.__compute_photon_energy_from_ratio__(self)

        @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                    University hospital Lausanne, Switzerland, 2025
        """
        # init.
        dec_rnd = 3
        norm = self.config.normalization

        # loop over all filter ratios
        for key in self.ratio_photons.keys():

            # retrieve index of ratio in lookup table closest to exp. ratio
            index = find_nearest(
                array=self.lookup.ratio_photons.get(norm).get(key),
                value=self.ratio_photons.get(key).get('value'),
                side='left'
            )
            index_LB = find_nearest(
                array=self.lookup.ratio_photons.get(norm).get(key),
                value=self.ratio_photons.get(key).get('ci997')[0],
                side='left'
            )

            index_UB = find_nearest(
                array=self.lookup.ratio_photons.get(norm).get(key),
                value=self.ratio_photons.get(key).get('ci997')[1],
                side='right'
            )

            # energy from ratio
            E = self.lookup.response_photons.get('energy')[index]

            # energy interval of confidence, k=3 (99.7%)
            E_LB = self.lookup.response_photons.get('energy')[index_LB]
            E_UB = self.lookup.response_photons.get('energy')[index_UB]

            # impose that lower boundary <= upper boundary
            [E_LB, E, E_UB] = fastround(np.sort([E_LB, E, E_UB]), decimals=dec_rnd)
            if E_LB == E_UB:
                # shift by 1 keV
                E_LB -= 1e-3
                E_UB += 1e-3

            if True:
                # std. dev. of energy E
                E_sd = fastround(SD_from_CI(E, E_UB, E_LB, dist='half-cosine'), decimals=dec_rnd)
            else:
                # depreciated: usually less precise values of E
                # compute from derivative dR/dE: ∆E = ∆(Ri/Rj) / d(Ri/Rj)/dE
                dR_ov_dE = np.abs(self.lookup.ratio_photons_deriv.get(norm).get(key)[index])
                if np.abs(dR_ov_dE) < 1e-6:
                    # avoid division by zero
                    dR_ov_dE = 1e-6
                E_sd = self.ratio_photons.get(key).get('SD')/dR_ov_dE

            # minimal tweaking to avoid issues with division by zero
            if E_sd < 1e-3:
                E_sd = 1e-3

            # store to object
            self.energies_photons.update({
                key: {
                    'value':    float(E),
                    'SD':       float(E_sd),
                    'uncert':   float(3*E_sd),
                    'ci997':    [float(E_LB), float(E_UB)],
                    'validity': self.ratio_photons.get(key).get('validity')
                }
            })

    ##############################################################################

    def __seek_beta_presence__(self):
        """
        Seek signature of pure beta radiation fields.

        Syntax:
        ------
           self.__seek_beta_presence__(self)

        @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                    University hospital Lausanne, Switzerland, 2025
        """
        # init.
        L = self.lookup
        self.energies_beta.update({'validity': False})
        if self.energies_beta == {}:
            self.energies_beta.update({
                '85Kr':         'undetected',
                '90Sr/90Y':     'undetected'
                })

        # get beta energies and indices, respectively
        Eb_max = np.max(L.response_betas.get('energy'))
        Eb_min = np.min(L.response_betas.get('energy'))
        index_Eb_max = (np.abs(L.response_betas.get('energy') - Eb_max)).argmin()
        index_Eb_min = (np.abs(L.response_betas.get('energy') - Eb_min)).argmin()

        # set distinction thresholds between photons and betas
        threshold = {
            '90Sr/90Y': float(np.mean([
                np.max(self.lookup.dm_characteristics.ratio_lincomb_photons.get('(R4-R1)/R1')),
                self.lookup.dm_characteristics.ratio_lincomb_betas.get('(R4-R1)/R1')[index_Eb_max]
            ])),
            '85Kr': float(np.mean([
                np.max(self.lookup.dm_characteristics.ratio_lincomb_photons.get('(R4-R1)/R4')),
                self.lookup.dm_characteristics.ratio_lincomb_betas.get('(R4-R1)/R4')[index_Eb_min]
            ])),
        }

        # seek low energy beta, i.e. 85Kr
        if self.ratio_betas.get('(R4-R1)/R4').get('validity'):

            # ratio is valid, so there could be low energy betas contributing
            self.energies_beta.update({'validity': True})
            if self.energies_beta.get('85Kr') not in ('unknown', 'detected'):
                self.energies_beta.update({'85Kr': 'unknown'})

            if (
                    self.config.betas.get('85Kr').get('seek')
                    and not self.energies_beta.get('85Kr') == 'detected'
                    ):

                # seek betas from 85Kr
                if (
                        self.ratio_betas.get('(R4-R1)/R4').get('value') > threshold.get('85Kr')
                        and not self.ratio_betas.get('(R4-R1)/R1').get('validity')
                        and self._niter == 0
                ):
                    # 85Kr signature found
                    self.energies_beta.update({'85Kr': 'detected'})

        # seek high energy beta, i.e. 90Sr/90Y
        self.energies_beta.update({'90Sr/90Y': 'undetected'})
        if (
                self.ratio_betas.get('(R4-R1)/R1').get('validity')
                and self.readout.get('R1').get('validity')
                # and not self.energies_beta.get('85Kr') == 'detected'
                and self._niter == 0
        ):

            # ratio is valid, so there might be high E betas contributing
            self.energies_beta.update({'90Sr/90Y': 'unknown'})
            self.energies_beta.update({'validity': True})
            if self.energies_beta.get('85Kr') not in ('unknown', 'detected'):
                # retrograde indication for 85Kr, because R1 is usually invalid in pure 85Kr radiation fields
                self.energies_beta.update({'85Kr': 'unknown'})

            if self.energies_beta.get('90Sr/90Y') not in ('unknown', 'detected'):
                self.energies_beta.update({'90Sr/90Y': 'unknown'})

            if self.config.betas.get('90Sr/90Y').get('seek'):
                # search for 90Sr/90Y is required
                if (self.ratio_betas.get('(R4-R1)/R1').get('value') > threshold.get('90Sr/90Y')):
                    # 90Sr/90Y signature found
                    self.energies_beta.update({'90Sr/90Y': 'detected'})

        return

    ##############################################################################

    def compute_effective_energy(self, **kwargs):
        """
        Compute the effective photon energy.

        Syntax:
        ------
           self.compute_effective_energy(self, **kwargs)

        Args:
        ----
            - channels (list or tuple) to be used in the calculation of the
              effective photon energy
            - method (str) defining the method used to compute the effective
              photon energy (weighted, scoring, logic or logic2 or logic2-scoring, harmonic or geometric)

        @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                    University hospital Lausanne, Switzerland, 2025
        """
        # init.
        channels = self.config.channels.get('energy')
        if len(channels) == 0:
            raise ValueError('Channel list for calculation of effective photon energy cannot be empty.')

        method = self.config.calc_method.get('energy')
        dm_charact = self.lookup.dm_characteristics
        E_ph = E_ph_LB = E_ph_UB = None

        for key in kwargs:
            if key == 'channels':
                channels = kwargs[key].copy()
            if key == 'method':
                method = kwargs[key]

        # update object
        self.effective_photon_energy.update({'channels': channels})

        if self.config.verbose:
            print(80 * '-' + '\neffective energy\n' + 80 * '-')

        # verify validity of channel energies
        keys = []
        if not any([self.energies_photons.get(key).get('validity') for key in self.energies_photons.keys()]):

            # photon part
            if self.config.verbose:
                print('{:<20s}:\tNo valid ratio. Set photon energy to 137Cs.'.format('WARNING'))

            # beta part
            if self.energies_beta.get('validity'):
                if self.config.verbose:
                    print('{:<20s}:\tBeta radiation detected.\t'.format('NOTE'))

        # at least one valid photon energy -> compute effective photon energy
        else:

            # use all ratios
            if channels == ['R1', 'R2', 'R3', 'R4']:

                if hasattr(self, 'ratio_photons'):
                    # n-th iteration
                    keys = list(self.ratio_photons.keys())
                else:
                    # first iteration
                    keys = list(self.ratio_measured.keys())

            # use ratios containing only valid filters defined by the variable 'channels'
            else:

                for key in self.ratio_photons.keys():

                    if self.ratio_photons[key].get('validity'):
                        ch1 = key[0:2]
                        ch2 = key[3:5]
                        if (ch1 in channels) and (ch2 in channels):
                            keys.append(key)

                if keys == []:

                    # no valid ratio yet, which indicates a very low photon energy or betas only -- > use ratio R4/R1
                    if self.ratio_photons.get('R1/R4').get('validity'):
                        keys = ['R1/R4']
                    else:
                        pprint.pprint(self.readout)
                        raise ValueError('No valid channel ratio to determine effective photon energy. Abort.')

            # >>>>> compute effective photon energy <<<<<
            if self.config.verbose > 1:
                print('compute effective photon energy')

            # compute weighted average energy
            elif method in ('weighted', 'harmonic', 'geometric', 'scoring', 'logarithmic', 'power'):
                E_ph, E_ph_LB, E_ph_UB = self.__energy_alg_wght_mean__(keys)

            # logical approach using some ad hoc assumptions
            elif method == 'logic':
                E_ph, E_ph_LB, E_ph_UB = self.__energy_alg_logic__()

            # same as before with some improvements
            elif method in ('logic2', 'logic2-scoring'):
                E_ph, E_ph_LB, E_ph_UB = self.__energy_alg_logic2__()

            else:
                raise ValueError(
                    'Unknown method for photon effective energy determination'
                    )

            # >>>>> if required, do postprocessing of photon energy <<<<<
            if (
                    self.config.optimization.get('energy_postprocessing')
                    and E_ph is not None
                    ):
                if self.config.verbose > 1:
                    print('post-processing of effective photon energy:')
                E_ph, E_ph_LB, E_ph_UB = self.__energy_postprocessing__(E_ph, E_ph_LB, E_ph_UB)

        if E_ph is None:
            # set photon effective energy to 137Cs (approx. for residual background at high photon energy)
            E_ph = dm_charact.response_photons.get('energy')[dm_charact.response_photons.get('beam').index('137Cs')]
            E_ph_LB = dm_charact.response_photons.get('energy')[dm_charact.response_photons.get('beam').index('N-300')]
            E_ph_UB = dm_charact.response_photons.get('energy')[dm_charact.response_photons.get('beam').index('60Co')]

        # compute std. dev.
        E_ph_uc = SD_from_CI(E_ph, E_ph_LB, E_ph_UB, dist='half-cosine_LB')

        # write to object
        self.channels_used.update({'energy': channels})
        self.effective_photon_energy.update(
            {
                'value':    float(fastround(E_ph, decimals=4)),
                'uncert':   float(fastround(E_ph_uc, decimals=4)),
                'ci997':    [float(v) for v in fastround([E_ph_LB, E_ph_UB], decimals=4)],
                'unit':     'MeV'
            })

        # display photon energy
        if self.config.verbose:
            print('{:<20s}:\t{}'.format('method', method))
            if method in ('weighted', 'harmonic', 'geometric', 'scoring', 'logarithmic', 'power'):
                print('{:<20s}:\t{}'.format('signals', keys))
            print('{:<20s}:\t{:4.3f}, CI(99.7%) = [{:4.3f} , {:4.3f}] {}'.format(
                'photons',
                self.effective_photon_energy.get('value'),
                self.effective_photon_energy.get('ci997')[0],
                self.effective_photon_energy.get('ci997')[1],
                self.effective_photon_energy.get('unit')
            ))

            if self.energies_beta.get('validity', False):
                print('{:<20s}:\t90Sr/90Y: {}, 85Kr: {}'.format(
                    'betas',
                    self.energies_beta.get('90Sr/90Y'),
                    self.energies_beta.get('85Kr')
                )
                )
            else:
                print('{:<20s}:\tn/a'.format('betas'))

        return

    ##############################################################################

    def __energy_alg_logic__(self, cfg=None) -> tuple[float, float, float]:
        """
        Compute effective photon energy with simple top-down approach in ratio energies.

        It uses the first ratio that fulfills 0.05 <= Ri/Rj <= 0.95 going from high down
        towards low photon energy.

        Syntax:
        ------
           E_ph, E_ph_LB, E_ph_UB = self.__energy_alg_logic__()

        Output:
        ------
            - E_ph (float): effective photon energy
            - E_ph_LB (float): lower boundary of effective photon energy
            - E_ph_UB (float): upper boundary of effective photon energy

        @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                    University hospital Lausanne, Switzerland, 2025
        """
        # check validity of filter ratios
        validity = list()
        for key in self.energies_photons.keys():
            validity.append(self.energies_photons.get(key).get('validity'))

        if not any(validity):
            raise ValueError('No valid filter ratio. Inappropriate method.')

        # init.
        EE = self.energies_photons
        if cfg is None:
            cfg = {
                'highE':    'R2/R3',
                'medE':     'R2/R1',
                'lowE':     'R3/R1',
                'verylowE': 'R1/R4'
            }

        if hasattr(self, 'ratio_photons'):
            # n-th iteration
            RR = self.ratio_photons
        else:
            # first iteration
            RR = self.ratio_measured

        # loop over the config keys
        E_ph = E_ph_LB = E_ph_UB = None
        for key in cfg.keys():

            # the first valid ratio yields the photon effective energy
            if (
                    RR.get(cfg.get(key))['validity']
                    and RR.get(cfg.get(key))['validity']
                    # tweaking parameter 1
                    and RR.get(cfg.get(key))['value'] <= 0.95
                    # tweaking parameter 2
                    and RR.get(cfg.get(key))['value'] >= 0.05
            ):

                # avoid increase of R2/R3 at E < 65 keV
                if key == 'highE':

                    # cross-check with next lower E ratio
                    if EE.get(cfg.get('medE')).get('value') >= 0.065:

                        continue

                E_ph = EE.get(cfg.get(key)).get('value')
                E_ph_LB, E_ph_UB = EE.get(cfg.get(key)).get('ci997')

        if E_ph is not None:
            return float(E_ph), float(E_ph_LB), float(E_ph_UB)
        else:
            raise ValueError('Method yields no valid energy.')

    ##############################################################################

    def __energy_alg_logic2__(self) -> tuple[float, float, float]:
        """
        Improved version of 'logic'-method.

        It uses all ratios fulfilling 0.05 <= Ri/Rj <= 0.95 and computes a
        weighted average of the effective photon energy

        Syntax:
        ------
           E_ph, E_ph_LB, E_ph_UB = self.__energy_alg_logic2__()

        Output:
        ------
            - E_ph (float): effective photon energy
            - E_ph_LB (float): lower boundary of effective photon energy
            - E_ph_UB (float): upper boundary of effective photon energy

        @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                    University hospital Lausanne, Switzerland, 2025
        """
        # get filter ratios
        if hasattr(self, 'ratio_photons'):
            # n-th iteration
            RR = self.ratio_photons
        else:
            # first iteration
            RR = self.ratio_measured

        # loop over usable filter ratios
        E, E_sd, E_LB, E_UB = list(), list(), list(), list()
        for key in self.energies_photons.keys():

            # check for valid energy from ratio
            if self.energies_photons.get(key)['validity']:

                # check on value of ratio
                if (
                        RR.get(key)['validity']
                        and RR.get(key)['validity']
                        # tweaking parameter 1
                        # don't set <1.0 or the low energy ratio H1/H4 will dominate due to a too important weight
                        and RR.get(key)['value'] <= 0.95
                        # tweaking parameter 2
                        and RR.get(key)['value'] >= 0.05
                ):
                    # if check is passed, at value to list
                    E.append(self.energies_photons.get(key).get('value'))
                    E_sd.append(self.energies_photons.get(key).get('SD'))
                    E_LB.append(self.energies_photons.get(key).get('ci997')[0])
                    E_UB.append(self.energies_photons.get(key).get('ci997')[1])

        # if no data, return to defaults
        if len(E) == 0:
            if self.config.verbose > 0:
                print(
                    "Solver '{}' doesn't yield any value for E. Fallback to default.".format(
                        self.config.calc_method.get('energy')
                        ))
            return None, None, None

        # pass to numpy for maths
        E, E_sd = np.array(E), np.array(E_sd)
        E_LB, E_UB = np.array(E_LB), np.array(E_UB)

        # apply weighted average
        w = energy_weighting_factor(E, E_sd)

        if self.config.calc_method.get('energy') == 'logic2-scoring':
            Eeff = scoring_mean(E, weights=w)
            # Eeff_sd = scoring_mean(E_sd, weights=w)
            Eeff_UB = scoring_mean(E_UB, weights=w)
            Eeff_LB = scoring_mean(E_LB, weights=w)
        else:
            Eeff = np.average(E, weights=w)
            # Eeff_sd = np.average(E_sd, weights=w)
            Eeff_UB = np.average(E_UB, weights=w)
            Eeff_LB = np.average(E_LB, weights=w)

        return float(Eeff), float(Eeff_LB), float(Eeff_UB)

    ##############################################################################

    def __energy_alg_wght_mean__(self, keys=None) -> tuple[float, float, float]:
        """
        Compute the effective photon energy by weighted averaging over the individual ratio energies.

        Syntax:
        ------
           E_ph, E_ph_LB, E_ph_UB = __energy_alg_wght_mean__(self, keys=None)

        Args:
        ----
            - keys (list, tuple) of ratio energy used in the weighted average,
              by default all are used
             - setting self.config.calc_method.get('energy') to be:
                 'weighted' : std. weighted mean
                 'harmonic' : harmonic mean
                 'geometric': geometric mean
                 'scoring'  : scoring mean

        Output:
        ------
            - E_ph (float): effective photon energy
            - E_ph_LB (float): lower boundary of effective photon energy
            - E_ph_UB (float): upper boundary of effective photon energy

        @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                    University hospital Lausanne, Switzerland, 2025
        """
        # init.
        method = self.config.calc_method.get('energy')
        if keys is None:
            keys = list(self.energies_photons.keys())

        # check validity of keys
        validity = list()
        for key in keys:
            validity.append(self.energies_photons.get(key).get('validity'))

        if not any(validity):
            # take all available energies (usually add combinations with R4)
            keys = list(self.energies_photons.keys())

        # get energy and its uncertainty for each filter ratio
        E, E_sd, E_UB, E_LB, validity = list(), list(), list(), list(), list()
        for key in keys:
            E.append(self.energies_photons.get(key).get('value'))
            E_sd.append(self.energies_photons.get(key).get('uncert'))
            E_LB.append(self.energies_photons.get(key).get('ci997')[0])
            E_UB.append(self.energies_photons.get(key).get('ci997')[1])
            validity.append(self.energies_photons.get(key).get('validity'))

        # go NumPy
        E, E_sd = np.array(E), np.array(E_sd)
        E_LB, E_UB = np.array(E_LB), np.array(E_UB)

        # check validity of ratios
        idx = [i for i, v in enumerate(validity) if v is True]

        if len(idx) > 1:

            # at least more than one valid ratio -> launch weighting procedure
            if method in ('harmonic', 'geometric', 'scoring', 'trimmed'):
                weight = energy_weighting_factor(E[idx], E_sd[idx], method='standard')
            elif method in ('weighted', 'power'):
                weight = energy_weighting_factor(E[idx], E_sd[idx], method='loglin')
            else:
                raise ValueError('Unknown method for computation of weights.')

            # sort weight from largest value to smallest (req. by scoring avg.)
            idx = np.argsort(weight)[::-1]
            weight, E, E_sd = weight[idx], E[idx], E_sd[idx]
            E_LB, E_UB = E_LB[idx], E_UB[idx]

            # compute weighted average
            if method == 'weighted':
                E_ph = np.average(E, weights=weight)
                E_ph_UB = np.average(E_UB, weights=weight)
                E_ph_LB = np.average(E_LB, weights=weight)

            elif method == 'harmonic':
                E_ph = stats.hmean(E, weights=weight)
                E_ph_UB = stats.hmean(E_UB, weights=weight)
                E_ph_LB = stats.hmean(E_LB, weights=weight)

            elif method == 'geometric':
                E_ph = stats.hmean(E, weights=weight)
                E_ph_UB = stats.gmean(E_UB, weights=weight)
                E_ph_LB = stats.gmean(E_LB, weights=weight)

            elif method == 'scoring':
                E_ph = scoring_mean(E, weight)
                E_ph_UB = scoring_mean(E_UB, weight)
                E_ph_LB = scoring_mean(E_LB, weight)

            elif method == 'power':
                E_ph = stats.pmean(E, p=0.5, weights=weight)
                E_ph_UB = stats.pmean(E_UB, p=0.5, weights=weight)
                E_ph_LB = stats.pmean(E_LB, p=0.5, weights=weight)

            else:
                raise ValueError(
                    'Unknown method for computation of weighted average of photon energy.')

        else:

            # only one ratio is valid -> nothing sophisticated to do
            E_ph, E_ph_LB, E_ph_UB = float(E[idx]), float(E_LB[idx]), float(E_UB[idx])

        [E_ph_LB, E_ph, E_ph_UB] = list(fastround(np.sort([E_ph_LB, E_ph, E_ph_UB]), decimals=4))
        if E_ph_LB == E_ph_UB:
            E_ph_LB -= 1e-3
            E_ph_UB += 1e-3

        return E_ph, E_ph_LB, E_ph_UB

    ##############################################################################

    def __energy_postprocessing__(self, E_ph: float, E_ph_LB: float, E_ph_UB: float) -> tuple[float, float, float]:
        """
        Post-processing the effective photon energy.

        Below ca. 20 keV a branching approach is used.
            See also: __correct_low_photon_energy_branches__
        Above 20 keV a system of linear equations is solved yielding Eigenvalues being combined to
        a best guess for the effective photon energy.
            See also: __correct_photon_energy_SOLE__

        Syntax:
        ------
           E_ph, E_ph_LB, E_ph_UB = __energy_postprocessing__(E_ph, E_ph_LB, E_ph_UB)


        @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                    University hospital Lausanne, Switzerland, 2025
        """
        # init.
        norm = self.config.normalization
        idx_verylow = np.abs(np.array(self.lookup.ratio_photons[norm].get('R1/R4')) - 0.3).argmin()
        idx_low = np.abs(np.array(self.lookup.ratio_photons[norm].get('R1/R4')) - 0.75).argmin()

        # define threshold where the low energy response ratio starts to have a good energy resolution
        threshold = dict({
            # ca. 15 keV
            'E_ph_verylow': self.lookup.response_photons.get('energy')[idx_verylow],
            # ca. 20 keV
            'E_ph_low':     self.lookup.response_photons.get('energy')[idx_low],
            })

        # further optimization of effective photon energy
        if self.config.optimization.get('energy_postprocessing'):
            if E_ph <= threshold.get('E_ph_verylow'):

                # for very low energies: channel 4 is required
                if (
                        self.config.betas.get('90Sr/90Y').get('calc_dose', True) is False
                        or self.Hp007_beta_highE.get('value', 0.0) == 0.0
                        or self._niter == 1
                        ):
                    # fix energy to those of low energy N-qualities
                    # (more precise in pure photon fields, but less precise in mixed fields)
                    # in the first iteration this branch is always used to separate photon and beta signals
                    (E_ph_LB, E_ph, E_ph_UB) = self.__correct_low_photon_energy_branches__(E_ph)

                else:

                    # take most probable energy of the N-qualities
                    # (less precise in pure photon fields, but more precise in mixed fieds)
                    (E_ph_LB, E_ph, E_ph_UB) = self.__correct_photon_energy_SOLE__(
                        E_ph, E_ph_LB, E_ph_UB,
                        linear_comb=False, R4_required=True
                        )
            elif (E_ph > threshold.get('E_ph_verylow') and E_ph <= threshold.get('E_ph_low')):

                # threshold is around 30 keV: linear combination of solution
                (E_ph_LB, E_ph, E_ph_UB) = self.__correct_photon_energy_SOLE__(
                    E_ph, E_ph_LB, E_ph_UB,
                    linear_comb=True, R4_required=True
                    )

            # for energies larger than 20 to 30 keV: channel 4 in not required
            else:

                (E_ph_LB, E_ph, E_ph_UB) = self.__correct_photon_energy_SOLE__(
                    E_ph, E_ph_LB, E_ph_UB,
                    linear_comb=True, R4_required=False
                    )

        # avoid zero-confidence interval
        [E_ph_LB, E_ph, E_ph_UB] = fastround(np.sort([E_ph_LB, E_ph, E_ph_UB]), decimals=4)
        if E_ph_LB == E_ph_UB:
            E_ph_LB -= 1e-3
            E_ph_UB += 1e-3

        # display info
        if self.config.verbose > 1:
            print(
                '--> final effective photon energy: {:4.3f} MeV  [{:4.3f}, {:4.3f}]'.format(
                    E_ph, E_ph_LB, E_ph_UB
                    ))

        return float(E_ph), float(E_ph_LB), float(E_ph_UB)

    ##############################################################################

    def __correct_low_photon_energy_branches__(self, E_ph: float):
        """
        Post-processing effective photon energy below 30 keV.

        Syntax:
        ------
           __correct_low_photon_energy_branches__(self, E_ph, E_ph_LB, E_ph_UB)


        @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                    University hospital Lausanne, Switzerland, 2025
        """
        # set ref. energy based on N-qualities
        energy_ref = np.sort(np.array(self.lookup.dm_characteristics.response_photons.get('energy')))[:4]
        E_ph_new = energy_ref[np.abs(np.array(energy_ref) - E_ph).argmin()]

        # set boundaries based on energy of N-qualities
        E_ph_LB, E_ph_UB = np.nan, np.nan
        for i, e in enumerate(energy_ref):
            if e == E_ph_new:
                if i > 0:
                    E_ph_LB = energy_ref[i - 1]
                else:
                    E_ph_LB = energy_ref[0]

                if i < len(energy_ref) - 1:
                    E_ph_UB = np.max([energy_ref[i + 1], E_ph])
                else:
                    E_ph_UB = np.max([energy_ref[i], E_ph])

        # return energy, upper and lower limit
        return float(E_ph_new), float(E_ph_LB), float(E_ph_UB)

    ##############################################################################

    def __correct_photon_energy_SOLE__(
            self, E_ph: float, E_ph_LB: float, E_ph_UB: float,
            linear_comb: bool = True, R4_required: bool = False
            ) -> tuple[float, float, float]:
        """
        Post-processing for finding a best guess of the effective photon energy.

        Syntax:
        ------
            __correct_photon_energy_SOLE__(self, linear_comb=True, R4_required=False)

        Args:
        ----
                linear_comb (bool): return linear combination of solution (default)
                or the most probable solution

        Details:
        -------

            The function solves a system of linear equations (SOLE) Ax = lambda*x = b with:
            b: ratios from readout
            x: what we seek: best matching ratio Hi/Hj
            A: matrix   ¦ H2/H1(E1) ... H2/H1(E5) ¦
                        ¦ .                 .     ¦
                        ¦ .                 .     ¦
                        ¦ .                 .     ¦
                        ¦ H2/H3(E1) ... H2/H3(E5) ¦

        Note:
        ----
            Instead of solving the SOLE and obtaining potentially negative
            eigenvalues from A, we rather solve the minimization problem:

                min ||Ax - b||, with x_i >= 0 for all i in [0, n-1]

            with n the number of available ratios

        @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                    University hospital Lausanne, Switzerland, 2025
        """
        # init.
        dm_charact = self.lookup.dm_characteristics
        E_RR = np.asarray(dm_charact.response_photons.get('energy'))
        RR_table = dm_charact.ratio_photons

        # choose raw data
        if not bool(self.ratio_photons):
            # beta contribution already extracted
            RR = self.ratio_photons
        else:
            # uncorrected ratios, ie photons and betas are still mixed
            RR = self.ratio_measured

        # init. matrix for linear algebra
        A = list()
        readout, readout_LB, readout_UB = list(), list(), list()
        idx_Eeff = np.abs(E_RR - E_ph).argmin()

        # if possible, force 60Co energy into the range, otherwise it risks to be never detected
        if E_ph_UB > 1.0:
            E_ph_UB = dm_charact.response_photons['energy'][dm_charact.response_photons['beam'].index('60Co')]
        # do the same with N-300
        if E_ph_LB >= 0.15 and E_ph_LB < 0.5:
            E_ph_LB = dm_charact.response_photons['energy'][dm_charact.response_photons['beam'].index('N-250')]

        # indices of valid ratios/energies
        idx_E = np.where((E_RR >= E_ph_LB) & (E_RR <= E_ph_UB))[0]

        # correct for too small CI -> buid manually
        R4_required = False
        if len(idx_E) < 3:
            # confidence interval too small for building a matrix -> adapt signals to be taken into account
            if idx_Eeff > 0:
                # not the lowest possible energy
                R4_required = False
                if idx_Eeff < len(E_RR) - 1:
                    # take lower and higher energy as interval
                    idx_E = [idx_Eeff - 1, idx_Eeff, idx_Eeff + 1]
                else:
                    # take lower lower energies
                    idx_E = [idx_Eeff - 2, idx_Eeff - 1, idx_Eeff]
            else:
                # take interval at lowest energy possible
                R4_required = True
                idx_E = [idx_Eeff, idx_Eeff + 1, idx_Eeff + 2]

        # build vector b and matrix A
        for key in RR.keys():
            # exclude beta window unless it is required at very low Eeff
            if 'R4' not in key or R4_required:
                readout.append(self.ratio_photons.get(key).get('value'))
                readout_LB.append(self.ratio_photons.get(key).get('ci997')[0])
                readout_UB.append(self.ratio_photons.get(key).get('ci997')[1])
                A.append(list(np.asarray(RR_table.get(key))[idx_E]))

        # do minimization problem:
        # min ||Ax - b||, with x_i >= 0 for all i in [0, n-1]
        if self.config.verbose > 1:
            print(
                """
                Solving minimization problem:
                min ||Ax - b||, with x_i >= 0 for all i in [0, n-1]
                """
            )

        # 1) get matrix A and vector x=E_RR
        E_RR = E_RR[idx_E]
        A = np.array(A)

        # 2) compute Eigenvalues for E_ph and boundaries
        lambda_E = self.__SOLE_solver__(A, readout)
        lambda_UB = self.__SOLE_solver__(A, readout_UB, x_init=lambda_E)
        lambda_LB = self.__SOLE_solver__(A, readout_LB, x_init=lambda_E)

        # 3) compute energy and its boundary
        if linear_comb:

            # linear combination (in general little better results)
            E_ph = fastround(sum(E_RR * lambda_E) / sum(lambda_E), decimals=4)
            E_ph_UB = fastround(sum(E_RR * lambda_UB) / sum(lambda_UB), decimals=4)
            E_ph_LB = fastround(sum(E_RR * lambda_LB) / sum(lambda_LB), decimals=4)

        else:

            # most probable energy
            E_ph = E_RR[lambda_E.argmax()]
            E_ph_UB = E_RR[lambda_UB.argmax()]
            E_ph_LB = E_RR[lambda_LB.argmax()]

        # return energy, upper and lower limit
        return float(E_ph_LB), float(E_ph), float(E_ph_UB)

    ##############################################################################

    def __SOLE_solver__(
            self,
            A: NDArray[np.float64],
            b: NDArray[np.float64],
            x_init=None,
            bounds=None,
            version: int = 2) -> NDArray[np.float64]:
        """
        Minimize ||Ax - b||, with x_i >= 0 for all i in [0, n-1] and b = lambda*x.

        Syntax:
        ------
            lambdas = __SOLE_solver__(self, A, b, x_init=None, bounds=None, version=2)

        Args:
        ----
                A (NumPy array) : matrix A
                b (NumPy array) : vector b (=lambda*x)
                x_init (NumPy array) : initial guess of x
                bounds (NumPy array) : bounds of x
                version (int) : 1 - SciPy.minimize (default)
                                2 - SciPy.lsq_linear (faster, better results)

        Output:
        ------
                lambdas (NumPy array) : corresponding eigenvalue's'

        @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                    University hospital Lausanne, Switzerland, 2025

                    inspired by contribution of user 'pas-calc' stackoverflow:
                    https://stackoverflow.com/questions/36968955/numpy-linear-system-with-specific-conditions-no-negative-solutions, 2021
        """
        if not isinstance(A, np.ndarray):
            A = np.matrix(A)
        if not isinstance(b, np.ndarray):
            b = np.array(b)
        if x_init is not None:
            if not isinstance(x_init, np.ndarray):
                x_init = np.matrix(x_init)
        if bounds is not None:
            if not isinstance(bounds, np.ndarray):
                bounds = np.matrix(bounds)

        # number of parameters/quantities
        n = A.shape[1]

        # check initial condition
        if x_init is None:

            # init. to equal contributions
            x_init = np.ones(n)

        # check boundary conditions
        if bounds is None:
            bounds = [(0., 1.) for weights in range(n)]

        # Solves min ||Ax - b|| such as x_i >= 0 for all i in [0, n-1]
        # inspired by 'pas-calc' avoiding negative Eigenvalues
        if version == 1:

            # 1st version: use SciPy's minimize solver
            n = A.shape[1]
            if x_init is None:

                # init. to equal contributions
                x_init = np.ones(n)

            def fun(x): return np.linalg.norm(np.dot(A, x) - b)
            sol = minimize(
                fun,
                x0=x_init,
                method='L-BFGS-B',
                bounds=bounds
            )

            # get lambda's, avoid too small values
            lambdas = np.asarray(sol['x'])
            lambdas[np.where(lambdas < 1e-6)] = 1e-6

        else:

            # 2nd version: use lsq_linear from scipy.optimize, from "pas_calc"
            n = A.shape[1]
            verbosity = int(max([self.config.verbose - 1, 0]))
            res = lsq_linear(
                A, b,
                bounds=np.array(bounds).T,
                # method='trf',
                method='bvls',
                tol=1e-6,
                lsq_solver=None,
                lsmr_tol=1e-2,
                verbose=verbosity
                )

            if self.config.verbose > 1:
                print('{:<20s}:\t{:2d}\n'.format('LSQ-fitting status', res.status))
            lambdas = res.x

        return lambdas

    ##############################################################################

    def compute_dose(self):
        """
        Compute air kerma and dose equivalents.

        Syntax:
        ------
            self.compute_dose()

        Args: via the DoseCalcConfig object

        Output: populates dose object

        @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                    University hospital Lausanne, Switzerland, 2025
        """
        # retrieve channels to be used in dose calculation
        channels = self.config.channels.get('dose')
        method = self.config.calc_method.get('dose')

        if len(channels) == 0:
            raise ValueError('Channel list for for dose calculation cannot be empty.')

        # retrieve photon energy and its boundaries
        E_ph = self.effective_photon_energy.get('value')
        [E_ph_LB, E_ph_UB] = self.effective_photon_energy.get('ci997')

        # if for any reason the experimental data does not yield any
        # boundaries in E_ph, set default values and avoid negative energies;
        # cover 99.7% of confidence interval
        if E_ph_UB is None:
            E_ph_UB = E_ph + self.effective_photon_energy.get('uncert')

        if E_ph_LB is None:
            E_ph_LB = E_ph - self.effective_photon_energy.get('uncert')
        E_ph_LB = float(np.max([E_ph_LB, self.lookup.response_photons.get('energy').min()]))

        if E_ph_UB <= E_ph_LB:
            E_ph_LB, E_ph_UB = np.sort([E_ph_LB, E_ph_UB])

        # only method so far
        if method == 'weighted':
            self.__dose_alg_weighted__(E_ph, E_ph_LB, E_ph_UB, channels)
        else:
            raise ValueError('Unknown method for dose calculation')

        # >>>>> if required, do postprocessing of dose <<<<<
        if (
                self.config.optimization.get('dose_postprocessing')
                and len(self.Kerma) > 0
                ):

            # dose post-processing of dose values
            flg = False
            if isinstance(self.config.optimization.get('dose_postprocessing'), bool):
                flg = True
            if isinstance(self.config.optimization.get('dose_postprocessing'), float):
                if self.config.optimization.get('dose_postprocessing') <= self.effective_photon_energy.get('value'):
                    # if smaller than the provided effective photon energy
                    flg = True
            if flg is True:
                if self.config.verbose > 1:
                    print('post-processing of dose:')
                self.__dose_postprocessing__()

        # correct for background dose H*(d)
        self.background_correction()

        # increment counter
        self._niter += 1

        # store back to object
        self.channels_used.update({'dose': channels})

        # display
        if self.config.verbose > 0:
            self.print_dose()

        return

    ##############################################################################

    def print_dose(self):
        """
        Print dose equivalents in console.

        Syntax:
        ------
            print_dose(self)

        @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                    University hospital Lausanne, Switzerland, 2025
        """
        # init.
        method = self.config.calc_method.get('dose')

        # display results
        print(80 * '-' + '\ndose equivalent\n' + 80 * '-')
        print('{:<20s}:\t{:s}'.format('normalization', self.lookup.normalization))
        print('{:<20s}:\t{:s}'.format('method', method))
        print('{:<20s}:\t{:}'.format('signals', self.channels_used.get('dose')))
        print('{:<20s}:\t{:2d}'.format('iteration', self._niter))
        print('{:<20s}:\t{:6.3f}, CI(99.7%) = [{:6.3f} , {:6.3f}] {}'.format(
            'Hp(10)',
            self.Hp10.get('value'),
            self.Hp10.get('ci997')[0],
            self.Hp10.get('ci997')[1],
            self.Hp10.get('unit'),
        ))
        print('{:<20s}:\t{:6.3f}, CI(99.7%) = [{:6.3f} , {:6.3f}] {}'.format(
            'Hp(0.07)',
            self.Hp007.get('value'),
            self.Hp007.get('ci997')[0],
            self.Hp007.get('ci997')[1],
            self.Hp007.get('unit'),
        ))
        print('{:<20s}:\t{:6.3f}, CI(99.7%) = [{:6.3f} , {:6.3f}] {}'.format(
            ' photons',
            self.Hp007_ph.get('value'),
            self.Hp007_ph.get('ci997')[0],
            self.Hp007_ph.get('ci997')[1],
            self.Hp007_ph.get('unit'),
        ))
        print('{:<20s}:\t{:6.3f}, CI(99.7%) = [{:6.3f} , {:6.3f}] {}\t(included in Hp(0.07): {})'.format(
            ' betas, high E',
            self.Hp007_beta_highE.get('value'),
            self.Hp007_beta_highE.get('ci997')[0],
            self.Hp007_beta_highE.get('ci997')[1],
            self.Hp007_beta_highE.get('unit'),
            self.config.betas['90Sr/90Y']['include_dose']
        ))
        print('{:<20s}:\t{:6.3f}, CI(99.7%) = [{:6.3f} , {:6.3f}] {}\t(included in Hp(0.07): {})'.format(
            ' betas, low E',
            self.Hp007_beta_lowE.get('value'),
            self.Hp007_beta_lowE.get('ci997')[0],
            self.Hp007_beta_lowE.get('ci997')[1],
            self.Hp007_beta_lowE.get('unit'),
            self.config.betas['85Kr']['include_dose']
        ))
        print('{:<20s}:\t{:6.3f}, CI(99.7%) = [{:6.3f} , {:6.3f}] {}'.format(
            'Hp(3)',
            self.Hp3.get('value'),
            self.Hp3.get('ci997')[0],
            self.Hp3.get('ci997')[1],
            self.Hp3.get('unit'),
        ))
        print('{:<20s}:\t{:6.3f}, CI(99.7%) = [{:6.3f} , {:6.3f}] {}'.format(
            'Ka',
            self.Kerma.get('value'),
            self.Kerma.get('ci997')[0],
            self.Kerma.get('ci997')[1],
            self.Kerma.get('unit'),
        ))

        return

    ##############################################################################

    def __dose_alg_weighted__(self, E: float, E_LB: float, E_UB: float, channels: list = ['R1', 'R3']):
        """
        Dose calculation by weighted average of channel dose values.

        Weights are defined as: w = response**2, i.e. a channel with a low response value is depreciated

        Syntax:
        ------
            __dose_alg_weighted__(E, E_LB, E_UB, channels=['R1', 'R3'])

        Args:
        ----
            - E (float) : effective photon energy in MeV
            - E_LB (float): lower boundary of effective photon energy in MeV
            - E_UB (float): upper boundary of effective photon energy in MeV
            - channels (list): channels to be used for dose calculation (default: ['R1', 'R3'])

        Note:
        ----
            - channel 2 may be affected by over response due to natural background -> use with care
            - pure photon radiation: all channels can be selected (['R1', 'R3', 'R4'])
            - mixed photon+beta radiation (default): select only channels 1 to 3,
              the 4th channel (ABS-filtered) is dedicated to Hp(0.07, beta) calc. (['R1', 'R3'])
            - pure beta radiation: switch on dedicated beta-brancheschannels (['R1', 'R4'])

        @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                    University hospital Lausanne, Switzerland, 2025
        """
        # init.
        channels_used = list()
        L = self.lookup
        norm = self.config.normalization

        # get index of mean photon energy, upper and lower boundary
        index_E = int(np.abs(L.response_photons.get('energy') - E).argmin())
        index_E_UB = int(np.abs(L.response_photons.get('energy') - E_UB).argmin())
        index_E_LB = int(np.abs(L.response_photons.get('energy') - E_LB).argmin())

        # correction factor for normalization
        cf_norm = system_calibration_normalization(lookup=L)

        # get beta energies and indices, respectively
        Eb_max = np.max(L.response_betas.get('energy'))
        Eb_min = np.min(L.response_betas.get('energy'))
        index_Eb_max = (np.abs(L.response_betas.get('energy') - Eb_max)).argmin()
        index_Eb_min = (np.abs(L.response_betas.get('energy') - Eb_min)).argmin()

        ###################################################################
        # 1) photon radiation
        ###################################################################

        if bool(self.effective_photon_energy.get('value')):

            # compute dose from photons
            M, M_UB, M_LB = 0.0, 0.0, 0.0
            sum_weight, sum_weight_UB, sum_weight_LB = 0.0, 0.0, 0.0

            # compute measurand from selected channels
            for key in self.readout.keys():

                if key in channels:

                    # readout in terms of Hp(10) normalized to 137Cs due to system calibration
                    # -> convert to quantity Hp(0.07, 137Cs) for beta dose calculations
                    # -> do not correct for response, this is done when applying the weighting average
                    H = self.readout.get(key).get('value') * cf_norm.get('photons')
                    valid = self.readout.get(key).get('validity')
                    R = L.response_photons[norm].get(key)[index_E]
                    R_UB = L.response_photons[norm].get(key)[index_E_UB]
                    R_LB = L.response_photons[norm].get(key)[index_E_LB]

                    # if necessary, correct for presence of betas
                    Hi_beta_highE, Hi_beta_lowE = 0.0, 0.0

                    if not self.Hp007_beta_highE.get('value') is None:
                        # contribution from 90Sr/90Y
                        R_beta = L.response_betas[norm][key][L.response_betas.get('energy').index(Eb_max)]
                        Hi_beta_highE = R_beta * self.Hp007_beta_highE.get('value') / cf_norm.get('betas')

                    if not self.Hp007_beta_lowE.get('value') is None:
                        # contribution from 85Kr
                        R_beta = L.response_betas[norm][key][L.response_betas.get('energy').index(Eb_min)]
                        Hi_beta_lowE = R_beta * self.Hp007_beta_lowE.get('value') / cf_norm.get('betas')

                    # net dose for computation of photon dose only
                    H_net = H - Hi_beta_highE - Hi_beta_lowE
                    if H_net < 0.0:
                        H_net = 0.0

                    # weighting
                    if (H_net >= self.detection_limit.get(key)) and valid:
                        channels_used.append(key)

                        # weighted average with weight w=R**2
                        M += R * H_net
                        sum_weight += R ** 2

                        M_UB += R_UB * H_net
                        sum_weight_UB += R_UB ** 2

                        M_LB += R_LB * H_net
                        sum_weight_LB += R_LB ** 2

            # normalize by sum of weights
            if sum_weight > 0:
                M /= sum_weight
            else:
                # it may become zero in a pure beta radiation field
                sum_weight = 1.0
                # raise ValueError('Sum of weights equal to zero.')

            if sum_weight_UB > 0:
                M_UB /= sum_weight_UB
            if sum_weight_LB > 0:
                M_LB /= sum_weight_LB

            # compute air kerma from measurand in terms of Hp(0.07)
            cf007 = L.hpK.get('hpK007')[index_E]
            cf007_UB = L.hpK.get('hpK007')[index_E_UB]
            cf007_LB = L.hpK.get('hpK007')[index_E_LB]
            Ka, Ka_UB, Ka_LB = M / cf007, M_UB / cf007_LB, M_LB / cf007_UB

        else:

            # no dose from photons
            Ka, Ka_UB, Ka_LB = 0.0, 0.0, 0.0

        # if Ka*hpK(E) is below detection limit, set 0 mSv
        if (Ka * L.hpK.get('hpK007')[index_E]) < np.mean(list(self.detection_limit.values())):
            Ka = 0.0

        # if necessary, adjust upper and lower boundary
        if Ka_LB > Ka:
            Ka_LB = Ka

        if Ka_UB < Ka:
            Ka_UB = Ka

        # if upper boundary of Ka is below the detection limit, set latter
        Ka_UB = np.max([
            Ka_UB,
            np.mean(list(self.detection_limit.values())) / L.hpK.get('hpK007')[index_E]
        ])

        # if lower boundary of Hp(10) is below the detection limit, set equal 0
        if (Ka_LB * L.hpK.get('hpK007')[index_E]) < np.mean(list(self.detection_limit.values())):
            Ka_LB = 0.0

        # compute equivalent dose values
        # Hp(0.07) from photons only: use air kerma
        cf = L.hpK.get('hpK007')[index_E]
        self.hpK.update({'hpK007': float(cf)})
        cf_UB = L.hpK.get('hpK007')[index_E_UB]
        cf_LB = L.hpK.get('hpK007')[index_E_LB]
        Hp007ph, Hp007ph_UB, Hp007ph_LB = Ka * cf, Ka_UB * cf_UB, Ka_LB * cf_LB

        # Hp(10)
        cf = L.hpK.get('hpK10')[index_E]
        self.hpK.update({'hpK10': float(cf)})
        cf_UB = L.hpK.get('hpK10')[index_E_UB]
        cf_LB = L.hpK.get('hpK10')[index_E_LB]
        Hp10, Hp10_UB, Hp10_LB = Ka * cf, Ka_UB * cf_UB, Ka_LB * cf_LB

        # Hp(3)
        cf = L.hpK.get('hpK3')[index_E]
        self.hpK.update({'hpK3': float(cf)})
        cf_UB = L.hpK.get('hpK3')[index_E_UB]
        cf_LB = L.hpK.get('hpK3')[index_E_LB]
        Hp3, Hp3_UB, Hp3_LB = Ka * cf, Ka_UB * cf_UB, Ka_LB * cf_LB

        ###################################################################
        # 2) beta radiation
        ###################################################################
        # Note:
        #   R4(E_beta) = [H4/Hp(0.07)] / [H4/Hp(10,Cs)]
        #   and R4(ph) = [H4/Hp(10,E)] / [H4/Hp(10,Cs)]

        # init.
        Hp007b_lowE, Hp007b_lowE_UB, Hp007b_lowE_LB = 0.0, 0.0, 0.0
        Hp007b_highE, Hp007b_highE_UB, Hp007b_highE_LB = 0.0, 0.0, 0.0

        # check if beta doses need to be calculated
        if (
                self.config.betas.get('90Sr/90Y').get('calc_dose', True)
                or self.config.betas.get('85Kr').get('calc_dose', True)
                and self.energies_beta.get('validity')
        ):

            # photon contribution to H1 and H4
            H4_ph = Hp007ph * L.response_photons[norm].get('R4')[index_E] / cf_norm.get('photons')
            H1_ph = Hp007ph * L.response_photons[norm].get('R1')[index_E] / cf_norm.get('photons')

            # 90Sr/90Y - responses
            R4_beta_highE = L.response_betas[norm].get('R4')[index_Eb_max]
            R1_beta_highE = L.response_betas[norm].get('R1')[index_Eb_max]

            # 85Kr - response
            R4_beta_lowE = L.response_betas[norm].get('R4')[index_Eb_min]

            # beta dose calculation
            if (
                    self.config.betas.get('85Kr').get('calc_dose', False)
                    or self.config.betas.get('90Sr/90Y').get('calc_dose', True)
                    ):

                channels_used.append('R4')

                # compute the contribution from 85Kr
                if self.energies_beta.get('85Kr') == 'detected':

                    # only 85Kr was detected -> straight forward dose calculation
                    H1_ph = H4_ph = 0.0
                    Hp007b_lowE = (self.readout.get('R4').get('value') - H4_ph) * cf_norm.get('betas') / R4_beta_lowE

                elif self.energies_beta.get('85Kr') == 'unknown':

                    if self.energies_beta.get('90Sr/90Y') == 'detected':

                        # 90Sr/90Y has been detected -> assume that H4 comes mainly from high E betas
                        # and neglect those with low energy
                        if (
                                not self.readout['R2']['validity'] is True
                                and not self.readout['R3']['validity'] is True):

                            # likely that no (low energy) photons are present
                            H1_ph = H4_ph = 0.0

                        Hp007b_highE = (
                            (self.readout.get('R4').get('value') - H4_ph)
                            * cf_norm.get('betas') / R4_beta_highE
                            )

                    else:

                        # mixed field possible (worst case and less precise dose estimation):
                        # 1) channel 1 weakly (< 10%) detects high energy betas
                        # 2) channel 4 signal has mixed contributions (photons, high and low energy betas)
                        Hp007b_highE_R1 = (
                            (self.readout.get('R1').get('value') - H1_ph)
                            * cf_norm.get('betas') / R1_beta_highE
                            )
                        Hp007b_highE_R4 = (
                            (self.readout.get('R4').get('value') - H4_ph)
                            * cf_norm.get('betas') / R4_beta_highE
                            )
                        w = np.array([R1_beta_highE, R4_beta_highE])**2

                        if True:
                            # 3a) only take channel 4 (working best)
                            Hp007b_highE = Hp007b_highE_R4
                        elif False:
                            # 3b) average over signals from channel 1 and 4 (moderate results)
                            Hp007b_highE = np.average(
                                [Hp007b_highE_R1, Hp007b_highE_R4], weights=w)
                        else:
                            # 3c) take min. from both (worst results)
                            Hp007b_highE = np.min([Hp007b_highE_R1, Hp007b_highE_R4])

                    # knowing the photon and high beta energy contributions
                    # permits the estimation of the low energy beta contribution
                    if self.config.betas.get('85Kr').get('calc_dose'):
                        Hp007b_lowE = (
                            self.readout.get('R4').get('value')
                            - H4_ph
                            - Hp007b_highE * R4_beta_highE
                        ) * cf_norm.get('betas') / R4_beta_lowE

                # set to zero if result is below detection limit
                if (
                        Hp007b_lowE * R4_beta_lowE
                        < np.sqrt(2) * self.detection_limit.get('R4') * cf_norm.get('betas')
                           ):
                    Hp007b_lowE = 0.0

                if (
                        Hp007b_highE * R4_beta_highE
                        < np.sqrt(2) * self.detection_limit.get('R4') * cf_norm.get('betas')
                           ):
                    Hp007b_highE = 0.0

        # upper boundary in Hp(0.07, beta, high E)
        Hp007b_highE_UB = (
            Hp007b_highE
            + 3 * np.sqrt(self.readout.get('R4').get('SD') ** 2 + (Hp007ph_UB - Hp007ph) ** 2)
        )

        # lower boundary in Hp(0.07, beta, high E)
        Hp007b_highE_LB = (
            Hp007b_highE
            - 3 * np.sqrt((self.readout.get('R4').get('SD')) ** 2 + (Hp007ph - Hp007ph_LB) ** 2)
        )

        # avoid dose below DL or 0
        cond = (self.readout.get('R4').get('value') < np.sqrt(2) * self.detection_limit.get('R4'))
        if Hp007b_highE_LB < 0 or cond:
            Hp007b_highE_LB = 0.0

        # total Hp(0.07): combine photon and beta dose
        # note: use only dose from high energy betas for total dose
        Hp007, Hp007_LB, Hp007_UB = Hp007ph, Hp007ph_LB, Hp007ph_UB

        if self.config.betas.get('90Sr/90Y').get('include_dose'):
            # note: no factor of 3 for the confidence interval of 99.7%, since it is already included in the boundaries
            Hp007 += Hp007b_highE
            Hp007_UB = Hp007 + np.sqrt((Hp007ph_UB - Hp007ph) ** 2 + (Hp007b_highE_UB - Hp007b_highE) ** 2)
            Hp007_LB = Hp007 - np.sqrt((Hp007ph - Hp007ph_LB) ** 2 + (Hp007b_highE - Hp007b_highE_LB) ** 2)

        if self.config.betas.get('85Kr').get('include_dose'):
            print("Hp(0.07, 85Kr) included into total Hp(0.07)")
            Hp007 += Hp007b_lowE

        # store Hp(d) and Ka to object
        self.channels_used.update({'dose': [*{*channels_used}]})   # unique filters only
        self.Kerma.update(self.__Hp_to_dict__(Ka, None, Ka_UB, Ka_LB))
        self.Hp10.update(self.__Hp_to_dict__(Hp10, None, Hp10_UB, Hp10_LB))
        self.Hp007.update(self.__Hp_to_dict__(Hp007, None, Hp007_UB, Hp007_LB))
        self.Hp3.update(self.__Hp_to_dict__(Hp3, None, Hp3_UB, Hp3_LB))
        self.Hp007_ph.update(self.__Hp_to_dict__(Hp007ph, None, Hp007ph_UB, Hp007ph_LB))
        self.Hp007_beta_highE.update(self.__Hp_to_dict__(Hp007b_highE, None, Hp007b_highE_UB, Hp007b_highE_LB))
        self.Hp007_beta_highE.update({'energy': 0.8})
        self.Hp007_beta_lowE.update(self.__Hp_to_dict__(Hp007b_lowE, None, Hp007b_lowE_UB, Hp007b_lowE_LB))
        self.Hp007_beta_lowE.update({'energy': 0.25})

        # correct units
        self.Kerma.update({'unit': 'mGy'})

        return

    ##############################################################################

    def __dose_postprocessing__(self):
        """
        Post-processing dose values by solving a SOLE for air kerma beta contribution to Hp(0.07).

        Syntax:
        ------
           __dose_postprocessing__(self)

        Details:
        -------

            The function solves a system of linear equations (SOLE) Ax = lambda*x = b with:
            b: readout of channels 1 to 4
            x: dose values of (Ka, Hp(0.07, 90Y/90Sr), Hp(0.07, 85Kr))
            A: matrix   ¦ R1(nu) R1(90Y/90Sr) R1(85Kr)  ¦
                        ¦ R2(nu)    0           0       ¦
                        ¦ R3(nu)    0           0       ¦
                        ¦ R4(nu) R4(90Y/90Sr) R4(85Kr)  ¦

        Note:
        ----
            The minimization problem is solved by:

                min ||Ax - b||, with x_i >= 0 for all i in [0, n-1]

            with n the number of available dose values

        @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                    University hospital Lausanne, Switzerland, 2025
        """
        # init.
        L = self.lookup
        Eeff = self.effective_photon_energy
        idx_Eeff = int(np.abs(L.response_photons['energy'] - Eeff.get('value')).argmin())
        cf_norm_Ka = system_calibration_normalization(lookup=L, quantity='kerma')
        cf_norm_Hp007 = system_calibration_normalization(lookup=L, quantity='Hp007')

        # matrix A: channel response to particular dose equivalent
        # vector b: (air kerma, beta dose equivalents)
        A, b, b_LB, b_UB = list(), list(), list(), list()
        if self.config.betas.get('85Kr').get('include_dose'):
            # with low E betas
            for key in self.readout.keys():
                A.append([
                    float(L.response_photons['kerma'][key][idx_Eeff]) / cf_norm_Ka.get('photons'),
                    float(L.response_betas['kerma'][key][0]) / cf_norm_Hp007.get('betas'),
                    float(L.response_betas['kerma'][key][1]) / cf_norm_Hp007.get('betas'),
                    ])
                b.append(self.readout[key]['value']),
                b_LB.append(self.readout[key]['value'] - self.readout[key]['uncert']),
                b_UB.append(self.readout[key]['value'] + self.readout[key]['uncert'])
        else:
            # w/o low E betas
            for key in self.readout.keys():
                A.append([
                    float(L.response_photons['kerma'][key][idx_Eeff]) / cf_norm_Ka.get('photons'),
                    float(L.response_betas['kerma'][key][0]) / cf_norm_Hp007.get('betas'),
                    ])
                b.append(self.readout[key]['value']),
                b_LB.append(self.readout[key]['value'] - self.readout[key]['uncert']),
                b_UB.append(self.readout[key]['value'] + self.readout[key]['uncert'])

        # initial guess of dose equivalents
        x0 = [
            self.Kerma.get('value', 0.0),
            self.Hp007_beta_highE.get('value', 0.0),
            ]
        if self.config.betas.get('85Kr').get('include_dose'):
            x0.append(self.Hp007_beta_lowE.get('value', 0.0))

        # adjust intervals to ensure that the upper bound is always larger than the lower
        x_ci997 = [
            (0., max(self.Kerma.get('ci997')) + 0.1),
            (0., max(self.Hp007_beta_highE.get('ci997')) + 0.1 / L.response_betas['kerma'][key][0]),
            ]
        if self.config.betas.get('85Kr').get('include_dose'):
            x_ci997.append((0., max(self.Hp007_beta_lowE.get('ci997')) + 0.1 / L.response_betas['kerma'][key][1]))

        # solve SOLE
        x = self.__SOLE_solver__(A, b, x_init=x0, bounds=x_ci997).round(4)
        x_LB = self.__SOLE_solver__(A, b_LB, x_init=x0, bounds=x_ci997).round(4)
        x_UB = self.__SOLE_solver__(A, b_UB, x_init=x0, bounds=x_ci997).round(4)

        # retrieve quantities Ka et Hp(0.07, betas)
        Ka, Ka_LB, Ka_UB = x[0], min(x_LB[0], x_UB[0]), max(x_LB[0], x_UB[0])
        Hp007b_highE, Hp007b_highE_LB, Hp007b_highE_UB = x[1], min(x_LB[1], x_UB[1]), max(x_LB[1], x_UB[1])
        if self.config.betas.get('85Kr').get('include_dose'):
            Hp007b_lowE, Hp007b_lowE_LB, Hp007b_lowE_UB = x[2], min(x_LB[2], x_UB[2]), max(x_LB[2], x_UB[2])
        else:
            Hp007b_lowE, Hp007b_lowE_LB, Hp007b_lowE_UB = 0.0, 0.0, 0.0

        # compute remaining quantities
        hpK = self.hpK
        Hp10, Hp10_LB, Hp10_UB = hpK['hpK10'] * Ka, hpK['hpK10'] * Ka_LB, hpK['hpK10'] * Ka_UB
        Hp007ph, Hp007ph_LB, Hp007ph_UB = hpK['hpK007'] * Ka, hpK['hpK007'] * Ka_LB, hpK['hpK007'] * Ka_UB
        Hp3, Hp3_LB, Hp3_UB = hpK['hpK3'] * Ka, hpK['hpK3'] * Ka_LB, hpK['hpK3'] * Ka_UB

        # total Hp(0.07): combine photon and beta dose
        Hp007, Hp007_LB, Hp007_UB = Hp007ph, Hp007ph_LB, Hp007ph_UB
        if self.config.betas.get('90Sr/90Y').get('include_dose'):
            Hp007 += Hp007b_highE
            Hp007_UB = Hp007 + np.sqrt((Hp007ph_UB - Hp007ph) ** 2 + (Hp007b_highE_UB - Hp007b_highE) ** 2)
            Hp007_LB = Hp007 - np.sqrt((Hp007ph - Hp007ph_LB) ** 2 + (Hp007b_highE - Hp007b_highE_LB) ** 2)

        if self.config.betas.get('85Kr').get('include_dose'):
            Hp007 += Hp007b_lowE

        # update object
        self.Kerma.update(self.__Hp_to_dict__(Ka, None, Ka_UB, Ka_LB))
        self.Hp10.update(self.__Hp_to_dict__(Hp10, None, Hp10_UB, Hp10_LB))
        self.Hp007.update(self.__Hp_to_dict__(Hp007, None, Hp007_UB, Hp007_LB))
        self.Hp3.update(self.__Hp_to_dict__(Hp3, None, Hp3_UB, Hp3_LB))
        self.Hp007_ph.update(self.__Hp_to_dict__(Hp007ph, None, Hp007ph_UB, Hp007ph_LB))
        self.Hp007_beta_highE.update(self.__Hp_to_dict__(Hp007b_highE, None, Hp007b_highE_UB, Hp007b_highE_LB))
        self.Hp007_beta_highE.update({'energy': 0.8})
        self.Hp007_beta_lowE.update(self.__Hp_to_dict__(Hp007b_lowE, None, Hp007b_lowE_UB, Hp007b_lowE_LB))
        self.Hp007_beta_lowE.update({'energy': 0.25})

        return

    ##############################################################################

    def background_correction(self, Eeff: float = None):
        """
        Dose correction due to formerly subtracted background from readout.

        Syntax:
        ------
            background_correction(self, Eeff=None)

        Args:
        ----
            Eeff (float) : photon energy of background in MeV (default: 137Cs energy)

        @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                    University hospital Lausanne, Switzerland, 2025
        """
        if Eeff is None:
            # take 137Cs as reference energy
            dm_charact = self.lookup.dm_characteristics
            Eeff = dm_charact.response_photons.get('energy')[dm_charact.response_photons.get('beam').index('137Cs')]

        index_Eeff = np.abs(np.array(self.lookup.hpK.get('energy')) - Eeff).argmin()
        hpK = self.lookup.hpK.get('hpK007')[index_Eeff]
        H_bckgrd = self.readout.get('R1').get('background')

        if H_bckgrd > 0.0:

            # correct Hp(10)
            Hp10 = self.Hp10.get('value')
            Hp10_uncert = self.Hp10.get('uncert')
            Hp10_ci997 = self.Hp10.get('ci997')
            if Hp10 == 0.0:
                Hp10_uncert = np.diff(Hp10_ci997) / 3 * 2
                Hp10_uncert = SD_from_CI(Hp10, Hp10_ci997[1], Hp10_ci997[0], dist='half-cosine')
            else:
                Hp10_uncert *= (1 + H_bckgrd / Hp10)

            Hp10_ci997_corr = np.array(Hp10_ci997) + H_bckgrd
            self.Hp10.update(self.__Hp_to_dict__(
                (Hp10 + H_bckgrd),
                None,
                Hp10_ci997_corr[1],
                Hp10_ci997_corr[0]
                ))

            # correct Hp(0.07), total
            Hp007 = self.Hp007.get('value')
            Hp007_uncert = self.Hp007.get('uncert')
            Hp007_ci997 = self.Hp007.get('ci997')
            if Hp007 == 0.0:
                Hp007_uncert = SD_from_CI(Hp007, Hp007_ci997[1], Hp007_ci997[0], dist='half-cosine')
            else:
                Hp007_uncert *= (1 + H_bckgrd / Hp007)

            Hp007_ci997_corr = np.array(Hp007_ci997) + H_bckgrd
            self.Hp007.update(self.__Hp_to_dict__(
                (Hp007 + H_bckgrd),
                None,
                Hp007_ci997_corr[1],
                Hp007_ci997_corr[0]
                ))

            # correct Hp(0.07), photon contribution
            Hp007_ph = self.Hp007_ph.get('value')
            Hp007_ph_uncert = self.Hp007_ph.get('uncert')
            Hp007_ph_ci997 = self.Hp007_ph.get('ci997')
            if Hp007_ph == 0.0:
                Hp007_ph_uncert = SD_from_CI(Hp007_ph, Hp007_ph_ci997[1], Hp007_ph_ci997[0], dist='half-cosine')
            else:
                Hp007_ph_uncert *= (1 + H_bckgrd / Hp007_ph)

            Hp007_ph_ci997_corr = np.array(Hp007_ph_ci997) + H_bckgrd
            self.Hp007.update(self.__Hp_to_dict__(
                (Hp007_ph + H_bckgrd),
                None,
                Hp007_ph_ci997_corr[1],
                Hp007_ph_ci997_corr[0]
                ))

            # correct Hp(3)
            Hp3 = self.Hp3.get('value')
            Hp3_uncert = self.Hp3.get('uncert')
            Hp3_ci997 = self.Hp3.get('ci997')
            if Hp3 == 0.0:
                Hp3_uncert = SD_from_CI(Hp3, Hp3_ci997[1], Hp3_ci997[0], dist='half-cosine')
            else:
                Hp3_uncert *= (1 + H_bckgrd / Hp3)

            Hp3_ci997_corr = np.array(Hp3_ci997) + H_bckgrd
            self.Hp3.update(self.__Hp_to_dict__(
                (Hp3 + H_bckgrd),
                None,
                Hp3_ci997_corr[1],
                Hp3_ci997_corr[0]
                ))

            # correct air kerma
            Ka = self.Kerma.get('value')
            Ka_uncert = self.Kerma.get('uncert')
            Ka_ci997 = self.Kerma.get('ci997')
            if Ka == 0.0:
                Ka_uncert = SD_from_CI(Ka, Ka_ci997[1], Ka_ci997[0], dist='half-cosine')
            else:
                Ka_uncert *= (1 + (H_bckgrd / hpK) / Ka)

            Ka_ci997_corr = np.array(Ka_ci997) + H_bckgrd / hpK
            self.Kerma.update(self.__Hp_to_dict__(
                (Ka + H_bckgrd / hpK),
                None,
                Ka_ci997_corr[1],
                Ka_ci997_corr[0]
                ))

        return

    ##############################################################################

    def __Hp_to_dict__(
            self, val: float = None, val_sd: float = None, val_UB: float = None, val_LB: float = None
            ) -> dict:
        """
        Packs dose equivalents to dictionary that can be directly written to the dose object.

        Syntax:
        ------
            __Hp_to_dict__(val, val_sd, val_UB, val_LB)

        Args:
        ----
            val (float): dose equivalent in mSv
            val_sd (float): std. dev. of dose equivalent in mSv (if None, it is computed from the LB and UB)
            val_LB (float): lower boundary of dose equivalent
            val_UB (float): upper boundary of dose equivalent

        @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                    University hospital Lausanne, Switzerland, 2025
        """
        if val_sd is None:
            val_sd = SD_from_CI(val, val_UB, val_LB, dist='half-cosine')
        ci997 = list(np.sort([val_LB, val_UB]))
        DL = max(self.detection_limit.values())
        if val > 0:
            if val_sd / val < DL:
                # put value from reproducibility and recompute CI(99.7%)
                val_sd = DL * val
                ci997 = [val - 3 * val_sd, val + 3 * val_sd]

        return {
            'value':    float(fastround(val, decimals=4)),
            'SD':       float(fastround(val_sd, decimals=4)),
            'uncert':   float(fastround(3.0*val_sd, decimals=4)),
            'ci997':    [float(v) for v in fastround(ci997, decimals=4)],
            'unit':     'mSv'
        }


##############################################################################
# Class DoseLSQOptimizer
##############################################################################


class DoseLSQOptimizer(DoseCalculator):
    """
    Class for the LSQ-optimization.

    Note: Method depreciated for weak outcome with respect to dose_postprocessing approach and low speed

    @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                University hospital Lausanne, Switzerland, 2025
    """

    # initialize class
    def __init__(self, obj_dose=None, **kwargs):

        # super().__init__(**kwargs)
        if not isinstance(obj_dose, DoseCalculator):
            raise ValueError("The input must be a DoseCalcConfig object.")
        else:
            self.presolution = obj_dose

        # if necessary, rebuild lookup table as function of Hp(0.07)
        # --> we optimize for Hp(0.07) only
        if not obj_dose.config.normalization == 'Hp007':
            lsq_config = obj_dose.config
            lsq_config.normalization.update({'response': 'Hp007'})
            self.config = lsq_config
            self.dm_characteristics = DosimeterCharacteristics(
                config=lsq_config)
            self.lookup = LookupTable(
                config=self.config,
                dm_characteristics=self.dm_characteristics
            )
            self.readout = obj_dose.readout
        else:
            self.config = obj_dose.config
            self.dm_characteristics = obj_dose.lookup.dm_characteristics
            self.lookup = obj_dose.lookup
            self.readout = obj_dose.readout

        # init. arguments
        self.effective_photon_energy = self.presolution.effective_photon_energy
        self.Kerma = self.presolution.Kerma
        # self.filter = self.presolution.filter
        self.config.channels.update({'energy': ['R1', 'R2', 'R3', 'R4']})
        self.Hp007_beta_highE = self.presolution.Hp007_beta_highE
        self.Hp007_beta_lowE = self.presolution.Hp007_beta_lowE
        self.Hp007_ph = self.presolution.Hp007_ph
        self.Hp007 = self.presolution.Hp007
        self.Hp10 = self.presolution.Hp10
        self.Hp3 = self.presolution.Hp3

    ##############################################################################

    def optimize(self, lsq_method='Nelder-Mead', lsq_options=None):
        """
        Least-square optimization of photon energy, photon and beta dose Hp(0.07).

        Calls 'minimize' from the scipy-package, i.e. arguments 'lsq_method' and 'lsq_options' are passed to 'minimize'

        Please check: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

        Args:
        ----
            - lsq_method (string): 'minimize' methods 'L-BFGS-B' (default) and 'Nelder-Mead'
            - lsq_options (dict.): options of chosen method according to 'minimize' definitions
            - value (single value): reference value
            - side (string 'left' or 'right'): chose between left- or right-handed
              closest value

        @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                    University hospital Lausanne, Switzerland, 2025
        """
        # init.
        idx_Cs = self.lookup.dm_characteristics.response_photons.get('beam').index('137Cs')
        E_Cs = self.lookup.dm_characteristics.response_photons.get('energy')[idx_Cs]
        self.lsq_method = lsq_method

        if not isinstance(lsq_options, dict):
            if self.lsq_method == 'Nelder-Mead':
                self.lsq_options = {
                    'fatol': 1e-8,
                    'xatol': 1e-8,
                    'disp': (self.config.verbose > 0),
                }
            elif self.lsq_method == 'L-BFGS-B':
                self.lsq_options = {
                    'ftol': 1e-8,
                    'gtol': 1e-8,
                    'disp': (self.config.verbose > 0),
                }
            else:
                raise ValueError('Unknown lsq minimizer method.')
        else:
            self.lsq_options = lsq_options

        if self.config.verbose > 0:
            print(80 * '-' + '\nleast-square optimization\n' + 80 * '-')

        # if necessary, rebuild lookup table as function of Hp(0.07)
        if not self.config.normalization == 'Hp007':
            lsq_config = self.config
            lsq_config.normalization = 'Hp007'
            self.lsq_config = lsq_config
            self.lsq_dm_characteristics = DosimeterCharacteristics(config=lsq_config)
            self.lsq_lookup = LookupTable(
                config=self.lsq_config,
                dm_characteristics=self.lsq_dm_characteristics
            )
        else:
            self.lsq_config = self.config
            self.lsq_dm_characteristics = self.lookup.dm_characteristics
            self.lsq_lookup = self.lookup

        # define boundaries on variables E_ph and Hp(0.07)
        L = self.lsq_lookup
        bnds = (
            [L.response_photons.get('energy').min(),
             L.response_photons.get('energy').max()],  # E_ph
            [0, np.inf],    # Hp(0.07, photons)
            [0, np.inf],    # Hp(0.07, beta, highE)
            [0, np.inf]     # Hp(0.07, beta, lowE)
        )

        # define initial fit parameters in terms of Hp(0.07)
        par0 = np.array([
            self.presolution.effective_photon_energy.get('value'),
            self.presolution.Hp007_ph.get('value'),
            self.presolution.Hp007_beta_highE.get('value'),
            self.presolution.Hp007_beta_lowE.get('value'),
        ],
            dtype=float
        )

        # do minimization on chi-sq
        if self.config.verbose > 0:
            print('--> LSQ-fitting input parameters: {} (Eeff,  photon and beta Hp(0.07), high and low E)'.format(par0))
            print('--> feedback from minimization function:')
        res = minimize(
            self.__optimizer_chisq_fun__, par0,
            method=self.lsq_method, options=self.lsq_options, bounds=bnds
            )
        # res  = basinhopping(
        #     self.__optimizer_chisq_fun__,
        #     par0,
        #     )
        self.lsq_results = res
        if self.config.verbose > 1:
            print(res)

        # extract fit results
        E_ph = fastround(res.x[0], decimals=4)
        Hp007_ph = fastround(res.x[1], decimals=4)
        Hp007_b_highE = fastround(res.x[2], decimals=4)
        Hp007_b_lowE = fastround(res.x[3], decimals=4)

        # get conversion factors to swap between Ka and Hp(d)
        index_E = np.abs(L.response_photons.get('energy') - E_ph).argmin()
        index_Cs = np.abs(L.response_photons.get('energy') - E_Cs).argmin()

        cf_hpK10 = L.hpK.get('hpK10')[index_E] / L.hpK.get('hpK10')[index_Cs]
        cf_hpK3 = L.hpK.get('hpK3')[index_E] / L.hpK.get('hpK10')[index_Cs]
        cf_hpK007 = L.hpK.get('hpK007')[index_E] / L.hpK.get('hpK10')[index_Cs]

        # compute photon Hp(10) and Hp(3) from outputs
        Ka = (Hp007_ph / cf_hpK007)
        Hp10 = fastround(cf_hpK10 * Ka, 3)
        Hp3 = fastround(cf_hpK3 * Ka, 3)

        # update data structure
        self.effective_photon_energy.update({'value': E_ph})
        self.Kerma.update({'value': Ka})
        self.Hp007_beta_highE.update({'value': Hp007_b_highE})
        self.Hp007_beta_lowE.update({'value': Hp007_b_lowE})
        self.Hp007_ph.update({'value': Hp007_ph})
        self.Hp007.update({'value': Hp007_ph + Hp007_b_highE})
        self.Hp10.update({'value': Hp10})
        self.Hp3.update({'value': Hp3})

        # correct for background
        self.background_correction()

        # display data
        if self.config.verbose > 0:
            print('\n--> dose results from optimization\n' + 80 * '-')
            print('{:<20s}:\t{:6.3f} {}'.format(
                'E(ph.)',
                self.effective_photon_energy.get('value'),
                self.effective_photon_energy.get('unit'),
            ))
            print('{:<20s}:\t{:6.3f} {}'.format(
                'Hp(10)',
                self.Hp10.get('value'),
                self.Hp10.get('unit'),
            ))
            print('{:<20s}:\t{:6.3f} {}'.format(
                'Hp(0.07)',
                self.Hp007.get('value'),
                self.Hp007.get('unit'),
            ))
            print('{:<20s}:\t{:6.3f} {}'.format(
                ' photons',
                self.Hp007_ph.get('value'),
                self.Hp007_ph.get('unit'),
            ))
            print('{:<20s}:\t{:6.3f} {}\t(included in Hp(0.07): {})'.format(
                ' betas, high E',
                self.Hp007_beta_highE.get('value'),
                self.Hp007_beta_highE.get('unit'),
                self.config.betas['90Sr/90Y']['include_dose']
            ))
            print('{:<20s}:\t{:6.3f} {}\t(included in Hp(0.07): {})'.format(
                ' betas, low E',
                self.Hp007_beta_lowE.get('value'),
                self.Hp007_beta_lowE.get('unit'),
                self.config.betas['85Kr']['include_dose']
            ))
            print('{:<20s}:\t{:6.3f} {}'.format(
                'Hp(3)',
                self.Hp3.get('value'),
                self.Hp3.get('unit'),
            ))
            print('{:<20s}:\t{:6.3f} {}'.format(
                'Ka',
                self.Kerma.get('value'),
                self.Kerma.get('unit'),
            ))

    ##############################################################################

    def __optimizer_chisq_fun__(self, args):
        """
        Least-square chi-square optimization function.

        It computes the dose equivalent of all 4 detectors parting from
        the photon energy, E_ph, the air kerma, Ka, and the dose equivalent from
        beta radiation, Hp007_beta

        Please check: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

        Args:
        ----
            "args = E_ph, Ka, Hp007_beta_highE, Hp007_beta_lowE", with
            - E_ph: effective photon enenrgy in MeV
            - Ka: air kerma in mGy
            - Hp007_beta_highE: high energy betas Hp(0.07) dose equivalent in mSv
            - Hp007_beta_lowE: low energy betas Hp(0.07) dose equivalent in mSv

        @Author:    Andreas Pitzschke (andreas.pitzschke@chuv.ch),
                    University hospital Lausanne, Switzerland, 2025
        """
        # parse arguments
        E_ph, Ka, Hp007_beta_highE, Hp007_beta_lowE = args

        # compute readout from input, i.e. dose equivalents
        H_tot = generate_readout_from_dose(dose_in=self, cfg=self.lookup.config)

        # compute chi-square, i.e. sum((readout from dose - measured readout)**2)
        res = 0.0
        for key in H_tot.keys():
            res += (
                H_tot.get(key).get('value') -
                self.readout.get(key).get('value')
            ) ** 2

        return float(res)
