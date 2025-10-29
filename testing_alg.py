# -*- coding: utf-8 -*-
"""
Sandbox for dose calculation algorithm.

Created on 05.03.2021
@author: A. Pitzschke, IRA, CHUV
"""


##############################################################################
# HEADER
##############################################################################


# Import libraries
import myOSL as osl
import sys
# import pprint
sys.path.append("/file2.intranet.chuv/data2/IRA/GRM/APi/Python/Dosimetry/Utils")

##############################################################################
# USER SECTION
##############################################################################

# Co-60 ex.
readout_Co60 = dict({
    'R1': {'value': 0.999, 'background': 0.1, 'uncert': 0.05},
    'R2': {'value': 0.930, 'background': 0.1, 'uncert': 0.05},
    'R3': {'value': 0.973, 'background': 0.1, 'uncert': 0.05},
    'R4': {'value': 0.995, 'background': 0.1, 'uncert': 0.05}
})

# Cs-137 ex.
readout_highE = dict({
    'R1': {'value': 1.1, 'background': 0.1, 'uncert': 0.05},
    'R2': {'value': 1.1, 'background': 0.1, 'uncert': 0.05},
    'R3': {'value': 1.1, 'background': 0.1, 'uncert': 0.05},
    'R4': {'value': 1.1, 'background': 0.1, 'uncert': 0.05}
})

# test
readout_test = dict({
    'R1': {'value': 2.019+0.132, 'background': 0.132},
    'R2': {'value': 2.149+0.132, 'background': 0.132},
    'R3': {'value': 2.114+0.132, 'background': 0.132},
    'R4': {'value': 2.048+0.132, 'background': 0.132}
})

# very low E ex. (N-15, Hp(0.07)=48.8 mSv, Hp(10)=3.0 mSv, Ka=50.3 mSv)
readout_verylowE = dict({
    'R1': {'value': 5.8345},
    'R2': {'value': 0.0724},
    'R3': {'value': 0.1217},
    'R4': {'value': 25.271}
})
dose_in_verylowE = dict({
            'photons': dict({
                'Hp(10)':       3.000,
                'energy':       0.012,
            }),
            'betas': dict({
                'Hp(0.07)':     None,
                'energy':       None
            })
        })

# verification by RadPro pending (N-20, Hp(10)=3.5 mSv)
readout_RadPro = dict({
    'R1': {'value': 4.1287},
    'R2': {'value': 0.1352},
    'R3': {'value': 0.1569},
    'R4': {'value': 8.0356}
})

dose_in_lowE = dict({
            'photons': dict({
                'Hp(0.07)':     1.027,
                'energy':       0.033,
            }),
            'betas': dict({
                'Hp(0.07)':     None,
                'energy':       None
            })
        })

# low E ex. (N-40, Hp(10)=1.027 mSv, Hp(0.07)=1.086 mSv, Ka=0.849 mSv) --> ok
readout_lowE = dict({
    'R1': {'value': 0.8686},
    'R2': {'value': 0.0629},
    'R3': {'value': 0.0961},
    'R4': {'value': 0.9199}
})
dose_in_lowE = dict({
            'photons': dict({
                'Hp(0.07)':     1.027,
                'energy':       0.033,
            }),
            'betas': dict({
                'Hp(0.07)':     None,
                'energy':       None
            })
        })

# high E beta ex. (90Sr/90Y)
readout_highEb = dict({
    'R1': {'value': 0.066},
    'R2': {'value': 0.003},
    'R3': {'value': 0.007},
    'R4': {'value': 1.042}
})

# low E beta ex.
readout_lowEb = dict({
    'R1': {'value': 0.001},
    'R2': {'value': 0.001},
    'R3': {'value': 0.001},
    'R4': {'value': 0.100}
})

# N-80 and high E beta ex.
# readout_mixed = dict({
#     'R1': {'value': 0.066 + 0.721},
#     'R2': {'value': 0.003 + 0.111},
#     'R3': {'value': 0.005 + 0.477},
#     'R4': {'value': 1.029 + 0.705}
#     })

# N-250 (1 mSv) and high E beta (1 mSv) ex.
readout_mixed = dict({
    'R1': {'value': 0.066 + 0.865},
    'R2': {'value': 0.003 + 0.874},
    'R3': {'value': 0.005 + 0.931},
    'R4': {'value': 1.029 + 0.869}
})

# IC 2025, WB204435
readout_IC = dict({
    'R1': {'value': 4.612, 'background': 0.044},
    'R2': {'value': 0.662, 'background': 0.072},
    'R3': {'value': 1.244, 'background': 0.043},
    'R4': {'value': 4.708, 'background': 0.044}
})
# WB201864
# readout_IC = dict({
#     'R1': {'value': 3.892},
#     'R2': {'value': 3.916},
#     'R3': {'value': 3.932},
#     'R4': {'value': 3.862}
# })

#  example config
cfg_std = {
    # general settings
    'verbose':          1,          # degree of verbosity (0, 1, 2)
    'plot':             True,       # plot results
    'icru_version':     57,         # ICRU 57 (ISO 4037-3:2019) or 93 (future standard)
    'energy_axis': {
        'selection':    'full',     # energy resolution 'full' or 'calibration' (N-series, Cs-137 and Co-60)
        'elements':     100,        # list of values in MeV or number of axis elements (100 by default)
    },

    # normalization of the filter response data
    'normalization':    'Hp007',    # 'kerma', 'Hp10' or 'Hp007'

    # method for photon energy and dose calculation
    'calc_method': {
        # weighted*, scoring**, logic or logic2** or logic2-scoring***, harmonic*, geometric**
        'energy':       'logic2-scoring',
        # weighted
        'dose':         'weighted'
    },

    # criteria for beta dose calculation
    'betas': {
        '90Sr/90Y': {
            'seek':             True,
            'calc_dose':        True,
            'include_dose':     True
        },
        '85Kr': {
            'seek':             True,
            'calc_dose':        False,
            'include_dose':     False
        }
    },

    # filters used to determine photon energies
    'channels': {
        'energy':   ['R1', 'R2', 'R3'],        # take all except R4 (betas possible)
        'dose':     ['R1', 'R3'],              # R1 req., R2 & R3 opt., avoid R4
    },

    # interpolation methods: 'nearest', 'linear' (default), 'cubic'
    'interp_method': {
        'response': {'interpolate': True, 'method': 'linear'},
        'ratio':    {'interpolate': True, 'method': 'linear'},
    },

    # optimizations True/False
    'optimization': {
        # (recommended) if required, add channel specific to beta/low energy photons
        'add_beta_channel':         True,
        # solving a linear system of equations for improving Eeff
        'energy_postprocessing':    True,
        # solving a linear system of equations for improving the dose values (True/False or threshold Eeff ~ 0.15 MeV)
        # note: at Hp(10)=1 mSv and Hp(0.07, 90Sr)=1 mSv, select Eeff threshold or Hp(0.07, 137Cs) is off by 15%
        'dose_postprocessing':      0.15,
        # (NOT recommended) final results polishing with help of least-square-fitting
        'lsq':                      True,
    },
}

dose_benchmark = dict({
    'photons': dict({
        'Hp(0.07)': 1.0,
        'energy':   0.662,
    }),
    'betas': dict({
        'Hp(0.07)': [0.0, 0.0],
        'energy':   [0.8, 0.25],
    })
})

cfg_noise = dict({
    # seed of numpy rng
    'seed':             13395,
    # take a large number
    'niter':            100,
    # 'normal', 'uniform' or 'triangular',
    # according to http://www.isgmax.com/Articles_Papers/Selecting%20and%20Applying%20Error%20Distributions.pdf
    'distribution':     'normal',
    # 3*sigma at 1 mSv from https://doi.org/10.1016/j.radmeas.2024.107346,
    # Table 4, worst case
    'scale':            3*0.034,
    # renormalize total dose+noise to 1 mSv
    'normalize':        True,
    # export panda data frames
    'save_to_csv':      True,
    'plot_results':     True
})

if False:

    # test for user input
    cfg_user_input = osl.DoseCalcConfig(config=cfg_std)
    dm_charact_user_input = osl.DosimeterCharacteristics(config=cfg_user_input)
    lookup_fallback = osl.LookupTable()
    cfg_user_input.energy_axis['elements'] = [0.01, 0.1, 0.622, 1.0]
    lookup_user_input = osl.LookupTable(config=cfg_user_input, dm_charact=dm_charact_user_input)
    dose_user_input = osl.DoseCalculator(readout=readout_highE, lookup=lookup_user_input, config=cfg_user_input)

elif False:

    # test high/low energy distinction
    cfg_default = osl.DoseCalcConfig(selection='mixed')
    cfg_std = osl.DoseCalcConfig(cfg=cfg_std)
    dm_charact_default = osl.DosimeterCharacteristics(config=cfg_default)
    lookup_default = osl.LookupTable(config=cfg_default, dm_charact=dm_charact_default)

    # very low E (nok)
    readout_in_verylowE = osl.generate_readout_from_dose(dose_in=dose_in_verylowE, cfg=cfg_std)
    print(readout_in_verylowE)
    dose_verylowE = osl.compute_dose_from_readout(rdout=readout_verylowE, cfg=cfg_std)

    # low E (ok)
    # readout_in_lowE = osl.generate_readout_from_dose(dose_in=dose_in_lowE, cfg=cfg_std)
    # pprint.pprint(readout_in_lowE)   # norm. ok
    # dose_lowE = osl.compute_dose_from_readout(rdout=readout_lowE, cfg=cfg_std)
    # dose_verylowE = osl.compute_dose_from_readout(rdout=readout_verylowE, cfg=cfg_std)

    # others
    # dose_highE = osl.compute_dose_from_readout(rdout=readout_highE, cfg=cfg_std)
    # dose_highEb = osl.compute_dose_from_readout(rdout=readout_highEb, cfg=cfg_std)
    # dose_lowEb = osl.compute_dose_from_readout(rdout=readout_lowEb, cfg=cfg_std)
    # dose_mixed = osl.compute_dose_from_readout(rdout=readout_mixed, cfg=cfg_std)
    # dose_RadPro = osl.compute_dose_from_readout(rdout=readout_RadPro, cfg=cfg_std)
    # dose_test = osl.compute_dose_from_readout(rdout=readout_test, cfg=cfg_std)
    # dose_IC = osl.compute_dose_from_readout(rdout=readout_IC, cfg=cfg_std)

elif False:

    if cfg_std.get('optimization').get('lsq'):
        dose_std, dose_optim = osl.compute_dose_from_readout(readout_lowE, cfg=cfg_std)
    else:
        dose_std = osl.compute_dose_from_readout(readout_lowE, cfg=cfg_std)
        dose_optim = dose_std

else:
    df, df_quart, df_req = osl.benchmark_algorithm_robustness(
        dose_in=dose_benchmark,
        cfg_alg=cfg_std,
        cfg_noise=cfg_noise
    )
