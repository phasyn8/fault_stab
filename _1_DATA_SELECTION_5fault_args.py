#importing packages
repo = ''

import sys
sys.path.insert(1, 'src/')
import argparse as arg
#sys.path.insert(1, '/Users/amjonz/Documents/GitHub/probable-robot/src')
#sys.path.insert(1, '/Users/amjonz/Documents/GitHub/brunton/src')
#sys.path.insert(1, '/Users/amjonz/Documents/GitHub/mesher/src')
#import mesher as meshr
import brunton_csv_builder as obs
#import gempyModelBuilder as gpmb
#import proBot as sto
#import gempy as gp
import numpy as np
import pandas as pd
import geopandas as gpd
#from gempy_engine.core.data.stack_relation_type import StackRelationType
#import matplotlib.pyplot as plt
#import pyvista as pv
import os

# System call
os.system("")

# Class of different styles
class style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

print(style.YELLOW + "Hello, We are now working to import and parameterize all your input observations...")

# Some initial fault dips that are referenced in literature
TEGdip  = 62.5
DULKdip = 62.5
BELDdip = 62.5
VIERSdip = 67
CARBdip = 73
model_suffix = "cal_mod9_1_E"
parallel_computing_suffix = ''

slurm_parallel = arg.ArgumentParser(description='slurm array version')

slurm_parallel.add_argument('--iter', action='store', dest='iter', default=1)
slurm_parallel.add_argument('--hpc_suffix', action='store', dest='hpc', default='')

args = slurm_parallel.parse_args()

print(f"Comment : {args.hpc}")
print(f"Now computing iteration: {args.iter}")
#parallel_computing_suffix = args.iter

#ADDING SELECTED DATA

#Pointers 
data_path = '../CAL_model_9_obs/'

# ---- 1 ----- depth of formation surfaces interpolated in Petrel from seismic and well logs 
# THESE INDICATE THE BOTTOME OF FORMATIONS
"""True Viersen falut splays geometry"""
#Surf_namurian_pointer = f'{data_path}strat_surfaces/cal_mod9_Top_Zeeland_depth.shp'
#Surf_zeeland_pointer =  f'{data_path}strat_surfaces/cal_mod9_Top_Banyaard_depth.shp'
"""Simplified Viersen Fault Splay geometries"""
Surf_namurian_pointer = f'{data_path}strat_surfaces/cal_model_9_Top_Zeeland.shp'
Surf_nsg_pointer =  f'{data_path}strat_surfaces/cal_model_9_Bottom_NSG_top_chalk.shp'
Surf_zeeland_pointer =  f'{data_path}strat_surfaces/cal_model_9_bottom_banjaard.shp'


# --- 2 ---- fault surface traces...LANDSURFACE shapefiles
surface_trace_teg =  f'{data_path}fault_trace/Tegelen_linestr.shp'
surface_trace_dulk = f'{data_path}fault_trace/Dulkener_linestr.shp'
surface_trace_beld_points = f'{data_path}fault_trace/belfeld_extended_points.shp'
surface_trace_beld_orients =  f'{data_path}fault_trace/Belfeld_linestr.shp'
surface_trace_viers = f'{data_path}fault_trace/Viersen_linestr.shp'

# ----- 3 ----- stratigraphic contacts with seismic traces
seismic_obs_top_zeeland = f'{data_path}seismic_contacts/cal_mod8_Top_Banjaard.shp'
seismic_obs_top_carb = f'{data_path}seismic_contacts/cal_model_9_Top_Zeeland_seismic_interp.shp'
seismic_obs_bottom_nsg = f'{data_path}seismic_contacts/cal_model_9_Bottom_NSG_seismic_interp.shp'

# ----- 4 ----- fault seismic contacts Observations 
obs_fault_Tegelen = f'{data_path}fault_seismic_contacts/cal_model_9_Tegelen_seismic_contacts.csv'
obs_fault_Dulkener = f'{data_path}fault_seismic_contacts/cal_model_9_Dulkener_seismic_contacts.csv'
obs_fault_Belfeld = f'{data_path}fault_seismic_contacts/cal_model_9_Belfeld_seismic_contacts.csv'
obs_fault_Viersen = f'{data_path}fault_seismic_contacts/cal_model_9_Viersen_seismic_contacts.csv'
obs_fault_Carb_teg_anti =f'{data_path}fault_seismic_contacts/cal_model_9_Carbon_teg_anti_seismic_contacts.csv'

# ----- 5 ----- fault sticks from seismic interpretation CSV ()
#fault_sticks_teg = f'{data_path}fault_sticks/cal_mod8_fault_sticks_Tegelen.csv'
#fault_sticks_vier = f'{data_path}fault_sticks/cal_mod8_fault_sticks_Viersen.csv'
#fault_sticks_dulk = f'{data_path}fault_sticks/cal_mod8_fault_sticks_Dulkener.csv'
#fault_sticks_beld = f'{data_path}fault_sticks/cal_mod8_fault_sticks_Belfeld.csv'


#### Section 1: This section established the minimum and maximum uncertainty ranges (parametersized by variance in meters) 
# and the ingesting of datasets and building a pandas datafraem that will be used to parameterize probabilistic model construction
##### The Dataset compilation field computes the pole orientations for surfaces, and feature traces, etc. and adds these 
# to a DataFrame that will be reduced for interpolation. Correlation values will be used to spawn new interpolation datasets 
# by the probable-robot class in section 4. These emulate probable observations randomly selected from the their uncertainties 
# distributions with their associated dimentional (X,Y,Z,Dip,Azimuth) parameters variance and depth confidence.


# Surfaces from seismic interpretation uncertainty ( shapefile )
surface_pnt_xvar = 792
surface_pnt_yvar = 792
surface_pnt_zvar = 831.19
surface_pnt_dipvar = 81
surface_pnt_azivar = 36

# Fault surface trace mapping uncertainty
fault_trace_xvar = 15**2
fault_trace_yvar = 15**2
fault_trace_zvar = 1
fault_trace_dipvar = 81
fault_trace_azivar = 64

# Seismic Interfaces picks uncertainty
interF_xvar = 792
interF_yvar = 792
interF_zvar = 831.19
interF_dipvar = 81
interF_azivar = 64

# Fault Sticks uncertainty

fault_sticks_xvar = 792
fault_sticks_yvar = 792
fault_sticks_zvar = 831.19
fault_sticks_dipvar = 81
fault_sticks_azivar = 64

# Fault Seismic Contact Unceratainty

fault_seismic_xvar = 792
fault_seismic_yvar = 792
fault_seismic_zvar = 831.19
fault_seismic_dipvar = 100
fault_seismic_azivar = 64


#Correlation scalars for formations and faults (formations are indicated from their bottom) (range =  ~0.0  - .999~ )
fault_mapping_corr = 0.6 # this is mostly for dip angle uncertainty 
zeeland_corr = 0.3 # highly discontinuous indications in seismic profiles and very few wells intersect it
namurian_corr = 0.3 # also discontinuous, but better indicated than Zeeland bottom ( assumed, mostly concordant with Namurian Bottom ) 
NSG_corr = 0.9   # High correlation due to strong well defined seismic reflector
fault_sticks_corr = 0.9 # v. high correlation as this is a linked feature
fault_seismic_corr = 0.9 # v. high correlation as this is a linked feature
#Depth confidence levels (TVD)

Zeeland_depth_max = 33
Zeeland_depth_min = -3000

Namurian_depth_max = 33
Namurian_depth_min = -3000

NSG_depth_max = 800
NSG_depth_min = -3000

fault_depth_max = 35    
fault_depth_min = 34

fault_sticks_max = 33
fault_sticks_min = -3000

fault_seismic_max = 33
fault_seismic_min = -3000





"""  BRUNTON FUNCTIONS ____ DATASET BULDING FROM PRIMARY INTERPRETATIONS"""


#path= '/Users/amjonz/Desktop/stocastic_modeling/cal_mod7_b/'
cal_mod9 = obs.observations(data_path='../CAL_model_9_obs/', output_path='./', init_interfaces=None, init_orients=None, output_prefix=model_suffix) # This instantiates an observation classifier object from BRUNTON ! 

# ---- 1 ----- Adding Surfaces (These are 3D shpefile surfaces)
#linking pointers admittedly a little redundant...
Zeeland_bottom = Surf_zeeland_pointer
Namurian_bottom = Surf_namurian_pointer
NSG_bottom = Surf_nsg_pointer

print(style.RED + "Brunton is adding formation surfaces to observations")
#ORIENTATIONS
cal_mod9.add_surface_points_to_orients(Zeeland_bottom, sample_method='grid', num_points=30, grid_spacing=1100, frac=.3, formation='ZeelandFm',
                                          xvar=surface_pnt_xvar, yvar=surface_pnt_yvar ,zvar=surface_pnt_zvar, dipvar=surface_pnt_dipvar, azivar=surface_pnt_azivar,  flip_normal=True,
                                          input_ID='Zeeland bottom surface interpolation from seismic interpretation', input_type='surface', self_correlation=zeeland_corr,
                                            source=Zeeland_bottom, max_depth_confidence=Zeeland_depth_max, min_depth_confidence=Zeeland_depth_min
                                            )
#INTERFACES
cal_mod9.add_surface_points_to_interfaces(Zeeland_bottom, sample_method='grid', num_points=30, grid_spacing=1100, frac=1, formation='ZeelandFm',
                                          xvar=surface_pnt_xvar, yvar=surface_pnt_yvar ,zvar=surface_pnt_zvar, dipvar=surface_pnt_dipvar ,azivar=surface_pnt_azivar,  flip_normal=True,
                                          input_ID='Zeeland bottom surface interpolation from seismic interpretation', input_type='surface', self_correlation=zeeland_corr, 
                                          source=Zeeland_bottom, max_depth_confidence=Zeeland_depth_max, min_depth_confidence=Zeeland_depth_min
                                            )


#OREINTATIONS
cal_mod9.add_surface_points_to_orients(Namurian_bottom, sample_method='grid', num_points=30, grid_spacing=1100, frac=.4, formation='Namurian',
                                          xvar=surface_pnt_xvar, yvar=surface_pnt_yvar ,zvar=surface_pnt_zvar, dipvar=surface_pnt_dipvar ,azivar=surface_pnt_azivar,  flip_normal=True,
                                          input_ID='Namurian bottom surface interpolation from seismic interpretation', input_type='surface', self_correlation=namurian_corr, 
                                          source=Namurian_bottom, max_depth_confidence=Namurian_depth_max, min_depth_confidence=Namurian_depth_min
                                            ) 
#INTERFACE
cal_mod9.add_surface_points_to_interfaces(Namurian_bottom, sample_method='grid', num_points=100, grid_spacing=1100, frac=1, formation='Namurian',
                                          xvar=surface_pnt_xvar, yvar=surface_pnt_yvar ,zvar=surface_pnt_zvar, dipvar=surface_pnt_dipvar ,azivar=surface_pnt_azivar,  flip_normal=True,
                                          input_ID='Namurian bottom surface interpolation from seismic interpretation', input_type='surface', self_correlation=namurian_corr, 
                                          source=Namurian_bottom, max_depth_confidence=Namurian_depth_max, min_depth_confidence=Namurian_depth_min
                                            )

#ORIENTATIONS
cal_mod9.add_surface_points_to_orients(NSG_bottom, sample_method='grid', num_points=30, grid_spacing=1100, frac=.3, formation='NorthSeaGroup', auto_orient_normals=False, flip_normal=True,
                                          xvar=surface_pnt_xvar, yvar=surface_pnt_yvar ,zvar=surface_pnt_zvar, dipvar=surface_pnt_dipvar ,azivar=surface_pnt_azivar,
                                          input_ID='NSG bottom surface interpolation from seismic interpretation', input_type='surface', self_correlation=NSG_corr, 
                                          source=NSG_bottom, max_depth_confidence=NSG_depth_max, min_depth_confidence=NSG_depth_min
                                            ) 
#INTERFACES
cal_mod9.add_surface_points_to_interfaces(NSG_bottom, sample_method='grid', num_points=100, grid_spacing=900, frac=0.75, formation='NorthSeaGroup', auto_orient_normals=False, flip_normal=True,
                                          xvar=surface_pnt_xvar, yvar=surface_pnt_yvar ,zvar=surface_pnt_zvar, dipvar=surface_pnt_dipvar ,azivar=surface_pnt_azivar,
                                          input_ID='NSG bottom surface interpolation from seismic interpretation', input_type='surface', self_correlation=NSG_corr, 
                                          source=NSG_bottom, max_depth_confidence=NSG_depth_max, min_depth_confidence=NSG_depth_min
                                            ) 


#-- 2 --#   Fault land-surface traces... also shapefiles
#linking pointers
teg_surface_trace = surface_trace_teg
dulk_surface_trace = surface_trace_dulk
beld_surface_trace = surface_trace_beld_orients
viers_surface_trace = surface_trace_viers

print("Brunton is adding Fault surface traces to observations")
print(style.RESET)
cal_mod9.add_shapefile_linestrings_and_compute_azimuth_to_orients(teg_surface_trace, fraction=.2, z=33, formation='Tegelen', dip=TEGdip, azimuth_reverse=False,
                                             xvar=fault_trace_xvar, yvar=fault_trace_yvar ,zvar=fault_trace_zvar, dipvar=fault_trace_dipvar ,azivar=fault_trace_azivar,
                                            input_ID='Tegelen surface mapping', input_type='mapping', self_correlation=fault_mapping_corr, source=teg_surface_trace,
                                            max_depth_confidence=fault_depth_max, min_depth_confidence=fault_depth_min
                                            )
cal_mod9.add_shapefile_linestrings_and_compute_azimuth_to_orients(dulk_surface_trace, fraction=.2, z=33, formation='Dulkener', dip=DULKdip,
                                             xvar=fault_trace_xvar, yvar=fault_trace_yvar ,zvar=fault_trace_zvar, dipvar=fault_trace_dipvar ,azivar=fault_trace_azivar,
                                            input_ID='Dulkener surface mapping', input_type='mapping', self_correlation=fault_mapping_corr, source=dulk_surface_trace,
                                            max_depth_confidence=fault_depth_max, min_depth_confidence=fault_depth_min
                                            )
cal_mod9.add_shapefile_linestrings_and_compute_azimuth_to_orients(beld_surface_trace, fraction=.2, z=28, formation='Belfeld', dip=BELDdip, azimuth_reverse=False,
                                             xvar=fault_trace_xvar, yvar=fault_trace_yvar ,zvar=fault_trace_zvar, dipvar=fault_trace_dipvar ,azivar=fault_trace_azivar,
                                            input_ID='Belfeld surface mapping', input_type='mapping', self_correlation=fault_mapping_corr, source=beld_surface_trace,
                                            max_depth_confidence=fault_depth_max, min_depth_confidence=fault_depth_min
                                            )
cal_mod9.add_shapefile_linestrings_and_compute_azimuth_to_orients(viers_surface_trace, fraction=.2, z=33, formation='Viersen', dip=VIERSdip, azimuth_reverse=False,
                                             xvar=fault_trace_xvar, yvar=fault_trace_yvar ,zvar=fault_trace_zvar, dipvar=fault_trace_dipvar ,azivar=fault_trace_azivar,
                                            input_ID='Viersen surface mapping', input_type='mapping', self_correlation=fault_mapping_corr, source=viers_surface_trace,
                                            max_depth_confidence=fault_depth_max, min_depth_confidence=fault_depth_min
                                            )
cal_mod9.add_shapefile_linestrings_and_compute_azimuth_to_interfaces(teg_surface_trace, fraction=1, z=28, formation='Tegelen', dip=TEGdip, azimuth_reverse=False,
                                            xvar=fault_trace_xvar, yvar=fault_trace_yvar ,zvar=fault_trace_zvar, dipvar=fault_trace_dipvar ,azivar=fault_trace_azivar,
                                            input_ID='Tegelen surface mapping', input_type='mapping', self_correlation=fault_mapping_corr, source=teg_surface_trace,
                                            max_depth_confidence=fault_depth_max, min_depth_confidence=fault_depth_min
                                            )
cal_mod9.add_shapefile_linestrings_and_compute_azimuth_to_interfaces(dulk_surface_trace, fraction=1, z=31, formation='Dulkener', dip=DULKdip,
                                             xvar=fault_trace_xvar, yvar=fault_trace_yvar ,zvar=fault_trace_zvar, dipvar=fault_trace_dipvar, azivar=fault_trace_azivar,
                                            input_ID='Dulkener surface mapping', input_type='mapping', self_correlation=fault_mapping_corr, source=dulk_surface_trace,
                                            max_depth_confidence=fault_depth_max, min_depth_confidence=fault_depth_min
                                            )
cal_mod9.add_shapefile_linestrings_and_compute_azimuth_to_interfaces(beld_surface_trace, fraction=1, z=28, formation='Belfeld', dip=BELDdip, azimuth_reverse=False,
                                             xvar=fault_trace_xvar, yvar=fault_trace_yvar ,zvar=fault_trace_zvar, dipvar=fault_trace_dipvar ,azivar=fault_trace_azivar,
                                            input_ID='Belfeld surface mapping', input_type='mapping', self_correlation=fault_mapping_corr, source=beld_surface_trace,
                                            max_depth_confidence=fault_depth_max, min_depth_confidence=fault_depth_min
                                            )
cal_mod9.add_shapefile_linestrings_and_compute_azimuth_to_interfaces(viers_surface_trace, fraction=1, z=31, formation='Viersen', dip=VIERSdip, azimuth_reverse=False,
                                             xvar=fault_trace_xvar, yvar=fault_trace_yvar ,zvar=fault_trace_zvar, dipvar=fault_trace_dipvar, azivar=fault_trace_azivar,
                                            input_ID='Viersen surface mapping', input_type='mapping', self_correlation=fault_mapping_corr, source=viers_surface_trace,
                                            max_depth_confidence=fault_depth_max, min_depth_confidence=fault_depth_min
                                            )
 
#-- 3 --# fault contacts with stratigraphic surfaces  (interpreatations)

# None implemented here

#-- 4 --#    surface contacts with seismic traces
nsg_interF = seismic_obs_bottom_nsg
carb_interF = seismic_obs_top_carb
Zee_interF = seismic_obs_top_zeeland

#No Orientations only interfaces 
'''
cal_mod7.add_shapefile_to_orients(file_list=[nsg_interF], zdepth=None,
                                             formation_field=None, azimuth_field=None, dip_field=None, formation='NorthSeaGroup',
                                             xvar=interF_xvar, yvar=interF_yvar ,zvar=interF_zvar, dipvar=interF_dipvar ,azivar=interF_azivar, sample_size=0.1)
cal_mod7.add_shapefile_to_orients(file_list=[carb_interF], zdepth=None, 
                                             formation_field=None, azimuth_field=None, dip_field=None, formation='Carboniferous',
                                             xvar=interF_xvar, yvar=interF_yvar ,zvar=interF_zvar, dipvar=interF_dipvar ,azivar=interF_azivar, sample_size=0.1)
cal_mod7.add_shapefile_to_orients(file_list=[Zee_interF], zdepth=None, 
                                             formation_field=None, azimuth_field=None, dip_field=None, formation='ZeelandFm',
                                             xvar=interF_xvar, yvar=interF_yvar ,zvar=interF_zvar, dipvar=interF_dipvar ,azivar=interF_azivar, sample_size=0.1)

cal_mod9.add_shapefile_to_interfaces(file_list=[nsg_interF], zdepth=None, 
                                            formation_field=None, azimuth_field=None, dip_field=None, formation='NorthSeaGroup',
                                            xvar=interF_xvar, yvar=interF_yvar ,zvar=interF_zvar, dipvar=interF_dipvar ,azivar=interF_azivar, sample_size=0.004,
                                            input_ID='NSG bottom seismic interpretation', input_type='seismic', self_correlation=NSG_corr, source=nsg_interF, 
                                            max_depth_confidence=NSG_depth_max, min_depth_confidence=NSG_depth_min
                                            ) 
# This is top Carbon, or rather Zechstein ------- cal_mod7.add_shapefile_to_interfaces(file_list=[carb_interF], zdepth=None, 
#                                            formation_field=None, azimuth_field=None, dip_field=None, formation='Carboniferous',
#                                             xvar=interF_xvar, yvar=interF_yvar ,zvar=interF_zvar, dipvar=interF_dipvar ,azivar=interF_azivar, sample_size=0.01)

cal_mod9.add_shapefile_to_interfaces(file_list=[Zee_interF], zdepth=None, 
                                            formation_field=None, azimuth_field=None, dip_field=None, formation='Namurian',
                                             xvar=interF_xvar, yvar=interF_yvar, zvar=interF_zvar, dipvar=interF_dipvar, azivar=interF_azivar, sample_size=0.004,
                                            input_ID='Namurian bottom seismic interpretation', input_type='seismic', self_correlation=namurian_corr, source=Zee_interF, 
                                            max_depth_confidence=Namurian_depth_max, min_depth_confidence=Namurian_depth_min
                                            )
'''

### Seismic interpretations


# This feature extend the Belfeld fault trace beyond the current surface trace... as seismic suggests it extends further.
cal_mod9.add_shapefile_to_interfaces(file_list=[surface_trace_beld_points], zdepth=33, 
                                            formation_field=None, azimuth_field=None, dip_field=None, formation='Belfeld',
                                            xvar=fault_trace_xvar, yvar=fault_trace_yvar, zvar=1, dipvar=fault_trace_dipvar, azivar=fault_trace_azivar, sample_size=0.3,
                                            input_ID='Belfeld surface mapping', input_type='mapping', self_correlation=fault_mapping_corr, source=surface_trace_beld_points,
                                            max_depth_confidence=fault_depth_max, min_depth_confidence=fault_depth_min
                                            )


#-- 5 -- #fault seismic contacts obs (Fault sticks from seismic profiles)

print("Brunton is adding Fault seismic contacts to observations")
print(style.RESET)

# fault_sticks_list = [fault_sticks_teg, fault_sticks_dulk, fault_sticks_vier, fault_sticks_beld] # if all other attributes were equal, this list could be passed into the CSV ingesters 

cal_mod9.add_csv_to_interfaces(file_list=[obs_fault_Tegelen], delimiter=",",zdepth=None, 
                                            formation_field=None, azimuth_field=None, dip_field=None, azimuth=52, dip=TEGdip, formation='Tegelen',
                                            xvar=fault_seismic_xvar, yvar=fault_seismic_yvar, zvar=fault_seismic_zvar, dipvar=fault_seismic_dipvar, azivar=fault_seismic_azivar, sample_size=1,
                                            input_ID='Tegelen seismic interp', input_type='seismic', self_correlation=fault_seismic_corr, source=obs_fault_Tegelen,
                                            max_depth_confidence=fault_seismic_max, min_depth_confidence=fault_seismic_min
                                            )

cal_mod9.add_csv_to_orients(file_list=[obs_fault_Tegelen], delimiter=",",zdepth=None, 
                                            formation_field=None, azimuth_field=None, dip_field=None, azimuth=52, dip=TEGdip, formation='Tegelen',
                                            xvar=fault_seismic_xvar, yvar=fault_seismic_yvar, zvar=fault_seismic_zvar, dipvar=fault_seismic_dipvar, azivar=fault_seismic_azivar, sample_size=1,
                                            input_ID='Tegelen seismic interp', input_type='seismic', self_correlation=fault_seismic_corr, source=obs_fault_Tegelen,
                                            max_depth_confidence=fault_seismic_max, min_depth_confidence=fault_seismic_min
                                            )

cal_mod9.add_csv_to_interfaces(file_list=[obs_fault_Dulkener],delimiter=",", zdepth=None,
                                            formation_field=None, azimuth_field=None, dip_field=None, azimuth=52, dip=DULKdip, formation='Dulkener',
                                            xvar=fault_seismic_xvar, yvar=fault_seismic_yvar, zvar=fault_seismic_zvar, dipvar=fault_seismic_dipvar, azivar=fault_seismic_azivar, sample_size=1,
                                            input_ID='Dulkener seismic interp', input_type='seismic', self_correlation=fault_seismic_corr, source=obs_fault_Dulkener,
                                            max_depth_confidence=fault_seismic_max, min_depth_confidence=fault_seismic_min
                                            )

cal_mod9.add_csv_to_orients(file_list=[obs_fault_Dulkener], delimiter=",",zdepth=None, 
                                            formation_field=None, azimuth_field=None, dip_field=None, azimuth=52, dip=DULKdip, formation='Dulkener',
                                            xvar=fault_seismic_xvar, yvar=fault_seismic_yvar, zvar=fault_seismic_zvar, dipvar=fault_seismic_dipvar, azivar=fault_seismic_azivar, sample_size=1,
                                            input_ID='Dulkener seismic interp', input_type='seismic', self_correlation=fault_seismic_corr, source=obs_fault_Dulkener,
                                            max_depth_confidence=fault_seismic_max, min_depth_confidence=fault_seismic_min
                                            )

cal_mod9.add_csv_to_interfaces(file_list=[obs_fault_Viersen],delimiter=",", zdepth=None, 
                                            formation_field=None, azimuth_field=None, dip_field=None, azimuth=250, dip=VIERSdip, formation='Viersen',
                                            xvar=fault_seismic_xvar, yvar=fault_seismic_yvar, zvar=fault_seismic_zvar, dipvar=fault_seismic_dipvar, azivar=fault_seismic_azivar, sample_size=1,
                                            input_ID='Viersen seismic interp', input_type='seismic', self_correlation=fault_seismic_corr, source=obs_fault_Viersen,
                                            max_depth_confidence=fault_seismic_max, min_depth_confidence=fault_seismic_min
                                            )

cal_mod9.add_csv_to_orients(file_list=[obs_fault_Viersen],delimiter=",", zdepth=None, 
                                            formation_field=None, azimuth_field=None, dip_field=None, azimuth=250, dip=VIERSdip, formation='Viersen',
                                            xvar=fault_seismic_xvar, yvar=fault_seismic_yvar, zvar=fault_seismic_zvar, dipvar=fault_seismic_dipvar, azivar=fault_seismic_azivar, sample_size=1,
                                            input_ID='Viersen seismic interp', input_type='seismic', self_correlation=fault_seismic_corr, source=obs_fault_Viersen,
                                            max_depth_confidence=fault_seismic_max, min_depth_confidence=fault_seismic_min
                                            )

cal_mod9.add_csv_to_interfaces(file_list=[obs_fault_Belfeld], delimiter=",", zdepth=None, 
                                            formation_field=None, azimuth_field=None, dip_field=None, azimuth=52, dip=BELDdip, formation='Belfeld',
                                            xvar=fault_seismic_xvar, yvar=fault_seismic_yvar, zvar=fault_seismic_zvar, dipvar=fault_seismic_dipvar, azivar=fault_seismic_azivar, sample_size=1,
                                            input_ID='Belfeld seismic interp', input_type='seismic', self_correlation=fault_seismic_corr, source=obs_fault_Belfeld,
                                            max_depth_confidence=fault_seismic_max, min_depth_confidence=fault_seismic_min
                                            )

cal_mod9.add_csv_to_orients(file_list=[obs_fault_Belfeld], delimiter=",",zdepth=None, 
                                            formation_field=None, azimuth_field=None, dip_field=None, azimuth=52, dip=BELDdip, formation='Belfeld',
                                            xvar=fault_seismic_xvar, yvar=fault_seismic_yvar, zvar=fault_seismic_zvar, dipvar=fault_seismic_dipvar, azivar=fault_seismic_azivar, sample_size=1,
                                            input_ID='Belfeld seismic interp', input_type='seismic', self_correlation=fault_seismic_corr, source=obs_fault_Belfeld,
                                            max_depth_confidence=fault_seismic_max, min_depth_confidence=fault_seismic_min
                                            )

cal_mod9.add_csv_to_interfaces(file_list=[obs_fault_Carb_teg_anti],delimiter=",", zdepth=None, 
                                            formation_field=None, azimuth_field=None, dip_field=None, azimuth=270, dip=CARBdip, formation='CarboniferousFaultTegAnti',
                                            xvar=fault_seismic_xvar, yvar=fault_seismic_yvar, zvar=fault_seismic_zvar, dipvar=fault_seismic_dipvar, azivar=fault_seismic_azivar, sample_size=1,
                                            input_ID='CarbTegAnti seismic interp', input_type='seismic', self_correlation=fault_seismic_corr, source=obs_fault_Carb_teg_anti,
                                            max_depth_confidence=fault_seismic_max, min_depth_confidence=fault_seismic_min
                                            )

cal_mod9.add_csv_to_orients(file_list=[obs_fault_Carb_teg_anti],delimiter=",", zdepth=None, 
                                            formation_field=None, azimuth_field=None, dip_field=None, azimuth=270, dip=CARBdip, formation='CarboniferousFaultTegAnti',
                                            xvar=fault_seismic_xvar, yvar=fault_seismic_yvar, zvar=fault_seismic_zvar, dipvar=fault_seismic_dipvar, azivar=fault_seismic_azivar, sample_size=1,
                                            input_ID='CarbTegAnti seismic interp', input_type='seismic', self_correlation=fault_seismic_corr, source=obs_fault_Carb_teg_anti,
                                            max_depth_confidence=fault_seismic_max, min_depth_confidence=fault_seismic_min
                                            )


""" Selective data pruning """

# Saving dataframe from "Brunton" observation aggregator object to an intermediary ALL INCLUSIVE dataframe  
orientationsALL = cal_mod9.orientsDF()
interfacesALL = cal_mod9.interfacesDF()


''' read in a CSV from a previous data ingesting run with brunton '''
#orientationsALL = pd.read_csv('cal_model_8_d_All_orientations.csv')
#interfacesALL = pd.read_csv('cal_model_8_d_All_interfaces.csv')

print("Pruning to relavent model extents")
# X, Y dimentions for the modeling extents [195000, 220000, 373000, 389000] below we exclude data that lies more than a km beyond the modeling extent, could vary based on other factors
orientations = orientationsALL[(orientationsALL['X'].between(194000, 221000)) & (orientationsALL['Y'].between(372000, 390000))].sample(frac=1)
interfaces = interfacesALL[(interfacesALL['X'].between(194000, 221000)) & (interfacesALL['Y'].between(372000, 390000))].sample(frac=1)

'''Column lables list'''
#["input_ID", 'input_type', 'correlation', 'doi', 'source', "X", "Y", "Z", 'formation', 'azimuth', 'dip', 'polarity','X_variance', 'Y_variance', 'Z_variance', 'azimuth_variance', 'dip_variance']

orientations = orientations[["input_ID", 'input_type', 'correlation', 'doi', 'source', 'X', 'Y', 'Z', 'azimuth','dip', 'polarity','formation','X_variance','Y_variance','Z_variance','azimuth_variance','dip_variance', 'min_depth_confidence', 'max_depth_confidence']]
interfaces = interfaces[["input_ID", 'input_type', 'correlation', 'doi', 'source', 'X', 'Y', 'Z','polarity','formation','X_variance','Y_variance','Z_variance','min_depth_confidence','max_depth_confidence']]
 
#orientations['formation'].unique()
print(style.GREEN +"Exporting CSV's")
#orients_file='cal_model_7b_interfaces.csv', interF_file='cal_model_7b_orientations.csv'
orientations.to_csv(f'./model_realizations/_{model_suffix}{parallel_computing_suffix}_{args.iter}_all_orientations.csv')
interfaces.to_csv(f'./model_realizations/_{model_suffix}{parallel_computing_suffix}_{args.iter}_all_interfaces.csv')
print(style.RESET)
