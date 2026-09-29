


start = 1   #  <--------- change this to the current model iteration, so you can build these models in chunks
n_MODEL_RUNS = 5

repo_path = 'src/'
results_path = ''

# importing packages
import argparse as arg
import matplotlib.pyplot as plt
#from PIL import Image

import sys
import os
sys.path.insert(1, './src/')

#import brunton_csv_builder as obs # this is the information handler class
import gempyModelBuilder as gpmb # GemPy builder class,
import proBot_depth_values as sto  # probabilistic modeling engine

import gempy as gp   
import numpy as np
import pandas as pd
import geopandas as gpd
from gempy_engine.core.data.stack_relation_type import StackRelationType
import datetime
import pyvista as pv


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

print(style.YELLOW + "Hello, We are now working to build all your structural models and export their formations surfaces ...")
print(style.RESET)
et = "NO DATA YET, BUILDING FIRST MODEL"
# METADATA

model_suffix = "cal_mod9_1_E"

slurm_parallel = arg.ArgumentParser(description='slurm array version')

slurm_parallel.add_argument('--iter', action='store', dest='iter', default=1)
slurm_parallel.add_argument('--hpc_suffix', action='store', dest='hpc', default='')

args = slurm_parallel.parse_args()

print(f"Comment : {args.hpc}")
print(f"Now computing iteration: {args.iter}")
#parallel_computing_suffix = args.iter

parallel_computing_suffix = ''

#pv.start_xvfb()
def return_mesh_from_gempy(geo_model, surface):
    """Gather vertices and faces to create polydata sets for meshing"""
    """ this is a temporary functions as this is working in gemgis v1.2  "create_depth_map_from gempy" or something like that"""
    
    """
    Parameters:
    geo_model (GemPy Computed geo-model): Model object from GemPy
    surface (int): Surface id from the computed geomodel

    Returns:
    Pyvista Polydata meshed Surface: 
    """
    
    
    #Collect vertices
    vertices = geo_model.input_transform.apply_inverse(geo_model.solutions.raw_arrays.vertices[surface])
    
    #Collect faces
    faces = np.hstack(np.pad(geo_model.solutions.raw_arrays.edges[surface], ((0, 0), (1, 0)), 'constant', constant_values=3))

    mesh = pv.PolyData(vertices, faces)
    mesh['Depth [m]'] = mesh.points[:, 2]

    return mesh

def create_contact_sheet(plotter, label, output_path="contact_sheet.jpg"):
    """
    Create a contact sheet with 4 different angles of a PyVista plotter object and save it as a JPG file.

    Parameters:
        plotter (pv.Plotter): A PyVista plotter object with the desired scene.
        label (str): A label to add to the contact sheet.
        output_path (str): File path to save the output JPG image.

    Returns:
        None
    """
    # Define camera positions for the 4 views
    camera_positions = [
        [1, 0, 1],  # +X
        [1,-1.3,0],  # +Y
        [0, 0, 1],  # +Z
        [1, -1, 1]  # -X -Y -Z (Diagonal view)
    ]

    # List to store the rendered images
    images = []

    # Retrieve actors from the original plotter
    actors = plotter.actors

    # Create a new Plotter for each view
    for pos in camera_positions:
        temp_plotter = pv.Plotter(off_screen=True)  # Off-screen rendering
        for actor in actors.values():
            temp_plotter.add_actor(actor)  # Add each actor to the new plotter

        temp_plotter.camera_position = pos
        image = temp_plotter.screenshot(transparent_background=False, return_img=True)
        images.append(image)
        temp_plotter.close()  # Ensure the plotter is closed after rendering

    # Assemble the images into a 2x2 grid
    rows, cols = 2, 2
    fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i])
        ax.axis('off')
    
    # Add the label below the grid
    fig.text(0.5, 0.1, label, ha='center', fontsize=16, weight='bold')

    # Save the assembled contact sheet as a JPG file
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(output_path, dpi=300, format="jpg")
    plt.close()
 
def create_and_export_geomodel(stochastic_obj, sample=0.6, project_name='NO_NAME_ENTERED', resolution = [100, 50, 40], extents = [195000,220000,373000,389000,-4000, 500] , data_path='./'):

    """
    Create a gempy model from a probabalistic draw from primary observations

    Paramerters:
    """
    
    FAULT = StackRelationType.FAULT
    ERODE = StackRelationType.ERODE
    ONLAP = StackRelationType.ONLAP
    BASEMENT = StackRelationType.BASEMENT


    #   **************   INDEPENDENT RANDOM SAMPLING ***************
    ''' 
    interF_output, orients_output, output_prefix, version = cal_StoOrients.generate_independent_random_sample(sample=0.6) #generate_sample
    print(f'You are currently on interface model {interF_output}')
    '''
    
    

    #   **************  CORRELATED RANDOM SAMPLING ******************
    
    interF_output, orients_output, output_prefix, version = stochastic_obj.generate_correlated_random_sample(variable='all', sample=sample, weighted_by_combined_variance=True, depth_weighted=True)
    
    print(f'You are currently on Orients model{output_prefix}{orients_output} and Interface model {output_prefix}{interF_output}')



    orients = orients_output
    surface_points = interF_output

    

    project_name = f'Realization_{version:04d}_{model_suffix}'
    extents = extents #[195000,220000,373000,389000,-3500, 500]   #[Xmin, Xmax, Ymin, Ymax, Zmin, Zmax]
    resolution = resolution #[120, 75, 120]
    formation_map_type = {
                        "SE_dipping_faults" : ['Viersen', FAULT],
                        "NE_intersecting_faults" : ['Dulkener', FAULT],
                        "NE_dipping_faults" : [('Belfeld', 'Tegelen'), FAULT],
                        "Cenezoic": ['NorthSeaGroup', ERODE],
                        "CarboniferousFaults" : ["CarboniferousFaultTegAnti", FAULT],
                        "Paleozoic" : [('Namurian', 'ZeelandFm' ), ERODE],
                    }
    ones = np.ones(9, dtype=int).reshape(3,3)
    all_zeros = np.zeros(36, dtype=int).reshape(6,6)
    zeros = np.zeros(9, dtype=int).reshape(3,3)
    se_faults = np.array([0,0,0,0,0,0])
    carb_fault = np.array([0,0,0,0,0,0])
    ne_faults = np.array([0,0,0,0,0,0])
    ne_intersecting = np.array([0,0,0,0,0,0])

    ones7 = np.ones(7)
    zero7 = np.zeros(7)
    stratpaleo = np.array([0,0,0,0,0,0])
    stratceno = np.array([0,0,0,0,0,0])
    #fault_relations = np.zeros(49).reshape(7,7)# np.hstack((np.vstack((zeros,zeros)),np.vstack((ones,zeros))))
    fault_relations = np.vstack((se_faults,ne_faults,ne_intersecting,stratceno,carb_fault,stratpaleo))
    print(datetime.datetime.now())
    print(style.YELLOW + f"the last model was build in {et}")
    #model = gpmb.create_gempy_model(surface_points, orients, extents, resolution, formation_map_type, fault_relations, data_path=data_path, refinement=6, project_name=project_name, gempy_backend='PYTORCH', chunk_size=100000, range_scaler=0.9).return_3d_plot_inputs()
    model = gpmb.create_gempy_model(surface_points, orients, extents, resolution, formation_map_type, fault_relations, data_path=data_path, refinement=6, project_name=project_name, gempy_backend='PYTORCH', chunk_size=100000, range_scaler=0.9).return_geo_data()
    print(style.RESET),
    return output_prefix, version, model

def create_sheet_from_geomodel(geo_model, label="NO_LABEL_ENTERED", output_path='./'):
    plotter = pv.Plotter(notebook=True)
    surf_dict = {0:'Viersen', 2: 'Belfeld', 3:"Tegelen", 1:'Dulkener', 4:'NSG', 5:'CarboniferousFault', 6:'Namurian', 7:'Zeeland'}
    color_dict = {0: 'red', 1: 'orange', 2: 'gold', 3:'pink', 4:'green', 5: 'grey', 6:'blue', 7:'cyan'}
    opacity_dict = {0:0.7, 1:0.7, 2:0.7, 3:0.7, 4:0.4, 5:0.4, 6:0.6, 7:0.6}
    #model_boundaries = (197500, 209500, 375500, 386500, -3000, 30)
    #xmin, xmax, ymin, ymax, zmin, zmax = model_boundaries
    #model_extents= (xmin, (xmax-xmin), ymin, (ymax-ymin), zmin, (zmax-zmin))
    for i in range(8):
        mesh = return_mesh_from_gempy(geo_model, surface=i)
        plotter.add_mesh(mesh, opacity=opacity_dict[i], label=surf_dict[i], color=color_dict[i])

    #plotter.add_mesh(pv.Box(model_boundaries).extract_feature_edges(), opacity=0.5)
    #plotter.add_mesh(pv.Box(model_boundaries))#.extract_feature_edges(), opacity=0.5)
    #plotter.add_legend()

    #plotter.camera.azimuth = 260
    #plotter.camera_position = [1,-1,1]
    #plotter.camera.elevation = 0
    #plotter.azimuth = 220
    #plotter.show()
    create_contact_sheet(plotter, label=label, output_path=f"{output_path}{label}_contact_sheet.jpg")

def export_cali_vtks(geo_model, file_path='./', model_suffix="no_suffix", version='9999'):
    vier = return_mesh_from_gempy(geo_model, surface=0)
    belf = return_mesh_from_gempy(geo_model, surface=2)
    tegel = return_mesh_from_gempy(geo_model, surface=3)
    dulk = return_mesh_from_gempy(geo_model, surface=1)
    NSGu = return_mesh_from_gempy(geo_model, surface=4)
    carbfault = return_mesh_from_gempy(geo_model, surface=5)
    Namu = return_mesh_from_gempy(geo_model, surface=6)
    Zeemu = return_mesh_from_gempy(geo_model, surface=7)

    vier.save(f'{file_path}Viersen_{model_suffix}_{version:04d}.vtk')
    tegel.save(f'{file_path}Tegelen_{model_suffix}_{version:04d}.vtk')
    belf.save(f'{file_path}Belfeld_{model_suffix}_{version:04d}.vtk')
    dulk.save(f'{file_path}Dulkener_{model_suffix}_{version:04d}.vtk')
    carbfault.save(f'{file_path}carbFaultTegAnti_{model_suffix}_{version:04d}.vtk')
    NSGu.save(f'{file_path}NSG_{model_suffix}_{version:04d}.vtk')
    Namu.save(f'{file_path}Namurian_{model_suffix}_{version:04d}.vtk')
    Zeemu.save(f'{file_path}Zeemu_{model_suffix}_{version:04d}.vtk')
 

#cal_StoOrients = sto.stochastic_engine(data_path='./model_realizations/', output_path='./model_realizations/', orients_file=f'_{model_suffix}_orientations.csv', interF_file=f'_{model_suffix}_interfaces.csv', output_prefix=model_suffix, start_at_number=start)


import subprocess
version = start

# using datetime module
#import datetime
#current_range = f"Single instance" #start+n_MODEL_RUNS
i=int(args.iter) # this is the current model run
#for i in range(start,current_range):
ct = datetime.datetime.now()
print(style.GREEN + f'************* Start Building Model instance {i} at current time: {ct}')
print(style.RESET)
subprocess.run(["python", f"_1{parallel_computing_suffix}_DATA_SELECTION_5fault_args.py", f'--iter={i}'])
cal_StoOrients = sto.stochastic_engine(data_path='./model_realizations/', output_path='./model_realizations/', orients_file=f'_{model_suffix}{parallel_computing_suffix}_{args.iter}_all_orientations.csv', interF_file=f'_{model_suffix}{parallel_computing_suffix}_{args.iter}_all_interfaces.csv', output_prefix=model_suffix, start_at_number=i) 

output_prefix_current, version, geo_model = create_and_export_geomodel(stochastic_obj=cal_StoOrients, sample=0.8, project_name=f'{model_suffix}_{i:04d}', resolution=[80, 33, 50], extents=[195000,220000,373000,389000,-3500, 300], data_path='')

print(f"Completed model version {i:04d}")

create_sheet_from_geomodel(geo_model, label=f"{output_prefix_current}_{version:04d}", output_path='./results/contact_sheets/')

export_cali_vtks(geo_model, file_path="./results/surfaces/", model_suffix=output_prefix_current, version=version)
bt = datetime.datetime.now()
et = bt-ct
print(style.YELLOW + f"Model {i} was build in {et} ")
print(style.RESET)
