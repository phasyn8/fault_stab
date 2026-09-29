'''Gempy model builder class:
    Intended use for distributed probabalistic model generation or,
    if you would just like to create model objects from observation data'''

"""
Parameters:
    surface_points:    *.CSV for gempy surface point inputs ()
    orientations:      *.CSV for gempy formatted orientations
    extents:           [list] 3D model extents in any units you would like [xmin,xmax,ymin,ymax,zmin,zmax]
    formation_map_type:[dict] dictionary keys are ordered model formation groups, 
                                values must be a [list] of the element names [str], and a single variable, 
                                choices are from Stack Relation type (FAULT, ERODE, ONLAP) e.g. {"Befeld_fault" : ['Belfeld' , FAULT],
                                                                                                      "Cenezoic": ['NSG', ERODE],
                                                                                                     "Mesozoic" : ['Mesozoic', ONLAP],}
    fault_relations: [bool, matrix] must be square MxN dimensional matrix with lengths equal to the number groups defined in formation map
    data_path: [str] pointer to input data *.CSV (surface and orientation points)
    refinement: [int] refinement calculation in Gempy solver, default is 6
    project_name: [str] to set model metadata
    **kwargs: [TODO, pass kwargs to gempy_viewer class]
    

    kwargs:
        model_name: [str] desired model name
        data_path: [str] path to the input csv surface or orientation files
        topography: [str] path to " gempy.set_topography_from_file() " compatible file (like a GeoTIFF)  
"""



"""
Comes with some output functions:

    return_geo_data: this returns a computed geo-model from the input params

    return_3d_plot_inputs: This is a visualization function to look at input data

    TODO: 2D plots with an input data.
"""


import gempy as gp
import gempy_viewer as gpv
from gempy_engine.core.data.stack_relation_type import StackRelationType
import pyvista as pv
import numpy as np
from gempy_engine.config import AvailableBackends

class create_gempy_model(object):

    def __init__(self, surface_points, orientations, extents, resolution, formation_map_type, fault_relations, data_path, refinement=6, project_name='my project', **kwargs):
    
            
        # Stack realtions references, should probably be handled in the input...
            FAULT = StackRelationType.FAULT
            ERODE = StackRelationType.ERODE
            ONLAP = StackRelationType.ONLAP
            
            
        #Keyword Args assingment
            self.data_path = kwargs.get('data_path')   # eg.. 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
            self.project_name = kwargs.get('model_name')  # This is the project name
            self.gempy_backend = kwargs.get('gempy_backend')
            self.topography = kwargs.get('topography')
            self.chunk_size = kwargs.get('chunk_size')
            self.range_scaler = kwargs.get('range_scaler')

            if self.gempy_backend == None:
                self.gempy_backend = AvailableBackends.PYTORCH
            elif self.gempy_backend == "PYTORCH":
                self.gempy_backend = AvailableBackends.PYTORCH
            elif self.gempy_backend == 'numpy':
                self.gempy_backend = AvailableBackends.numpy
            else:
                self.gempy_backend = AvailableBackends.PYTORCH

            if self.project_name == '':
                self.project_name = 'No_Name'

            if self.chunk_size == None:
                self.chunk_size = 50_000
            
            if self.range_scaler == None:
                self.range_scaler = 1


    
        #Required arguments 
            self.surface_points = data_path + surface_points
            self.orients = data_path + orientations
            self.extent = extents
            
            #formations are held as the first value in the formation_map_type dictionary
            formation_map = {key: value[0] for key, value in formation_map_type.items()}

            #stack relation is held as
            stack_relation = {key: value[1] for key, value in formation_map_type.items()}
        
        # Create a GeoModel instance
            data = gp.create_geomodel(
                project_name=project_name,
                extent=self.extent,
                resolution=resolution,
                refinement=refinement,
                importer_helper=gp.data.ImporterHelper(
                    path_to_orientations=self.orients,
                    path_to_surface_points=self.surface_points
                    )
                )
        # Map geological series to surfaces
            gp.map_stack_to_surfaces(
                gempy_model=data,
                mapping_object=formation_map
                )
        # Define fault groups
            for i, relation in zip(range(len(stack_relation)), stack_relation):
                data.structural_frame.structural_groups[i].structural_relation = stack_relation[relation]
                
        # Set Fault relationships
            data.structural_frame.fault_relations = fault_relations
            
            if self.topography != None:
                gp.set_topography_from_file(grid=data.grid, filepath=self.topography)
                                
            #data.structural_frame.structural_groups[0].structural_relation = StackRelationType.FAULT
            #data.structural_frame.fault_relations = np.array([[0,0,0,1], [0,0,0,1],[0,0,0,1],[0,0,0,0]])
            #data.structural_frame.structural_groups[1].structural_relation = StackRelationType.FAULT
            #data.structural_frame.fault_relations = np.array([[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,0]])
            #data.structural_frame.structural_groups[2].structural_relation = StackRelationType.FAULT
        # Define NSG group
            #data.structural_frame.structural_groups[3].structural_relation = StackRelationType.ERODE
            
        # Define fault groups
            #data.structural_frame.structural_groups[2].structural_relation = StackRelationType.ERODE
        # Compute the geological model
        #    gp.compute_model(data)
        #    self.geo_data = data
            self.data = data
            
    def return_geo_data(self):
        # Compute the geological model with the model inputs and return to object
        
        #return gp.compute_model(self.data)
        self.data.interpolation_options.kernel_options.range *= self.range_scaler
        self.data.interpolation_options.evaluation_options.evaluation_chunk_size = self.chunk_size
        gp.compute_model(self.data, engine_config=gp.data.GemPyEngineConfig(
        backend=self.gempy_backend, dtype='float64'))
        self.geo_data = self.data
        return self.geo_data
    
    def return_3d_plot_inputs(self, show_data=True, show_boundaries=True, show_lith=False, kwargs_notebook_plotter=True):
        return gpv.plot_3d(self.data, show_data=show_data, show_boundaries=show_boundaries, show_lith=show_lith, kwargs_plotter={'notebook' : kwargs_notebook_plotter})

    def return_structural_frame(self):
        return self.data.structural_frame

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

