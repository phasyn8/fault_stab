import pandas as pd
import numpy as np
from scipy.stats import norm, expon, uniform
import pyro
import pyro.distributions as dist
import torch
#import scipy.linalg

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

class stochastic_engine(object):
    
    ''' This class encapsulates a GemPy formatted input (though will be extended to other implicit modeling softwares) of observations and their uncertainties
    and through a selected recasting algorithm 

    
    Parameters:
        data_path:(str) path to working directory (where you store you input files)
        output_location:(str) path to output directory
        output_prefix:(str) identifying prefix
        orients_file:(path) path to input CSV  
            CSV format needs to include a few items:
            e.g. for a simple normal dirtibution generator [import_label, type, data_source, X,Y,Z,azimuth,dip,polarity,formation,X_variance,Y_variance,Z_variance,dip_variance,azimuth_variance] 
        interF_file:(path) path to input CSV  
            CSV format needs to include a few items:
            e.g. for a simple normal dirtibution generator [import_label, type, data_source, X,Y,Z,azimuth,dip,polarity,formation,X_variance,Y_variance,Z_variance] 
    '''
    
    def __init__(self, data_path=None, output_path=None, orients_file=None, interF_file=None, output_prefix='No_entry', start_at_number=0):
        
        # output parameters
        self.output_path = output_path 
        self.output_prefix = output_prefix
        
        # input parameters
        #self.input_file = input_file 
        self.data_path = data_path
        self.orients_file = orients_file
        self.interF_file = interF_file

        #combining file path and input to use internally in the object
        self.orients_file_path = self.data_path+self.orients_file
        self.interF_file_path = self.data_path+self.interF_file
        # Load the CSV file
        try:
            self.interF =  pd.read_csv(self.interF_file_path)
        except Exception as e:
            raise ValueError(f"Error reading the CSV file: {e}")(self.interF_file_path)
        try:
            self.orients =  pd.read_csv(self.orients_file_path)
        except Exception as e:
            raise ValueError(f"Error reading the CSV file: {e}")(self.orients_file_path)
        #realization counter
        self.realization = start_at_number


    def return_interfaces_df(self):
        return self.interF
    
    def return_orients_df(self):
        return self.orients

    def show_parameters(self):

        print("Model input stocastic engine Parameters")
        print(f'Output path: {self.output_path}')
        print(f'Orients file: {self.orients_file_path}')
        print(f'Interface file: {self.interF_file_path}')
        print(f'Output Prefix: {self.output_prefix}')
        print(f"Current Realization count: {self.realization:04d}")


        print("Input observations orients head:")
        print(self.orients.head(5))

        print("Input observations interfaces head:")
        print(self.interF.head(5))


    def apply_distribution_operator(self, df, operator_func):
        """
        Applies a user-defined operator function to a DataFrame, modifying the values based on the function's logic.
        
        Args:
            df (pd.DataFrame): The input DataFrame.
            operator_func (callable): A function that operates on rows of the DataFrame and returns modified rows.
            
        Returns:
            pd.DataFrame: A modified DataFrame after applying the operator function.
        """
        try:
            return df.apply(operator_func, axis=1)
        except Exception as e:
            raise ValueError(f"Error applying the operator function: {e}")



    def compute_z_score(self, value, mean, std_dev):
        """
        Calculates the z-score for a given value based on the distribution's mean and standard deviation.

        Parameters:
            value (float): The value for which the z-score is to be calculated.
            mean (float): The mean of the distribution.
            std_dev (float): The standard deviation of the distribution.

        Returns:
            float: The z-score of the given value.
        """
        if std_dev == 0:
            raise ValueError("Standard deviation cannot be zero.")
        
        z_score = (value - mean) / std_dev
        return z_score


    # fit distribution with a distribution from an input set of datas 
    def fit_distributions(data=None, distribution_names=['normal', 'exponential', 'uniform'], draw=False):
        """
        Fits the input data to specified distributions and returns the best-fit parameters for each distribution.
        If data is None, returns a default uniform distribution over [0, 1].

        Parameters:
            data (array-like or None): Array of input data to fit, or None for default output.
            distribution_names (list or None): List of distribution names to fit ('normal', 'exponential', 'uniform').
                                            If None, defaults to fitting all supported distributions.

        Returns:
            dict: A dictionary where keys are distribution names and values are the fitted parameters for each distribution.
        """
        # Supported distributions
        supported_distributions = {
            'normal': norm,
            'exponential': expon,
            'uniform': uniform
        }
        #if data == []:
        #    data = None
        # Default to fitting all distributions if none are specified
        if distribution_names is []:
            distribution_names = list(supported_distributions.keys())
        
        fits = {}

        # Return a default uniform distribution if data is None
        if len(data) == 0:
            if draw:
                return uniform(0,0.1).rvs(), {'uniform': (0, 0.1), 'normal': (0,0.1)}
                #return {'uniform': (0, 0.1)}
            else:
                return {'uniform': (0, 0.1)}

        for dist_name in distribution_names:
            if dist_name not in supported_distributions:
                raise ValueError(f"Unsupported distribution: {dist_name}")
            
            # Retrieve the distribution from scipy.stats
            distribution = supported_distributions[dist_name]
            
            # Fit the data to the distribution and store the parameters
            params = distribution.fit(data)
            fits[dist_name] = params
        
        if draw:
            return norm(fits['normal'][0],fits['normal'][1]).rvs(), fits
            
        else:
            return fits



    # pole to vector and reverse conversion functions
    def dip_azimuth_to_pole_vector(dip, azimuth):
        """
        Converts dip and dip direction azimuth to a pole (normal) vector.
        
        Parameters:
            dip (float): Dip angle in degrees (0° to 90°).
            azimuth (float): Azimuth angle in degrees (0° to 360°).
        
        Returns:
            np.ndarray: Pole vector as [x, y, z].
        """
        # Convert angles from degrees to radians
        dip_rad = np.radians(dip)
        azimuth_rad = np.radians(azimuth)
        
        # Compute the components of the pole vector
        x = np.sin(dip_rad) * np.sin(azimuth_rad)
        y = np.sin(dip_rad) * np.cos(azimuth_rad)
        z = np.cos(dip_rad)
        
        return np.array([x, y, z])

    def pole_vector_to_dip_azimuth(pole_vector):
        """
        Converts a pole (X,Y,Z) normal vector to dip and azimuth.
        
        Parameters:
            pole_vector (np.ndarray): Pole vector as [x, y, z].
        
        Returns:
            tuple: (dip in degrees, azimuth in degrees)
        """
        # Normalize the vector to ensure it's a unit vector
        pole_vector = pole_vector / np.linalg.norm(pole_vector)
        x, y, z = pole_vector
        
        # Compute dip angle (inverse cosine of the z component)
        dip = np.degrees(np.arccos(z))
        
        # Compute azimuth (angle in the x-y plane)
        azimuth = np.degrees(np.arctan2(x, y))
        
        # Adjust azimuth to lie between 0° and 360°
        azimuth = azimuth % 360
        
        return dip, azimuth


    # normal estimator function
    def generate_orients_random_normal_sample(self, row):
        """
        Generates random samples from normal distribution for X, Y, Z, and theta based on their means and variances.
        
        Args:
            row (pd.Series): A row from the DataFrame.
            
        Returns:
            pd.Series: The modified row with updated X, Y, Z, and theta values.
        """
        try:
            # Extract means and variances from the row
            x_mean, y_mean, z_mean, dip_mean, azi_mean = row['X'],          row['Y'],          row['Z'],          row['dip'],          row['azimuth']
            x_var,  y_var,  z_var,  dip_var,  azi_var  = row['X_variance'], row['Y_variance'], row['Z_variance'], row['dip_variance'], row['azimuth_variance']
            
            # Generate random samples from normal distributions
            row['X'] = np.random.normal(x_mean, np.sqrt(x_var))
            row['Y'] = np.random.normal(y_mean, np.sqrt(y_var))
            row['Z'] = np.random.normal(z_mean, np.sqrt(z_var))

            row['dip'] = np.random.normal(dip_mean, np.sqrt(dip_var)) # This should be a vonmises random on a azimuth / dip pole
            row['azimuth'] = np.random.normal(azi_mean, np.sqrt(azi_var)) # This should be a vonmises random on a azimuth / dip pole 

            row['x_resultz'] = self.compute_z_score(row['X'], x_mean, np.sqrt(x_var))
            row['y_resultz'] = self.compute_z_score(row['Y'], y_mean, np.sqrt(y_var))
            row['z_resultz'] = self.compute_z_score(row['Z'], z_mean, np.sqrt(z_var))
            row['dip_resultz'] = self.compute_z_score(row['dip'], dip_mean, np.sqrt(dip_var))
            row['azi_resultz'] = self.compute_z_score(row['azimuth'], azi_mean, np.sqrt(azi_var))
            

            
            return row
        except KeyError as e:
            raise ValueError(f"Missing required column: {e}")
    
    def generate_interF_random_normal_sample(self, row):
        """
        Generates random samples from normal distribution for X, Y, Z, and theta based on their means and variances.
        
        Args:
            row (pd.Series): A row from the DataFrame.
            
        Returns:
            pd.Series: The modified row with updated X, Y, Z, and theta values.
        """
        try:
            # Extract means and variances from the row
            x_mean, y_mean, z_mean  = row['X'],          row['Y'],          row['Z']          
            x_var,  y_var,  z_var   = row['X_variance'], row['Y_variance'], row['Z_variance']
            
            # Generate random samples from normal distributions
            row['X'] = np.random.normal(x_mean, np.sqrt(x_var))
            row['Y'] = np.random.normal(y_mean, np.sqrt(y_var))
            row['Z'] = np.random.normal(z_mean, np.sqrt(z_var))

            #row['dip'] = np.random.normal(dip_mean, np.sqrt(dip_var)) # This should be a vonmises random on a azimuth / dip pole
            #row['azimuth'] = np.random.normal(azi_mean, np.sqrt(azi_var)) # This should be a vonmises random on a azimuth / dip pole 

            #evaluates Z-score for the ramdom picks and adds them to the dataframe
            row['x_resultz'] = self.compute_z_score(row['X'], x_mean, np.sqrt(x_var))
            row['y_resultz'] = self.compute_z_score(row['Y'], y_mean, np.sqrt(y_var))
            row['z_resultz'] = self.compute_z_score(row['Z'], z_mean, np.sqrt(z_var))
            #row['dip_resultz'] = self.compute_z_score(row['dip'], dip_mean, np.sqrt(dip_var))
            #row['azi_resultz'] = self.compute_z_score(row['azimuth'], azi_mean, np.sqrt(azi_var))
            

            
            return row
        except KeyError as e:
            raise ValueError(f"Missing required column: {e}")


    #Generate a sum by rows of a dataframe These are also represented in the generate sample functions, should be reduced? 
    def sample_weights(csv, col_list=['X_variance','Y_variance','Z_variance','azimuth_variance','dip_variance']):
        df = pd.read_csv(csv)
        weights = pd.Series(df[col_list].sum(axis=1))
        return weights


    def interF_sample_weights(interF_csv, col_list=['X_variance','Y_variance','Z_variance']):
        df = pd.read_csv(interF_csv)
        interF_weights = pd.Series(df[col_list].sum(axis=1))
        return interF_weights


    def orient_sample_weights(orient_csv, col_list=['X_variance','Y_variance','Z_variance','azimuth_variance','dip_variance']):
        df = pd.read_csv(orient_csv)
        orient_weights = pd.Series(df[col_list].sum(axis=1))
        return orient_weights

    '''
    def generate_sample(self, sample=1, weighted_by_combined_variance=True):
        

        output_file=f'{self.output_prefix}_{self.realization:04d}.csv'
        
        #Generate a sum by rows of [col_list] in a dataframe
        def sample_weights(df, col_list=['X_variance','Y_variance','Z_variance','azimuth_variance','dip_variance']):
            weights = pd.Series(df[col_list].sum(axis=1))
            return weights
        
        try:
            # Apply the operator function to the DataFrame apply_distribution_operator(df, operator_func)
            modified_df = self.apply_distribution_operator(self.obs, self.generate_random_normal_sample)
            
            #draw weights from the sum of variances
            if weighted_by_combined_variance:
                weights = sample_weights(modified_df)
            else:
                weights = None
            
            # Save the modified DataFrame to a new CSV
            modified_df.sample(frac=sample, weights=weights).to_csv(f'{self.output_path}{output_file}', index=False)
            print(f"Modified table written to {output_file}")
            version = self.realization
            self.realization = self.realization+1
            return f'{self.output_path}{output_file}', self.output_prefix, version
        except ValueError as e:
            print(e)
        
    '''
    def compute_depth_scalars(self, depths, min_depth=None, max_depth=None, confine_to_input_boundary=False, low_pass=0.00001, high_pass=0.999999, scale_type='linear'):
        """
        Computes an array of variance scalars based on input depths values.
        
        Parameters:
        depths (array-like): Array of depth values.
        min_depth (float, optional): Minimum depth threshold.
        max_depth (float, optional): Maximum depth threshold.
        confine_to_input_boundary (bool): set the scale boundary explicitly from ~0 to ~1, the same thing would happend if both (min_depth/max_depth == None )
        low_pass (float): minimum variance scalar reduction  (Sorry these are logically backwards because we "return 1-scalars")
        high_pass (float): maximum variance scalar value
        scale_type (str): Type of scaling ('linear', 'exponential', 'sqrt').

        Returns:
        np.array: Array of scalars normalized between ~0 and ~1.
        """
        depths = np.array(depths)
        
        # Set min and max depth if not provided
        if min_depth is None:
            min_depth = np.min(depths)
        if max_depth is None:
            max_depth = np.max(depths)
        
        # Ensure min and max depth are within the array bounds
        if confine_to_input_boundary:
            min_depth = max(min_depth, np.min(depths))
            max_depth = min(max_depth, np.max(depths))
        
        
        # Compute scalars
        scalars = np.zeros_like(depths, dtype=float)
        
        for i, depth in enumerate(depths): # this 
            if depth <= min_depth:
                scalars[i] = 0.00001
            elif depth >= max_depth:
                scalars[i] = 0.999999
            else:
                norm_depth = (depth - min_depth) / (max_depth - min_depth)
                    # the low/high pass filters are for numerical stability, mostly for the high pass as if this results to 0.0 bad things happen in the random selection
                if norm_depth > high_pass:
                    norm_depth = high_pass
                    print("highpass")
                if norm_depth < low_pass: 
                    norm_depth = low_pass
                    print("lowpass")
                if scale_type == 'linear':
                    scalars[i] = norm_depth
                elif scale_type == 'exponential':
                    scalars[i] = norm_depth ** 2
                elif scale_type == 'sqrt':
                    scalars[i] = np.sqrt(norm_depth)  
        
        return 1-scalars # returns array of input length scalar weights based on their numerical values (depthm, but could really be anything else too).  
    

    def build_cov_matrix(self, mean_values, formation_variance, weight_by_depth=False, min_depth_confid=None, max_depth_confid=None, weights_type='linear', correlation=.5, low_pass=0.00001, high_pass=0.999999, confine_to_input_boundary=False):
        """
        Computes a covariance matrix for inference may include normalized weighted scalars to scale variance based input values (like depth).
        
        Parameters:
        mean_values (array-like): Array of values.
        formation_variance: (float): maximum or general formation variance (stdDev**2)
        min_depth (float, optional): Minimum depth threshold.
        max_depth (float, optional): Maximum depth threshold.
        scale_type (str): Type of scaling ('linear', 'exponential', 'sqrt').
        
        Returns:
        np.array: matrix of covariance with the diagonal of variance.
        """
        num_points = len(mean_values)
        
        if weight_by_depth:
            scalars = self.compute_depth_scalars(mean_values, min_depth=min_depth_confid, max_depth=max_depth_confid, 
                                                 scale_type=weights_type, confine_to_input_boundary=confine_to_input_boundary, 
                                                 low_pass=low_pass, high_pass=high_pass)
            print("Scalars: ", scalars)
            variance = formation_variance * scalars 
        else:
            variance = formation_variance
        cov_matrix = np.min(variance) * (correlation * np.ones((num_points, num_points))) + (variance)*(np.eye(num_points)) - (np.min(variance) * (correlation * np.eye(num_points))+0.000001) #Adding point-zero^6-one for numerical stability
        return cov_matrix

    def draw_random_correlated_sample(self, mean_values, cov_matrix, label=f"randCorr_test"):
        new_vals = np.zeros(len(mean_values))

        # Convert mean and covariance to PyTorch float tensors
        mean_tensor = torch.tensor(mean_values, dtype=torch.float32)
        cov_tensor =torch.tensor(cov_matrix, dtype=torch.float32)
        #print("Covariance lines 1 + 2 :", cov_tensor[:2])
        #print("Mean Tensor", mean_tensor)

        # Sample new Z-values from a multivariate normal distribution
        new_vals = pyro.sample(
            f"sample_{label}",
            dist.MultivariateNormal(mean_tensor, cov_tensor)
        )

        return new_vals


    def generate_independent_random_sample(self, sample=1, weighted_by_combined_variance=True):

        #def generate_interF_sample(self, sample=1, weighted_by_combined_variance=True):
            

        interF_output_file=f'interF_{self.output_prefix}_{self.realization:04d}.csv'
            
            #Generate a sum by rows of [col_list] in a dataframe
            #def interF_sample_weights(self, col_list=['X_variance','Y_variance','Z_variance']):
        if weighted_by_combined_variance:
            interF_weights = pd.Series(self.interF[['X_variance','Y_variance','Z_variance']].sum(axis=1))
        else:
            interF_weights = None
        
                #return interF_weights
            
        #    try:
                # Apply the operator function to the DataFrame apply_distribution_operator(df, operator_func)
        interF_modified_df = self.apply_distribution_operator(self.interF, self.generate_interF_random_normal_sample)
                
                #draw weights from the sum of variances
        
                
                # Save the modified DataFrame to a new CSV
        interF_modified_df.sample(frac=sample, weights=interF_weights).to_csv(f'{self.output_path}{interF_output_file}', index=False)
        print(f"Modified table written to {interF_output_file}")
                #version = self.realization
                #self.realization = self.realization+1
                #return f'{self.output_path}{interF_output_file}', self.output_prefix
            #except ValueError as e:
            #    print(e)

        #def generate_orients_sample(self, sample=1, weighted_by_combined_variance=True):
            

        orients_output_file=f'orients_{self.output_prefix}_{self.realization:04d}.csv'
                
                #Generate a sum by rows of [col_list] in a dataframe
        #def orients_sample_weights(self, col_list=['X_variance','Y_variance','Z_variance','azimuth_variance','dip_variance']):
        #df = self.orients
        if weighted_by_combined_variance:
            orients_weights = pd.Series(self.orients[['X_variance','Y_variance','Z_variance','azimuth_variance','dip_variance']].sum(axis=1))
        else:
            orients_weights = None
        
                #return orients_weights
                
            #try:
                    # Apply the operator function to the DataFrame apply_distribution_operator(df, operator_func)
        orients_modified_df = self.apply_distribution_operator(self.orients, self.generate_orients_random_normal_sample)
                    
                    #draw weights from the sum of variances
                    
                    # Save the modified DataFrame to a new CSV
        orients_modified_df.sample(frac=sample, weights=orients_weights).to_csv(f'{self.output_path}{orients_output_file}', index=False)
        print(f"Modified table written to {orients_output_file}")
                #version = self.realization
                #self.realization = self.realization+1
        #        return f'{self.output_path}{orients_output_file}', self.output_prefix
         #   except ValueError as e:
         #       print(e)
        #interF_output, interF_out_prefix = generate_interF_sample(self, sample=sample, weighted_by_combined_variance=weighted_by_combined_variance)
        #orients_output, orients_out_prefix = generate_orients_sample(self, sample=sample, weighted_by_combined_variance=weighted_by_combined_variance)
        version = self.realization
        self.realization = self.realization+1

        return f'{self.output_path}{interF_output_file}', f'{self.output_path}{orients_output_file}', self.output_prefix, version
    


    """ Correlated Random Samples with Multivariate normals """

    def generate_correlated_random_sample(self, variable='all_cartesian', sample=1, weighted_by_combined_variance=True, depth_weighted=False, correlation=None):
        
        _variable = variable
        if _variable == 'all_cartesian':
            variables = ['X', 'Y', 'Z']
        elif _variable == 'all':
            variables = ['X', 'Y', 'Z']#, 'dip', 'azimuth']
        else:
            variables = _variable


        #try:
        #    self.interF[variable] == TypeError
        #except KeyError as e:
        #    raise ValueError(f"Missing required column: {e}")

        #Name the output file
        interF_output_file=f'interF_{self.output_prefix}_{self.realization:04d}.csv'
            
        # Create weight for random selection
        if weighted_by_combined_variance:
            interF_weights = pd.Series(self.interF[['X_variance','Y_variance','Z_variance']].sum(axis=1))  # This is not a true combined probability, for this we multiply!
        else:
            interF_weights = None
        

        # Find all unique formation "inputs_ID"
        input_grps = self.interF["input_ID"].unique()
        print(style.YELLOW)
        print("All groups for interfaces, ",input_grps)
        interF_modified_df = self.interF.copy()

        

        for grp in input_grps:
            print("Now working group: ",grp)
            for var in variables:
                print("correlating selection from: ", var)

                #extract indexes from the operating group and variable
                indexes = self.interF[self.interF['input_ID']==grp][var].index
                #pull the values at these index, which become the means of the random selection
                mean_values = self.interF.loc[indexes][var].to_numpy()
                #pull the max variance from this grp/var combination
                formation_variance = self.interF.loc[indexes][var+'_variance'].max() #[df['input_ID']==grp][var+'_variance'].max()

                #pull the max correlation value from the input dataframe (correlation should be between 0.001 and .999, near zero and near one)
                if correlation == None:
                    corr = self.interF.loc[indexes]['correlation'].max()
                else:
                    #Manual overwrite correlation from fucntion parameter
                    corr = correlation
                print("Num indexes ",len(indexes), " with correlation ", corr)
                #build covariance matrix
                cov_matrix = self.build_cov_matrix(mean_values, formation_variance, weight_by_depth=depth_weighted, 
                                                   min_depth_confid=self.interF.loc[indexes]['min_depth_confidence'].max(), max_depth_confid=self.interF.loc[indexes]['max_depth_confidence'].max(), 
                                                   weights_type='linear', correlation=corr, 
                                                   low_pass=0.00001, high_pass=0.999999, confine_to_input_boundary=False)
                new_vals = self.draw_random_correlated_sample(mean_values, cov_matrix, label=f"randCorr_{self.realization}")
            
                print(style.RED + f"These are the new values with correlation :{new_vals}")
                interF_modified_df.loc[indexes, var] = new_vals.numpy()



         # = self.apply_distribution_operator(self.interF, self.generate_interF_random_normal_sample)
                
        
                
        # Save the modified DataFrame to a new CSV
        interF_modified_df.sample(frac=sample, weights=interF_weights).to_csv(f'{self.output_path}{interF_output_file}', index=False)
        print(style.GREEN + f"Modified table written to {interF_output_file}"+ style.RESET)

        if _variable == 'all_cartesian':
            variables = ['X', 'Y', 'Z']
        elif _variable == 'all':
            variables = ['X', 'Y', 'Z', 'dip', 'azimuth']
        else:
            variables = _variable  

        orients_output_file=f'orients_{self.output_prefix}_{self.realization:04d}.csv'
                
                #Generate a sum by rows of [col_list] in a dataframe
        #def orients_sample_weights(self, col_list=['X_variance','Y_variance','Z_variance','azimuth_variance','dip_variance']):
        #df = self.orients
        if weighted_by_combined_variance:
            orients_weights = pd.Series(self.orients[['X_variance','Y_variance','Z_variance','azimuth_variance','dip_variance']].sum(axis=1))
        else:
            orients_weights = None
        
        input_grps = self.orients["input_ID"].unique()
        print("All groups for orients, ",input_grps)
        orients_modified_df = self.orients.copy()

        for grp in input_grps:
            print("Now working in group: ",grp)
            for var in variables:
                print("correlating selection from: ", var)
                #extract indexes from the operating group and variable
                indexes = self.orients.loc[self.orients['input_ID']==grp][var].index
                #pull the values at these index, which become the means of the random selection
                mean_values = self.orients.loc[indexes][var].to_numpy()
                #pull the max variance from this grp/var combination
                formation_variance = self.orients.loc[indexes][var+'_variance'].max() #[df['input_ID']==grp][var+'_variance'].max()
                #pull the max correlation value from the input dataframe (correlation should be between 0.001 and .999, near zero and near one)
                if correlation == None:
                    corr = self.orients.loc[indexes]['correlation'].max()
                else:
                    #Manual overwrite correlation from fucntion parameter
                    corr = correlation
                print("Num indexes ",len(indexes), " with correlation ", corr)
                #build covariance matrix
                cov_matrix = self.build_cov_matrix(mean_values, formation_variance, weight_by_depth=depth_weighted, 
                                                   min_depth_confid=self.orients.loc[indexes]['min_depth_confidence'].max(), max_depth_confid=self.orients.loc[indexes]['max_depth_confidence'].max(), 
                                                   weights_type='linear', correlation=corr, 
                                                   low_pass=0.00001, high_pass=0.999999, confine_to_input_boundary=False)
                new_vals = self.draw_random_correlated_sample(mean_values, cov_matrix, label=f"randCorr_{self.realization}")
                print(new_vals)
                orients_modified_df.loc[indexes, var] = new_vals.numpy()


                    # Save the modified DataFrame to a new CSV
        orients_modified_df.sample(frac=sample, weights=orients_weights).to_csv(f'{self.output_path}{orients_output_file}', index=False)
        print(style.GREEN + f"Modified table written to {orients_output_file}")
                #version = self.realization
                #self.realization = self.realization+1
        #        return f'{self.output_path}{orients_output_file}', self.output_prefix
         #   except ValueError as e:
         #       print(e)
        #interF_output, interF_out_prefix = generate_interF_sample(self, sample=sample, weighted_by_combined_variance=weighted_by_combined_variance)
        #orients_output, orients_out_prefix = generate_orients_sample(self, sample=sample, weighted_by_combined_variance=weighted_by_combined_variance)
        version = self.realization
        self.realization = self.realization+1

        return f'{self.output_path}{interF_output_file}', f'{self.output_path}{orients_output_file}', self.output_prefix, version
