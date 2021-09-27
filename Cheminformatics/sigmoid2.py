import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
IPythonConsole.ipython_useSVG=False  

import numpy as np
import random
from scipy import stats
from scipy.optimize import curve_fit
from scipy.optimize import minimize

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import umap 



def subset(dataset, strain):
    '''Subset the counts dataset to return bacterial counts for a given strain only.'''
    sub = dataset.loc[dataset['strain_name'] == strain]
    return sub

def statistics(dataset, compound):
    '''Calculate the mean and standard deviation of the bacterial counts at each dose for a given compound.'''
    data = dataset[dataset['pert_id'] == compound].groupby(['log_dose']).agg(['mean','std'])
    concentrations = data.index
    mean_counts = data['count']['mean'].values 
    mean_null_count = data['predicted_null_count']['mean'].values
    errors = data['count']['std'].values
    return concentrations, mean_counts, mean_null_count, errors 

def plot_plates(dataset, total, plot_compounds, size):
    '''Plot the dose response curve of a given antibiotic for each plate and the average over all the plates.'''
    random.seed(2)
    if len(plot_compounds) == 0:
        for i in range(total):
            compound = random.choice(dataset['pert_id'].unique())
            plot_compounds.append(compound)
    for compound in plot_compounds:
        data = dataset[dataset['pert_id'] == compound]
        plates = data['plate_pool'].unique()
        fig, ax = plt.subplots(nrows=1, ncols=len(plates)+1, figsize = size, squeeze = False)
        r,c = 0,0
        for plate in plates:
            plate_data = data[data['plate_pool'] == plate]
            concentrations = plate_data['log_dose']
            counts = plate_data['count']
            null_count = plate_data['predicted_null_count'].values[0]
            ax[r][c].scatter(concentrations, counts, marker = 'o')
            ax[r][c].axhline(y = null_count, ls = '--')
            ax[r][c].set_title(plate)
            c += 1     
        concentrations, mean_counts, mean_null_count, errors = statistics(dataset, compound)
        ax[r][len(plates)].errorbar(concentrations, mean_counts,fmt='o',  yerr = errors)
        ax[r][len(plates)].axhline(y = mean_null_count[0], ls = '--')
        ax[r][len(plates)].set_title('Mean counts')
        
        fig.legend(['--'],     labels=['Control'],   loc="center right",  borderaxespad=0.1)
        fig.suptitle(f'Dose response curves of {compound}:', y = 1.2)
        
        plt.show()

def logistic(x,b,c,d,e):
    '''Define a 4-parameter logistic curve where e is equivalent to the log(IC50).'''
    response = (c+(d-c)/(1+np.exp(b*(x-e))))     
    return response

def ic50(to_plot, dataset):
    '''Fit a logistic curve for all of the compounds in the dataset and then estimate the log(IC50).'''
    compounds = dataset['pert_id'].unique()
    total = len(compounds)
    if to_plot:
        r,c = 0,0
        rows = 2
        cols = 3
        fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize = (13,6.5), squeeze = False)
    logIC50 = {}
    no_fit = [] 
    type_err = []
    for i, compound in enumerate(compounds):
        concentrations, mean_counts, mean_null_count, errors = statistics(dataset, compound)
        concentrations = np.append(concentrations,(-7,20))
        mean_counts = np.append(mean_counts, (mean_null_count[0],0))
        try:
            Coefs, covMatrix = curve_fit(logistic, concentrations, mean_counts, maxfev=5000)
            logIC50[compound] = Coefs[3]
            predictions = logistic(concentrations, *Coefs)
            if to_plot and i <= (rows*cols): 
                logIC50_std = np.sqrt(np.diag(covMatrix))
                error = logistic(Coefs[3],*Coefs+logIC50_std)-logistic(Coefs[3], *Coefs)

                zipped_lists = zip(concentrations, mean_counts, predictions)
                sorted_pairs = sorted(zipped_lists)
                tuples = zip(*sorted_pairs)
                concentrations, mean_counts, predictions = [ list(tuple) for tuple in  tuples]
                conc_plot = np.linspace(concentrations[0],concentrations[-1],256)
                pred_plot = logistic(conc_plot, *Coefs)

                ax[r][c].set_title(compound)
                ax[r][c].scatter(concentrations, mean_counts)
                ax[r][c].plot(conc_plot, pred_plot, c = 'tab:red')
                ax[r][c].plot(Coefs[3], logistic(Coefs[3], *Coefs),'o', c= 'tab:green')

                c+=1
                if c%cols==0:
                    r += 1
                    c = 0
        except RuntimeError:
            no_fit.append(compound)
        except TypeError:
            type_err.append(compound)
    fig.legend(['o','-','o'],     labels=['Logistic curve','logIC50','Raw data'],   loc="center right",  borderaxespad=0.1)
    logIC50_df = pd.DataFrame(logIC50.items(), columns=['pert_id', 'log_IC50'])
    return logIC50_df, no_fit, type_err

def fingerprint(dataset, compound):
    '''Calculate the Morgan fingerprint given the SMILES string of a compound.'''
    smile = dataset.loc[dataset['pert_id'] == compound]['pert_canonical_smiles'].values[0]
    molecule = Chem.MolFromSmiles(smile)
    radius = 4
    fp = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=1024)
    return np.array(fp)

def fps(antibiotics, dataset):
    '''Determine the Morgan fingerprints for all the compounds in a given dataset.'''
    antibiotic_ls = dataset['pert_id'].tolist()
    fps = []
    for anti in antibiotic_ls:
        fp = fingerprint(antibiotics, anti)
        fps.append(fp)
    return fps

def draw_mols(names, dataset):
    smiles = []
    for n in names:
        smile = dataset[dataset['pert_id'] == n]['pert_canonical_smiles'].values[0]
        smiles.append(smile)
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    return Draw.MolsToGridImage(mols, molsPerRow=5, legends=names)

def potent(dataset, logIC50_df, inhibition):
    '''Return a subset of a dataset containing compounds with a logIC50 lower than the specified level of inhibition.'''
    potent_df = logIC50_df[logIC50_df['log_IC50'] < inhibition]
    potent_df = pd.merge(potent_df, dataset[['pert_id','pert_iname','pert_canonical_smiles']], how = 'inner', on = 'pert_id')
    return potent_df

def strain_IC50(dataset, strain):
    '''Given the dose response data for a certain strain, calculate the logIC50s for all the compounds tested on that strain.'''
    counts_strain = subset(dataset, strain)
    logIC50_strain, no_fit, type_err = ic50(False, counts_strain)
    return logIC50_strain

def target_to_id(string, t_dict):
    if string in t_dict.keys():
        return t_dict[string]
    else:
        return 'unknown target'