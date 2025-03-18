#!/usr/bin/env python
# coding: utf-8

# In[221]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# __Params__

# In[223]:


plt.rcParams["legend.frameon"] = False
plt.rcParams['legend.fontsize'] = 15

plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.labelweight'] = 'bold'

plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18  

plt.rcParams['font.size'] = 16

plt.rcParams['figure.dpi'] = 1000


# __Functions__

# * __data processing__

# In[226]:


#data load (from specified path) and basic processing
#data: data export form Bioscreen machines, plate_layot: description of well-strain-condition-technical repeat
#inputs: path to dataset and plate layout
def individual_repeats_data_processing(path_to_data, path_to_plate_layout):
    
    #raw data and plate layout load
    data= pd.read_csv(path_to_data,
                      skiprows= 2)
    plate_layout= pd.read_csv(path_to_plate_layout)
    #raw data unpivot
    data= pd.melt(data,
                  id_vars= ['Time', 'Blank'],
                  var_name= ['Well'],
                  value_name= 'OD600')
    #time column split
    data= data.assign(Hours= data.Time.apply(lambda x: x.split(':')[0]),
                      Minutes= data.Time.apply(lambda x: x.split(':')[1]))
    #dtype adjustment
    data= data.astype({'Blank':'Float32', 'Well':'Int32', 'OD600':'Float32', 'Hours':'Int32', 'Minutes':'Int32'}) 
    #hour timepoint calculation
    data= data.assign(Hours= data.Hours + data.Minutes/60)    
    #eliminating unnecessary columns
    data= data.loc[:, ['Blank', 'Well', 'OD600', 'Hours']]    
    #merge to plate_layout
    data= data.merge(plate_layout, how= 'left', on= 'Well')
    #reorder columns
    data= data.reindex(columns= ['Strain', 'TechnicalRepeat', 'Well', 'Hours', 'OD600', 'Blank', 'Medium', 'AsConcentration'])  
    #dropping NaNs
    data= data.dropna()
    
    return data

#remove specific technical replicates (wells), 'manual' selection based on the results of 'individual_repeats_visualisation'
#inputs: dataset (result of 'individual_repeats_data_processing'), list of wells to be removed
def data_clean(data, wells_to_drop):
    
    #remove specific wells
    data= data.loc[~data.Well.isin(wells_to_drop)]
    
    return data

#averaging of technical replicates for each strain-condition-timpoint (As concentration)
#inputs: dataset (result of 'data_clean')
def individual_repeats_tech_rep_avg(data):
    
    #grouping, averaging and sorting
    data= data.groupby(['Strain', 'Hours', 'Medium', 'AsConcentration']).agg({'OD600':'mean', 'Blank':'mean'}).reset_index()
    data= data.sort_values(['Strain', 'AsConcentration', 'Hours'], ascending= [False, True, True]).reset_index(drop= True)
    
    return data

#averaging of individual biological replicates, calculation of STD and MOE (CL 95%)
#inputs: final datasets for each biological repeat (results of 'individual_repeats_tech_rep_avg') 
def final_dataset(repeat1_data, repeat2_data, repeat3_data):
    
    #concatenating, grouping, transforming into lists and sorting
    data= pd.concat([repeat1_data, repeat2_data, repeat3_data])
    data= data.groupby(['Strain', 'Hours', 'Medium', 'AsConcentration']).agg({'Blank': 'mean',
                                                                              'OD600': list}).reset_index()
    data= data.sort_values(['Strain', 'AsConcentration', 'Hours'], ascending= [False, True, True]).reset_index(drop= True)
    
    #averaging and statistics (as a new column)
    data= data.assign(OD600Mean= data.OD600.apply(lambda x: np.array(x).mean()),
                      OD600STD= data.OD600.apply(lambda x: np.array(x).std(ddof= 1)),
                      OD600MOE95= data.OD600.apply(lambda x: t_margin_of_error_cl95(x)))
    
    #replace missing values (for biol. replicate) with np.NaN
    data= data.assign(OD600= data.OD600.apply(lambda x: np.round(np.append(np.array(x), np.NaN), 4) if len(x)< 3 else np.round(np.array(x),4)))
    
    #order the mutants alphabetically keeping the 'wt control' as the first
    ctrl_data= data.loc[data.Strain== 'wt control']
    muta_data_ordered= data.loc[data.Strain!= 'wt control']
    muta_data_ordered= muta_data_ordered.sort_values(['Strain', 'AsConcentration', 'Hours'])
    data= pd.concat([ctrl_data, muta_data_ordered], axis= 0)
    
    return data


# * __statistics functions__

# In[228]:


#margin of error: t-distribution, CL 95%, confidence intervals: mean +/- margin of error
#input: list of OD600 values from individual biological replicates (for a single timepoint)
def t_margin_of_error_cl95(data):
    cl=0.95
    std= np.std(data, ddof=1)
    n=len(data)
    
    std_err= std/np.sqrt(n)
    t_score = stats.t.ppf((1 + cl) / 2, df=n - 1)  
    t_margin_of_error = t_score * std_err
    return t_margin_of_error


# * __visualisation__

# In[230]:


#visualisation of all technical repeats for each strain-condition (for single biological repeat)
#inputs: individual biological repeat dataset (result of individual_repeats_data_processing)
def individual_repeats_visualisation(data, export):
    
    fig, ax= plt.subplots(len(data.Strain.unique()), 3, figsize= (6.4*3, 4.8*len(data.Strain.unique())), sharex= 'all', sharey= 'all')
    for row, strain in enumerate(data.Strain.unique()):
        #control conditions (0 mM As)
        _ctrl_condition_data= data.loc[(data.Strain==strain)&(data.AsConcentration==0)]
        _ctrl_condition_data_tech_rep1= _ctrl_condition_data.loc[_ctrl_condition_data.TechnicalRepeat==1]
        _ctrl_condition_data_tech_rep2= _ctrl_condition_data.loc[_ctrl_condition_data.TechnicalRepeat==2]
        _ctrl_condition_data_tech_rep3= _ctrl_condition_data.loc[_ctrl_condition_data.TechnicalRepeat==3]
        
        ax[row][0].plot(_ctrl_condition_data_tech_rep1.Hours,
                        _ctrl_condition_data_tech_rep1.OD600,
                        label= f'technical repeat 1, well: {_ctrl_condition_data_tech_rep1.Well.unique()[0]}',
                        color= 'Blue')
        ax[row][0].plot(_ctrl_condition_data_tech_rep2.Hours,
                        _ctrl_condition_data_tech_rep2.OD600,
                        label= f'technical repeat 2, well: {_ctrl_condition_data_tech_rep2.Well.unique()[0]}',
                        color= 'Orange')
        ax[row][0].plot(_ctrl_condition_data_tech_rep3.Hours,
                        _ctrl_condition_data_tech_rep3.OD600,
                        label= f'technical repeat 3, well: {_ctrl_condition_data_tech_rep3.Well.unique()[0]}',
                        color= 'Green')

        ax[row][0].legend(loc= 'lower right', frameon= False)
        ax[row][0].set_title(f'control conditions: {strain}', fontsize= 13, weight= 'bold')
        #0.5 mM As
        _0_5_mM_As_data= data.loc[(data.Strain==strain)&(data.AsConcentration==0.5)]
        _0_5_mM_As_data_tech_rep1= _0_5_mM_As_data.loc[_0_5_mM_As_data.TechnicalRepeat==1]
        _0_5_mM_As_data_tech_rep2= _0_5_mM_As_data.loc[_0_5_mM_As_data.TechnicalRepeat==2]
        _0_5_mM_As_data_tech_rep3= _0_5_mM_As_data.loc[_0_5_mM_As_data.TechnicalRepeat==3]
        
        ax[row][1].plot(_0_5_mM_As_data_tech_rep1.Hours,
                        _0_5_mM_As_data_tech_rep1.OD600,
                        label= f'technical repeat 1, well: {_0_5_mM_As_data_tech_rep1.Well.unique()[0]}',
                        color= 'Blue')
        ax[row][1].plot(_0_5_mM_As_data_tech_rep2.Hours,
                        _0_5_mM_As_data_tech_rep2.OD600,
                        label= f'technical repeat 2, well: {_0_5_mM_As_data_tech_rep2.Well.unique()[0]}',
                        color= 'Orange')
        ax[row][1].plot(_0_5_mM_As_data_tech_rep3.Hours,
                        _0_5_mM_As_data_tech_rep3.OD600,
                        label= f'technical repeat 3, well: {_0_5_mM_As_data_tech_rep3.Well.unique()[0]}',
                        color= 'Green')

        ax[row][1].legend(loc= 'lower right', frameon= False)
        ax[row][1].set_title(f'0.5 mM As: {strain}', fontsize= 13, weight= 'bold')
        #1 mM As
        _1_mM_As_data= data.loc[(data.Strain==strain)&(data.AsConcentration==1)]
        _1_mM_As_data_tech_rep1= _1_mM_As_data.loc[_1_mM_As_data.TechnicalRepeat==1]
        _1_mM_As_data_tech_rep2= _1_mM_As_data.loc[_1_mM_As_data.TechnicalRepeat==2]
        _1_mM_As_data_tech_rep3= _1_mM_As_data.loc[_1_mM_As_data.TechnicalRepeat==3]
        
        ax[row][2].plot(_1_mM_As_data_tech_rep1.Hours,
                        _1_mM_As_data_tech_rep1.OD600,
                        label= f'technical repeat 1, well: {_1_mM_As_data_tech_rep1.Well.unique()[0]}',
                        color= 'Blue')
        ax[row][2].plot(_1_mM_As_data_tech_rep2.Hours,
                        _1_mM_As_data_tech_rep2.OD600,
                        label= f'technical repeat 2, well: {_1_mM_As_data_tech_rep2.Well.unique()[0]}',
                        color= 'Orange')
        ax[row][2].plot(_1_mM_As_data_tech_rep3.Hours,
                        _1_mM_As_data_tech_rep3.OD600,
                        label= f'technical repeat 3, well: {_1_mM_As_data_tech_rep3.Well.unique()[0]}',
                        color= 'Green')

        ax[row][2].legend(loc= 'lower right', frameon= False)
        ax[row][2].set_title(f'0.5 mM As: {strain}', fontsize= 13, weight= 'bold')
        
    #export
    if export== True:
        plt.savefig(r"C:\Users\Jakub\Desktop\bioscreen_individ_fig.png", dpi= 1000)
    elif export== False:
        pass;
    else:
        raise ValueError(f"Invalid export argument: '{export}'. Expected: boolean ('True' or 'False').")

#growth curves visualisation
#inputs: final, averaged dataset (results of 'final_dataset'), As concentration and 'True' to export to desktop/'False' to not export 
def growth_curves(data, as_concentration, export):
    
    fig, ax= plt.subplots(figsize= (9.6, 7.2))
    data= data.loc[data.AsConcentration== as_concentration]
    colors= sns.color_palette("tab10", n_colors= len(data.Strain.unique()))
    
    #plotting
    for i, strain in enumerate(data.Strain.unique()):
        
        selected_strain= data.loc[data.Strain==strain]
        ax.plot(selected_strain.Hours,
                selected_strain.OD600Mean,
                lw= 4,
                label= f'{strain}' if strain=='wt control' else f'${strain}$',
                color= colors[i])
    
    ax.legend(frameon= False)
    ax.set_ylim(0, 1.8)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('$\mathregular{OD_{600}}$')
    
    #export
    if export== True:
        plt.savefig(r"C:\Users\Jakub\Desktop\bioscreen_fig.png", dpi= 1000)
    elif export== False:
        pass;
    else:
        raise ValueError(f"Invalid export argument: '{export}'. Expected: boolean ('True' or 'False').")
        
#growth curves visualisation for selected subset of strains
#inputs: final, averaged dataset (results of 'final_dataset'), As concentration, 'True' to export to desktop/'False' to not export and list of selected strains/mutants (excluding WT)
def growth_curves_selected_mutants(data, as_concentration, selected_mutants, export):
    
    fig, ax= plt.subplots(figsize= (9.6, 7.2))
    data= data.loc[(data.AsConcentration== as_concentration)&
                   ((data.Strain == 'wt control') | (data.Strain.isin(selected_mutants)))]
    colors= sns.color_palette("tab10", n_colors= len(data.Strain.unique()))
    
    #plotting
    for i, strain in enumerate(data.Strain.unique()):
        
        selected_strain= data.loc[data.Strain==strain]
        ax.plot(selected_strain.Hours,
                selected_strain.OD600Mean,
                lw= 4,
                label= f'{strain}' if strain=='wt control' else f'${strain}$',
                color= colors[i])
    
    ax.legend(frameon= False)
    ax.set_ylim(0, 1.8)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('$\mathregular{OD_{600}}$')
    
    #export
    if export== True:
        plt.savefig(r"C:\Users\Jakub\Desktop\bioscreen_fig.png", dpi= 1000)
    elif export== False:
        pass;
    else:
        raise ValueError(f"Invalid export argument: '{export}'. Expected: boolean ('True' or 'False').")
        
#growth curves visualisation with selected mutants highlighted (alpha and linewidth)
#inputs: final, averaged dataset (results of 'final_dataset'), As concentration, 'True' to export to desktop/'False' to not export and list of selected strains/mutants (excluding WT)
def growth_curves_highlighted_mutants(data, as_concentration, selected_mutants, export):
    
    fig, ax= plt.subplots(figsize= (9.6, 7.2))
    data= data.loc[data.AsConcentration== as_concentration]
    colors= sns.color_palette("tab10", n_colors= len(data.Strain.unique()))
    
    #plotting
    for i, strain in enumerate(data.Strain.unique()):
        
        alpha= 1 if strain== 'wt control' or strain in selected_mutants else 0.1  
        lw= 4 if strain== 'wt control' or strain in selected_mutants else 1.75
        
        selected_strain= data.loc[data.Strain==strain]
        ax.plot(selected_strain.Hours,
                selected_strain.OD600Mean,
                lw= lw,
                alpha= alpha,
                label= f'{strain}' if strain=='wt control' else f'${strain}$',
                color= colors[i])
    
    ax.legend(frameon= False)
    ax.set_ylim(0, 1.8)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('$\mathregular{OD_{600}}$')
    
    #lower alpha for non-selected strains in the legend
    legend= ax.legend()
    for label in legend.get_texts():
        raw_label= label.get_text().rstrip('$').lstrip('$') #get a raw text and remove italics formatting
        if raw_label in selected_mutants or raw_label== 'wt control':  
            label.set_alpha(1)  
        else:
            label.set_alpha(0.1)  

    #export
    if export== True:
        plt.savefig(r"C:\Users\Jakub\Desktop\bioscreen_fig.png", dpi= 1000)
    elif export== False:
        pass;
    else:
        raise ValueError(f"Invalid export argument: '{export}'. Expected: boolean ('True' or 'False').")


# ------------------------------------------------------------------------

