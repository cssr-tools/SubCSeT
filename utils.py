import pandas as pd
import numpy as np
import os
import colorsys

def get_production_monthly(update=False, 
                           p=r'./data/NOD/production_monthly.csv'):
    '''
    Parameters
    ----------
    update: bool
        if True (or "p" does not exists): fetch csv-data from FactPages and save to "p" 
        if False (and "p" exists): reload local data from "p"
    p: str
        path to a csv file
    '''
    if os.path.exists(p) and not update:
        df = pd.read_csv(p)
    else:
        # monthly by field saleable
        pp=r"https://factpages.sodir.no/public?/Factpages/external/tableview/field_production_monthly&rs:Command=Render&rc:Toolbar=false&rc:Parameters=f&IpAddress=not_used&CultureCode=en&rs:Format=CSV&Top100=false"
        # sum wellbores monthly by field
        # pp = r"https://factpages.sodir.no/public?/Factpages/external/tableview/field_production_gross_monthly&rs:Command=Render&rc:Toolbar=false&rc:Parameters=f&IpAddress=not_used&CultureCode=en&rs:Format=CSV&Top100=false"
        df = pd.read_csv(pp)
        df.to_csv(p)
    return df

def get_production_yearly(update=False, 
                          p = r'./data/NOD/production_yearly.csv'):
    '''
    Parameters
    ----------
    update: bool
        if True (or "p" does not exists): fetch csv-data from FactPages and save to "p" 
        if False (and "p" exists): reload local data from "p"
    p: str
        path to a csv file
    '''    
    if os.path.exists(p) and not update:
        df = pd.read_csv(p)
    else:
        pp=r"https://factpages.sodir.no/public?/Factpages/external/tableview/field_production_yearly&rs:Command=Render&rc:Toolbar=false&rc:Parameters=f&IpAddress=not_used&CultureCode=en&rs:Format=CSV&Top100=false"
        df = pd.read_csv(pp)
        df.to_csv(p)
    return df

def get_field_inplace_volumes(update=False, 
                              p=r'./data/NOD/inplace_volumes.csv'):
    '''
    Parameters
    ----------
    update: bool
        if True (or "p" does not exists): fetch csv-data from FactPages and save to "p" 
        if False (and "p" exists): reload local data from "p"
    p: str
        path to a csv file
    '''    
    if os.path.exists(p) and not update:
        df = pd.read_csv(p)
    else:
        pp=r"https://factpages.sodir.no/public?/Factpages/external/tableview/field_in_place_volumes&rs:Command=Render&rc:Toolbar=false&rc:Parameters=f&IpAddress=not_used&CultureCode=en&rs:Format=CSV&Top100=false"
        df = pd.read_csv(pp)
        df.to_csv(p)            
    return df

def get_field_reserves(update=False, 
                       p=r'./data/NOD/reserves.csv'):  
    '''
    Parameters
    ----------
    update: bool
        if True (or "p" does not exists): fetch csv-data from FactPages and save to "p" 
        if False (and "p" exists): reload local data from "p"
    p: str
        path to a csv file
    '''    
    if os.path.exists(p) and not update:
        df = pd.read_csv(p)
    else:
        pp=r"https://factpages.sodir.no/public?/Factpages/external/tableview/field_reserves&rs:Command=Render&rc:Toolbar=false&rc:Parameters=f&IpAddress=not_used&CultureCode=en&rs:Format=CSV&Top100=false"
        df = pd.read_csv(pp)
        df.to_csv(p) 
    return df

def get_all_wells(update=False, 
                  p=r'./data/NOD/wells_all.csv'):
    '''
    Parameters
    ----------
    update: bool
        if True (or "p" does not exists): fetch csv-data from FactPages and save to "p" 
        if False (and "p" exists): reload local data from "p"
    p: str
        path to a csv file
    '''    
    if os.path.exists(p) and not update:
        df = pd.read_csv(p)
    else:
        pp=r"https://factpages.sodir.no/public?/Factpages/external/tableview/wellbore_all_long&rs:Command=Render&rc:Toolbar=false&rc:Parameters=f&IpAddress=not_used&CultureCode=en&rs:Format=CSV&Top100=false"
        df = pd.read_csv(pp)
        df.to_csv(p, index=False) 
    return df

def get_development_wells(update=False, 
                          p=r'./data/NOD/wells_dev.csv'):
    '''
    Parameters
    ----------
    update: bool
        if True (or "p" does not exists): fetch csv-data from FactPages and save to "p" 
        if False (and "p" exists): reload local data from "p"
    p: str
        path to a csv file
    '''    
    if os.path.exists(p) and not update:
        df = pd.read_csv(p)
    else:
        pp=r"https://factpages.sodir.no/public?/Factpages/external/tableview/wellbore_development_all&rs:Command=Render&rc:Toolbar=false&rc:Parameters=f&IpAddress=not_used&CultureCode=en&rs:Format=CSV&Top100=false"
        df = pd.read_csv(pp)
        df.to_csv(p) 
    return df

def get_exploration_wells(update=False, 
                          p=r'./data/NOD/wells_exploration.csv'):
    '''
    Parameters
    ----------
    update: bool
        if True (or "p" does not exists): fetch csv-data from FactPages and save to "p" 
        if False (and "p" exists): reload local data from "p"
    p: str
        path to a csv file
    '''    
    if os.path.exists(p) and not update:
        df = pd.read_csv(p)
    else:
        pp=r"https://factpages.sodir.no/public?/Factpages/external/tableview/wellbore_exploration_all&rs:Command=Render&rc:Toolbar=false&rc:Parameters=f&IpAddress=not_used&CultureCode=en&rs:Format=CSV&Top100=false"
        df = pd.read_csv(pp)
        df.to_csv(p)
    return df

def get_wells_with_dst(update=False, 
                       p=r'./data/NOD/wells_with_dst.csv'):
    '''
    Parameters
    ----------
    update: bool
        if True (or "p" does not exists): fetch csv-data from FactPages and save to "p" 
        if False (and "p" exists): reload local data from "p"
    p: str
        path to a csv file
    '''    
    if os.path.exists(p) and not update:
        df = pd.read_csv(p)
    else:
        pp=r"https://factpages.sodir.no/public?/Factpages/external/tableview/wellbore_dst&rs:Command=Render&rc:Toolbar=false&rc:Parameters=f&IpAddress=not_used&CultureCode=en&rs:Format=CSV&Top100=false"            
        df = pd.read_csv(pp)
        df.to_csv(p)
    return df 

def get_field_status(update=False, 
                     p=r'./data/NOD/field_status.csv'):  
    '''
    Parameters
    ----------
    update: bool
        if True (or "p" does not exists): fetch csv-data from FactPages and save to "p" 
        if False (and "p" exists): reload local data from "p"
    p: str
        path to a csv file
    '''    
    if os.path.exists(p) and not update:
        df = pd.read_csv(p)
    else:    
        pp = r'https://factpages.sodir.no/public?/Factpages/external/tableview/field_activity_status_hst&rs:Command=Render&rc:Toolbar=false&rc:Parameters=f&IpAddress=not_used&CultureCode=en&rs:Format=CSV&Top100=false'
        df = pd.read_csv(pp)
        df.to_csv(p)
    return df

def get_field_overview(update=False, p=r'./data/NOD/field_overview.csv'):
    '''
    Parameters
    ----------
    update: bool
        if True (or "p" does not exists): fetch csv-data from FactPages and save to "p" 
        if False (and "p" exists): reload local data from "p"
    p: str
        path to a csv file
    '''    
    if os.path.exists(p) and not update:
        df = pd.read_csv(p)
    else:    
        pp = r'https://factpages.sodir.no/public?/Factpages/external/tableview/field&rs:Command=Render&rc:Toolbar=false&rc:Parameters=f&IpAddress=not_used&CultureCode=en&rs:Format=CSV&Top100=false'
        df = pd.read_csv(pp)
        df.to_csv(p, index=False)
    return df

def get_field_description(update=False, 
                          p=r'./data/NOD/field_description.csv'):  
    '''
    Parameters
    ----------
    update: bool
        if True (or "p" does not exists): fetch csv-data from FactPages and save to "p" 
        if False (and "p" exists): reload local data from "p"
    p: str
        path to a csv file
    '''    
    if os.path.exists(p) and not update:
        df = pd.read_csv(p)
    else:
        pp = r'https://factpages.sodir.no/public?/Factpages/external/tableview/field_description&rs:Command=Render&rc:Toolbar=false&rc:Parameters=f&IpAddress=not_used&CultureCode=en&rs:Format=CSV&Top100=false'
        df = pd.read_csv(pp)
        df.to_csv(p)
    return df


def generate_rainbow_colors(N, saturation=1.0, brightness=1.0, gap=.2):
    '''Generates N rainbow colors with specified saturation and brightness  
    which can be used in Plotly charts.

    Parameters
    ------------
    N : int or list-like
        number of colors or list-like item

    saturation : float
        
    brightness : float

    gap : float
        if <1, prevents the far right part of the spectrum  from coinciding with 
        the violet one in the left   

    Returns
    -------
    colors: list of strings 
        e.g. ['rgb(255,0,0)', 'rgb(51,255,0)', 'rgb(0,102,255)']

    Example
    -------
    # Generates and plots 20 colors:
    N = 20
    rainbow_colors = generate_rainbow_colors(N)
    fig = go.Figure(
        data=[go.Bar(x=list(range(N)), y=[1]*N, marker_color=rainbow_colors)])
    fig.show()   
    '''
    colors = []
    max_hue = 1.0-gap

    if isinstance(N,int): 
        NN = np.linspace(0,1,N) 
    else:  # if list-like ...
        # checks if list constists only on numerics
        if all(isinstance(x, (int, float)) for x in N):
            try:
                NN = (N - min(N))/(max(N)-min(N))
            except Exception as err_msg:
                NN = np.linspace(0,1,len(N))
                print(err_msg)
        else:
            NN = np.linspace(0,1,len(N))

    NN = max_hue*(1-NN)
    for n in NN:
        if isinstance(n,float) & (not np.isnan(n)):
            rgb = colorsys.hsv_to_rgb(n, saturation, brightness)
            colors.append(f'rgb({rgb[0]*255:.0f},{rgb[1]*255:.0f},{rgb[2]*255:.0f})')
        else:
            # replacing occasional nans with grey
            colors.append('grey')

    return colors