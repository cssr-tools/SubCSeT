# DEBUG=False # switch for many parameters, should be False for deployment
DEBUG=True

import pandas as pd
import numpy as np
# plotly & dash
import plotly.express as px
from plotly.io import write_html, to_json, from_json
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash.dash_table.Format import Format, Scheme
from dash.dash_table import DataTable
from dash import html
from dash_bootstrap_templates import \
    load_figure_template, ThemeChangerAIO, template_from_url
import dash_bootstrap_components as dbc
from dash import callback_context
from dash import dcc
from dash import Dash
from copy import deepcopy
import time
import timeit
import os
import re
from datetime import datetime, timedelta
import json
# import warnings
# import utils
import webbrowser
# microchange2 to test the fork
# %% constants
themes = [
    "bootstrap", "cerulean", "cosmo", "cyborg", "darkly", "flatly",
    "journal", "litera", "lumen", "lux", "materia", "minty", "morph",
    "pulse", "quartz", "sandstone", "simplex", "sketchy", "slate",
    "solar", "spacelab", "superhero", "united", "vapor", "yeti", "zephyr"]

themes_options=[{'label': i, 'value': eval('dbc.themes.'+i.upper())} \
                for i in themes]

# theme0 = "cosmo"  # sets the theme
# theme0 = "bootstrap"  # sets the theme
theme0 = "journal"  # sets the theme
THEME0 = theme0.upper()

# %% Button to change the themes
c_theme = ThemeChangerAIO(
    aio_id="theme",
    radio_props={
        "value": eval('dbc.themes.'+THEME0),
        # "options": themes_options
             },
    button_props={
        # "children": [html.I(className="bi bi-palette"),'change theme'],
        "children": [html.Img(src="/assets/palette.svg"),'change theme'],
        'outline': True, "color": "dark",
        'size': 'md',
        'style': {'width': '100%'}
    },
    offcanvas_props={
        "placement": "start", "scrollable": True, 'style': {'width': '15vw'}
        }
)

# %% Utilities
def replace_none_colors(fig,color='grey'):
    '''replaces None values in FIG dict with the COLOR'''
    for trace in fig['data']:
        if 'marker' not in trace: continue
        clrs = trace['marker']['color'] 
        if isinstance(clrs,list):
            clrs = [color if x is None else x for x in clrs] 
            fig['data'][0]['marker']['color'] = clrs  
    return fig


def normalize_series(s, method='min-max'):
    if method == 'min-max':
        s_min,s_max = s.min(), s.max()
        return (s - s_min)/(s_max - s_min)
    elif method=='median':
        return s/s.median()
    elif method=='min':
        return s/s.mean()
    elif method=='z-score':
        return (s - s.mean())/s.std()
    else:
        return s


def round_to_sign_digits(x, sig_digits=3):
    '''Custom function to round to significant digits'''
    return np.format_float_scientific(x, precision=sig_digits-1)


def extract_number(text):
    # Regular expression pattern to find digits enclosed in square brackets
    pattern = r'\[(\d+)\]'
    match = re.search(pattern, text)
    if match:
        return int(match.group(1))  # Convert the found number to an integer
    else:
        return None

# %% app layout ----------------------------------------------------------------
# %%
dbc_css = (
    "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.2/dbc.min.css")

app = Dash(
    external_stylesheets=[
        eval('dbc.themes.' + THEME0),
        dbc_css,
        dbc.icons.BOOTSTRAP,
        dbc.icons.FONT_AWESOME
    ],
    suppress_callback_exceptions=True)

app.title = 'CCS screening tool for NCS'
server = app.server

c_inp_fldr = dbc.Input(
    id='inp_fldr', type='text',
    value=r'./data/_main.csv'
)

c_mtable = DataTable(id='mtable', data=[], selected_rows=[])
c_mtable = html.Div(c_mtable, id='mtable_div')

# %%
c_map = dcc.Graph(
    id='map_fig',
    style={
        'height': '89vh',
    },
    config={'displayModeBar': True}
)

c_b_save = dbc.Button(
    'save selected', id='save_button', n_clicks=0,
    className="me-1", size='md',
)

c_toolbar = dbc.ButtonGroup([
    dbc.Button(html.I(className="bi bi-check2-square"),
               size='md', id='b_select',
               outline=True, color="dark",
               ),
    dbc.Button(html.I(className="bi bi-x-square"),
               size='md', id='b_deselect',
               outline=True, color="dark",
               ),
    dbc.Button(html.I(className="bi bi-box-arrow-up-right"),
               id='b_reopen', size='md',
               outline=True, color="dark",
               ),
    dbc.Button(html.I(className="bi bi-gear-fill"),
               outline=True, color="dark",
               id='b_settings', size='md',
               ),  
    dbc.Button(html.I(className="bi bi-question-lg"),
               outline=True, color="dark",
               id='b_help', size='md',
               ),                 
])

c_help=dbc.Offcanvas(
    dcc.Markdown(id='help_markdown'),
    id='help', 
    is_open=not DEBUG, 
    scrollable=True, style={'width': '40vw'}
    )
# alternative help component
# c_help=dbc.Modal([
#     dbc.ModalBody(dcc.Markdown(id='help_markdown')),
#     ], id='help', is_open=True, size='lg', scrollable=True)

COLORSCALES=['rainbow','hot','jet','RdBu','Bluered','Portland','PuOr','Temps']

c_settings=dbc.Offcanvas(
    dbc.Stack([
        html.H1('Settings'),
        c_theme,
        dbc.Row([
            dbc.Col(
                dbc.InputGroup([
                    dbc.InputGroupText("map colorscale", 
                                       style={'width': '45%'}),
                    dbc.Select(
                        id='select_map_colorscale', value='Portland',
                        options=COLORSCALES
                        )
                ]),
            width=9),
            dbc.Col(
                dbc.Switch(id='switch_reverse_map_cs', 
                           label="reverse", value=False),
            width=3),
        ]),     
        dbc.Row([
            dbc.Col(
                dbc.InputGroup([
                    dbc.InputGroupText("scatter colorscale", 
                                       style={'width': '45%'}),
                    dbc.Select(
                        id='select_sc_colorscale', value='Portland',
                        options=COLORSCALES
                        )
                ]),
            width=9),
            dbc.Col(
                dbc.Switch(id='switch_reverse_sc_cs', 
                           label="reverse", value=False),
            width=3),
        ]), 
        dbc.Row([
            dbc.Col(
                dbc.InputGroup([
                    dbc.InputGroupText("para-plot colorscale", 
                                       style={'width': '45%'}),
                    dbc.Select(
                        id='select_para_colorscale', value='Portland',
                        options=COLORSCALES
                        )
                ]),
            width=9),
            dbc.Col(
                dbc.Switch(id='switch_reverse_para_cs', 
                           label="reverse", value=False),
            width=3),
        ]),         
        dbc.Row([
            dbc.Col(
                dbc.InputGroup([
                    dbc.InputGroupText("map style",style={'width': '45%'}),
                    dbc.Select(
                        id='select_map_style', value='carto-positron',
                        options=['open-street-map', 
                                'carto-positron', 
                                'carto-darkmatter']
                        )]
                ), width=9
            ),
            dbc.Col([],width=3), 
        ]),     
        dbc.Checkbox(
            id='checkbox_URL', 
            label="click on a field to open its page on factpages.sodir.no "+\
                "(may not work in the cloud)", 
            # style={'alignSelf': 'center'},
            value=False)  
    ]),
    id='settings', is_open=False, scrollable=True,
    style={'width': '30vw'}
)

@app.callback(
    Output('help', 'is_open'),
    Input('b_help', 'n_clicks'),
    prevent_initial_call=True,
)
def open_import_help(n):
    return True

@app.callback(
    Output('ptable_div', 'is_open'),
    Input('hide_para_table', 'n_clicks'),
    State('ptable_div', 'is_open'),
    prevent_initial_call=True,
)
def collapse_para_table(n, is_open):
    return not is_open

@app.callback(
    Output('settings', 'is_open'),
    Input('b_settings', 'n_clicks'),
    prevent_initial_call=True,
)
def open_settings(n):
    return True

c_map_tab = html.Div([
    dbc.Stack([
        dbc.Button(
            'update', id='update_map', n_clicks=0,
            color='danger',
            className="me-1", size='md',
        ),
        dbc.InputGroup([
            dbc.InputGroupText('size'),
            dbc.Select(id='map_dd_size', value='CO2 SC'),
            dbc.Button(html.I(className="bi bi-x-square"),
                       size='md', outline=True, color="dark",
                       id='map_size_reset'),
            ], style={'width': '30%'}
        ),
        dbc.InputGroup([
            dbc.InputGroupText('color'),
            dbc.Select(id='map_dd_color', value='field'),
            # dbc.Button(html.I(className="bi bi-x-square"),
            #            size='md', outline=True, id='map_color_reset'),            
            ], style={'width': '30%'}
        ),         
        ], gap=3, direction="horizontal"),
    c_map, # map itself
])

c_toolbar = dbc.Stack([
    c_toolbar,    
    # dbc.Tooltip('add all filtered rows to selection', target='b_select'),
    # dbc.Tooltip('deselect all', target='b_deselect'),
    # dbc.Tooltip('reopen chart in new tab', target='b_reopen'),
    c_inp_fldr,
    c_help,
    c_settings
],
    direction="horizontal", gap=1
)
#%% scatter plot
c_sc_b_update = dbc.Button(
    'update', id='update_sc', n_clicks=0,
    color='danger', className="me-1", size='md',
    # style={'width': '10%'}
)

c_sc = dcc.Graph(
    id='sc_fig',
    style={
        'height': '87vh',
    },
    config={'displayModeBar': True}
)

c_sc_tab = html.Div([
    dbc.Stack([
        c_sc_b_update,
        dbc.InputGroup([
            dbc.InputGroupText('X'),
            dbc.Select(id='sc_dd_x',value='lon'),
            ], style={'width': '20%'}),  
        dbc.InputGroup([
            dbc.InputGroupText('Y'),
            dbc.Select(id='sc_dd_y',value='lat'),
            ], style={'width': '20%'}),              
        dbc.InputGroup([
            dbc.InputGroupText('size'),
            dbc.Select(id='sc_dd_size',value='CO2 SC'),
            dbc.Button(html.I(className="bi bi-x-square"), size='md', 
                       outline=True, color="dark", id='sc_size_reset'),            
            ], style={'width': '25%'}),
        dbc.InputGroup([
            dbc.InputGroupText('color'),
            dbc.Select(id='sc_dd_color',value='q_resv'),
            dbc.Button(html.I(className="bi bi-x-square"), size='md', 
                       outline=True, color="dark", id='sc_color_reset'),
            ], style={'width': '25%'}),
        ], direction="horizontal"),
    c_sc
])
#%% Parallel plot

# c_ptable = DataTable(id='ptable', columns=[], data=[], editable=True)
c_ptable_div = dbc.Collapse([], id='ptable_div',is_open=True)

c_para_tab = html.Div([
    dbc.Stack([
        dbc.Button(
            'update', id='update_para', n_clicks=0,
            color='danger', className="me-1", size='md',
        ),
        dbc.InputGroup([
            dbc.InputGroupText('color'),
            dbc.Select(id='para_dd_color', value='p0'),
            dbc.Button(html.I(className="bi bi-x-square"),
                        size='md', outline=True, color="dark",
                        id='para_color_reset'),
        ],  style={'width': '40%'}),
        dbc.Button(
            ['+/- ', html.I(className="bi bi-table")], 
            id='hide_para_table', n_clicks=0,
            # color='danger',
            className="me-1", size='md',
        ),        
    ], direction="horizontal", gap=3
    ),
    c_ptable_div,
    dcc.Graph(
        id='para_fig',
        # style={'height': '87vh'},
        config={'displayModeBar': True}),
    html.Div(id='para_selected', style={'whiteSpace': 'pre-line'})
])
#%% total score

c_wtable = DataTable(id='wtable', columns=[], data=[], editable=True)
c_wtable_div = html.Div(c_wtable, id='wtable_div')

c_w_tab=html.Div([
    dbc.Button(
        'update', id='update_ts', n_clicks=0,
        color='danger', className="me-1", size='md',
    ),    
    c_wtable_div,
    dcc.Graph(id='ts_fig')
], 
# style={'display': 'grid'}
)


#%%
c_tabs = dbc.Tabs([
    dbc.Tab(
        c_map_tab, 
        label='MAP', tab_id='tab_map',
        active_tab_style={"fontWeight": "bold"},
        ),
    dbc.Tab(
        c_sc_tab, label='SCATTER PLOT', tab_id='tab_sc',
        active_tab_style={"fontWeight": "bold"}
        ),
    dbc.Tab(
        c_para_tab, label='PARA-PLOT', tab_id='tab_para',
        active_tab_style={"fontWeight": "bold"}
        ),
    dbc.Tab(
        c_w_tab, label='TOTAL SCORE', tab_id='tab_ts',
        active_tab_style={"fontWeight": "bold"}
        ) 
    ], id='all_tabs', active_tab='tab_sc'
)

app.layout = html.Div([
    dbc.Row([
        dbc.Col([c_toolbar, c_mtable], width=6,
                style={'padding-right': '0.5vw'}
                ),
        dbc.Col(c_tabs, width=6,
                # style={'padding-left': '0.25vw'}
                )
    ], 
        className="g-0",
    ),
    dcc.Store(id='theme_store', data=theme0),
    dcc.Store(id='para_store_df', data={}),
    dcc.Store(id='para_store_ranges', data={}),    
    html.Div(id='dummy_output', hidden=True)
],
style={
    # 'display': 'grid', 
    # to account for scrollbars
    'width': 'calc(100vw - (100vw - 100%))',
    'height': 'calc(100vh - (100vh - 100%))',
    },
className="dbc"
)


@app.callback(
    Output('mtable_div', 'children'),
    Output('ptable_div', 'children'),
    Output('wtable_div', 'children'),    
    Output('update_map', 'n_clicks'),
    Output('update_sc', 'n_clicks'),
    Output('update_para', 'n_clicks'),    
    Output('update_ts', 'n_clicks'),    
    Output('map_dd_size', 'options'),
    Output('map_dd_color', 'options'),
    Output('sc_dd_x', 'options'),
    Output('sc_dd_y', 'options'),
    Output('sc_dd_size', 'options'),
    Output('sc_dd_color', 'options'),
    Output('para_dd_color','options'),
    Output('help_markdown','children'),
    #
    Input('inp_fldr', 'value'),  
    State(ThemeChangerAIO.ids.radio("theme"), "value"), 
)
def initial_setup(path2csv, theme_url):

    with open(r'./assets/_main_columns.json', 'r') as f:
        CLMNS = json.load(f)

    with open(r'./assets/_help_columns.json', 'r') as f:
        HELP_CLMNS = json.load(f)

    df = pd.read_csv(path2csv)
    
    # all columns
    all_clmns = list(df.columns)

    for v in ['reservoir', 'FactPageUrl']: all_clmns.remove(v)

    # only numerical columns
    num_clmns = df.select_dtypes(include=['number','bool']).columns

    # adding another field column in the end to improve readability
    df['field2'] = df['field']
    CLMNS['field2']=deepcopy(CLMNS['field'])

    # loading the themes for charts
    # t0=time.time()
    theme  = template_from_url(theme_url)
    load_figure_template(theme)
    # load_figure_template(themes)
    # t1=time.time()
    # print(f'load template(s): {t1-t0:.3f} s')    

    # # Determine which DataFrame columns are not in the priority list
    # priority_columns = [i for i in FANCY_CLMNS if i in df.columns]     
    # other_columns = [col for col in df.columns if col not in priority_columns]
    # # Combine the lists, with priority columns first, then others
    # new_column_order = priority_columns + other_columns
    # df = df[new_column_order]

    #%% mtable
    clmns = []
    for k, v in CLMNS.items():
        clmns.append({"name": v, "id": k, 'hideable': True,
                      #   'selectable': True
                      })

    for col in ['in-place liq./OE ratio',
                'rec. cond.', 'rec. NGL',
                'rem. NGL', 'rem. cond.',
                'RF oil', 'RF gas', 'gas FVF', 'oil FVF',
                'maturity oil', 'maturity gas', 'inj. score',
                'CO2 SC', 'H2 SC', 
                'produced oil PV', 'produced gas PV',
                'produced HCPV', 'CO2 SC', 'H2 SC'
                ]:
        
        if not col in df.columns:
            continue

        i = df.columns.get_loc(key=col)
        clmns[i]['format'] =\
            Format(precision=3, scheme=Scheme.decimal_or_exponent)
        clmns[i]['type'] = 'numeric'

    preselected_rows = df[df['sea'] == 'NORTH'].index.to_list()

    # Setting up tooltips to include first column value and column name
    # tooltip_data = [
    #     {column: {'value': f"{df.iloc[i]['field']} - {column}", 'type': 'markdown'}
    #     for column in df.columns} for i in range(len(df))
    # ]

    tooltip_data = [
        {column: {'value': f"{df.loc[i,'field']} - {column}", \
                  'type': 'markdown'}
        for column in df.columns} for i in df.index
    ]    

    tooltip_header = deepcopy(HELP_CLMNS)
    for k in HELP_CLMNS:
        tooltip_header[k] = ['',tooltip_header[k], '']

    mtable = DataTable(
        id='mtable', columns=clmns,
        data=df.to_dict('records'),
        editable=True, filter_action="native",
        sort_action="native",  sort_mode="multi",
        row_selectable="multi",
        # tooltip_header=tooltip_header,
        merge_duplicate_headers=True,
        column_selectable='single',
        selected_columns=[],
        selected_rows=preselected_rows,
        tooltip_data=tooltip_data, 
        tooltip_header=tooltip_header,
        # fixed_columns={'headers': True, 'data': 1},
        page_action="native", page_current=0,
        hidden_columns=[
            '#','FactPageUrl', 'reservoir', 
            'res. quality','faulted',
            'recovery', 'fldID', 'GOR'
            'size', 'lat', 'lon',
            'depth min',  'depth mean', 'depth median', 'depth max',
            'net oil yearly pr.', 'net gas yearly pr.'
            'net NGL yearly pr.', 'net condensate yearly pr.',
            'net OE yearly pr.',  'water yearly pr.',
            ],
        style_table={
            'height': '90vh',
            # 'height': 'auto',
            # 'width': 'auto',
            'width': '49.5vw', 
            # 'width': 'calc(50vw - (50vw - 100%))',
            'overflowX': 'auto', 
            'overflowY': 'auto'
        },   
        style_cell={'fontSize': 14, 
                    'textAlign': 'center',
                    'whiteSpace': 'normal'}
    )
    #%% ptable
    clmns = [{'name': i, 'id': i} \
             for i in ['#','parameter','log10','normalize','reverse']]
    clmns[0]['type'] = 'numeric'
    clmns[0]['editable'] = False
    clmns[1]['presentation'] = 'dropdown'
    clmns[2]['presentation'] = 'dropdown'
    clmns[3]['presentation'] = 'dropdown'
    clmns[4]['presentation'] = 'dropdown'

    dropdowns = {}
    dropdowns['parameter']={}
    dropdowns['parameter']['options']=\
        [{'label': i, 'value': i} for i in num_clmns]
    
    dropdowns['normalize']={}
    dropdowns['normalize']['options']=\
        [{'label': i, 'value': i} for i in ['min-max','median',
                                            'mean', 'z-score']]  
    
    dropdowns['log10']={}
    dropdowns['log10']['options']=\
        [{'label': 'Yes', 'value': True}, {'label': 'No', 'value': False}]    
    
    dropdowns['reverse']={}
    dropdowns['reverse']['options']=\
        [{'label': 'Yes', 'value': True}, {'label': 'No', 'value': False}]

    pdata=[\
        {'#': 1, 'parameter': 'CO2 SC','normalize': None, 'log10': True,
         'reverse': False}, 
        {'#': 2, 'parameter': 'q_resv','normalize': None, 'log10': True,
         'reverse': False}, 
        {'#': 3, 'parameter': 'depth', 'normalize': None, 'log10': False,
         'reverse': True},          
        {'#': 4, 'parameter': 'peak year', 'normalize': None, 'log10': False,
         'reverse': True},  
        {'#': 5, 'parameter': None, 'normalize': None, 'log10': False,
         'reverse': True},             
        ]

    ptable = DataTable(
        id='ptable', columns=clmns, data=pdata,
        dropdown=dropdowns, editable=True,
        style_table={'width': '45vw'},
        style_cell={'fontSize': 14, 
                    'textAlign': 'center',
                    'whiteSpace': 'normal'} 
    )

    #%% total score table
    clmns = [{'name': i, 'id': i} \
             for i in ['#','parameter','log10','normalize','weight']]
    clmns[0]['type'] = 'numeric'
    clmns[0]['editable'] = False
    clmns[1]['presentation'] = 'dropdown'
    clmns[2]['presentation'] = 'dropdown'
    clmns[3]['presentation'] = 'dropdown'
    clmns[4]['type'] = 'numeric'

    dropdowns = {}
    dropdowns['parameter']={}
    dropdowns['parameter']['options']=\
        [{'label': i, 'value': i} for i in num_clmns]
    
    dropdowns['normalize']={}
    dropdowns['normalize']['options']=\
        [{'label': i, 'value': i} for i in ['min-max','median',
                                            'mean', 'z-score']]  
    
    dropdowns['log10']={}
    dropdowns['log10']['options']=\
        [{'label': 'Yes', 'value': True}, {'label': 'No', 'value': False}]    

    wdata=[\
        {'#': 1, 'parameter': 'CO2 SC','normalize': 'min-max', 'log10': True,
         'weight': 5}, 
        {'#': 2, 'parameter': 'q_resv','normalize': 'min-max', 'log10': True,
         'weight': 1}, 
        {'#': 3, 'parameter': 'depth', 'normalize': 'min-max', 'log10': False,
         'weight': -1},          
        {'#': 4, 'parameter': 'peak year', 'normalize': 'min-max', 'log10': False,
         'weight': -1},  
        {'#': 5, 'parameter': None, 'normalize': 'min-max', 'log10': False,
         'weight': 1},   
        ]

    wtable = DataTable(
        id='wtable', columns=clmns, data=wdata,
        dropdown=dropdowns, editable=True,
        style_table={'width': '45vw'},
        style_cell={'fontSize': 14, 
                    'textAlign': 'center',
                    'whiteSpace': 'normal'} 
    ) 
    #%% preparing help content
    # loading ...
    with open(r'./assets/_help.md', 'r') as file:
        markdown_help = file.read()  
    # ... adding columns to Glossary
    for key, value in HELP_CLMNS.items():
        nn = CLMNS[key][-1]
        markdown_help += f"{nn}. **{key}**: {value}  \n" 

    out = (
        mtable, ptable, wtable,  #
        1, 1, 1, 1, # initializes the plots
        num_clmns,  # options for map's size dropdown
        all_clmns,  # options for map's color dropdown
        num_clmns, # options for scatter's X dropdown
        num_clmns, # options for scatter's Y dropdown
        num_clmns, # options for scatter's size dropdown
        all_clmns, # options for scatter's color dropdown
        all_clmns, # options for para's color dropdown        
        markdown_help  # help text
    )
    return out
        
# %% select/deselect
@app.callback(
    Output('mtable', 'selected_rows'),
    Input('b_deselect', 'n_clicks'),
    Input('b_select', 'n_clicks'),    
    State('mtable', 'selected_rows'),
    State('mtable', 'derived_virtual_data'),
    prevent_initial_call=True
)
def select_deselect(m, b, selected_rows, filtered_rows):

    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'b_deselect' in changed_id:
        selected_rows = []
    elif 'b_select' in changed_id:
        selected_rows += [row['#'] for row in filtered_rows]
        selected_rows.sort()
        selected_rows = list(set(selected_rows))  # adds to previous selection
    else:
        # raise PreventUpdate
        pass
    selected_rows.sort()
    return selected_rows


@app.callback(
    Output('b_reopen', "n_clicks"),
    Input('b_reopen', "n_clicks"),
    State("all_tabs", 'active_tab'),
    State('map_fig', 'figure'),
    State('sc_fig', 'figure'),    
    State('para_fig', 'figure'),      
    prevent_initial_call=True
)

def reopen_current_chart(n, active_tab, fig_map, fig_sc, fig_para):

    fig = None
    # print('active tab:', active_tab)
    if active_tab == 'tab_map':
        fig=fig_map
    elif active_tab == 'tab_sc':
        fig=fig_sc   
    elif active_tab == 'tab_para':
        fig=fig_para           
    else:
        pass   

    if fig is not None: 
        # Stranglely enough Plotly refuses to render None values in fig
        # here is a temp. fix which replaces None values with "grey"
        # Maybe not so efficient ...
        fig = replace_none_colors(fig) 
        fig = go.Figure(fig)
        fig.show(renderer='browser', validate=False)   

    return n


@app.callback(
    Output('map_fig', 'figure'),
    Input('update_map', 'n_clicks'),
    Input('map_dd_color', 'value'),
    Input('map_dd_size', 'value'),
    Input('select_map_colorscale', 'value'),  
    Input('switch_reverse_map_cs', 'value'),  
    Input('select_map_style', 'value'),      
    State('mtable', 'selected_rows'),
    State('mtable', 'data'),
    State('theme_store', 'data'),
    # prevent_initial_call=True
)
def update_map(n, color, size, colorscale, reverse_colorscale, map_style,
               sel_rows, records, theme):

    if sel_rows is None or sel_rows == []: raise PreventUpdate
    fig = go.Figure()
    df = pd.DataFrame(data=records)
    if reverse_colorscale: colorscale += "_r"

    df = df.loc[sel_rows, :]
    # df.reset_index(inplace=True)

    # legacy block kept just in case
    # df['color'] = 'grey'  # preallocating
    # if df[color].dtype in ['float', 'int64', 'int32', 'int16', 'int8']:
    #     df = df.sort_values(by=color, ascending=False, kind='stable')
    #     df['color']=cssr.generate_rainbow_colors(df[color]*(-1 if invert else 1))
    # else:
    #     df = df.sort_values(by=color, ascending=True, kind='stable')
    #     # all other datatypes are treated as categoricals
    #     foo = df[color].unique()
    #     clrs = cssr.generate_rainbow_colors(len(foo))
    #     if invert: clrs = clrs[::-1]
    #     for i, clr in zip(foo, clrs):
    #         ind = df[color] == i
    #         df.loc[ind, 'color'] = clr

    if df[color].dtype in ['float', 'int64', 'int32', 'int16', 'int8']:
        pass
    else:
        df = df.sort_values(by=color, ascending=True, kind='stable')

    if size is None:
        df['size'] = 5
        _size = size
        size_max = 5
    else:
        size_min = 0.5
        size_max = 50
        df['size'] = df[size]
        # df['size'] = df[size]**0.5
        df['size'] = size_min + (100-size_min)*(df['size']-df['size'].min())/\
            (df['size'].max())
        df['size'] = df['size'].round(2)
        _size = "size"

    # legacy block kept just in case
    # for i in df.index:
    #     row = df.loc[i, :]
    #     url=row['FactPageUrl']
    #     sz = f"{size}: {row[size]}<br>" if size is not None else ""
    #     fig.add_trace(
    #         go.Scattermapbox(
    #             lat=[row['lat']], lon=[row['lon']],
    #             mode='markers', name=row['field'],
    #             legendgroup='fields', legendgrouptitle_text='fields',
    #             customdata=[url],
    #             hovertemplate=  # f"%{name}<br>"+
    #             f"{row['field']}<br>" +
    #             sz +
    #             f"{color}: {row[color]}<br>" +
    #             "click to open<br>"+
    #             "field's page on<br>"+
    #             "sodir.no<br>"+
    #             "<extra></extra>",
    #             marker={'size': row['size'], 'opacity': .5,
    #                     'symbol': 'circle',
    #                     # 'line': {'color': 'black'},
    #                     'color': row['color']}
    #         )
    #     )

    fig=px.scatter_mapbox(
        df, lat='lat', lon='lon',size=_size, color=color,
        hover_data=['field','lat','lon',size,color,'size'],
        size_max=size_max, 
        custom_data=['FactPageUrl'],
        color_continuous_scale=colorscale,
        color_discrete_sequence=px.colors.qualitative.G10,
        )

    ref_lat, ref_lon = df.loc[:, ['lat', 'lon']].mean().values

    fig.update_layout(
        template=theme,
        mapbox={
            'style': map_style,
            'center': go.layout.mapbox.Center(lat=ref_lat, lon=ref_lon),
            'zoom': 5.5,
        },
        # colorbar to the left
        coloraxis_colorbar=dict(x=0.0,  y=1.0, xanchor='left', yanchor='top'),
        #  top-right legend
        legend=dict(groupclick="toggleitem", 
                    x=1.0, y=1.0, xanchor='right', yanchor='top'),
        # lower-left legend 
        # legend=dict(groupclick="toggleitem", 
        #             x=0.0, y=0.0, xanchor='left', yanchor='bottom'),                    
        modebar_orientation='v',
        margin={"r": 30, "t": 0, "l": 0, "b": 0},
        modebar_add=['toggleHover', 'drawline', 'drawopenpath',
                     'drawclosedpath', 'drawcircle', 'drawrect',
                     'eraseshape', 'toggleSpikelines'],
    )
    fig.update_geos(fitbounds="locations")

    return fig

@app.callback(
    Output('sc_fig', 'figure'),
    Input('update_sc', 'n_clicks'),
    Input('sc_dd_x', 'value'),    
    Input('sc_dd_y', 'value'),    
    Input('sc_dd_color', 'value'),
    Input('sc_dd_size', 'value'),
    Input('select_sc_colorscale', 'value'),    
    Input('switch_reverse_sc_cs', 'value'),      
    State('mtable', 'selected_rows'),
    State('mtable', 'data'),
    State('theme_store', 'data')
    # prevent_initial_call=True
)
def update_sc(n, x, y, color, size, colorscale, reverse_colorscale,
              sel_rows, records, theme):
    
    df = pd.DataFrame(data=records)
    if reverse_colorscale: colorscale += "_r"

    if sel_rows is None or sel_rows == []:
        return go.Figure()
    
    df = df.loc[sel_rows, :]

    size_max = 10
    if size is not None:  
        df['s'] = 1
        df['s'] = df[size]
        df.s = df.s**0.5
        # df.s = 1 + 95*(df.s - df.s.min())/(df.s.max())
        df.s = df.s.round(2)
        size_max = 50
        size = df.s

    fig=px.scatter(
        df, x=x, y=y, color=color, size=size, size_max=size_max,
        template=theme, color_continuous_scale=colorscale,
        hover_data=['field', x, y, color, size]
        )    

    fig.update_layout(
        modebar_add=['toggleHover', 'drawline', 'drawopenpath',
                     'drawclosedpath', 'drawcircle', 'drawrect',
                     'eraseshape', 'toggleSpikelines'])
    return fig

@app.callback(
    Output('map_dd_size', 'value'),
    Input('map_size_reset', 'n_clicks'),
    prevent_initial_call=True
)
def map_size_reset(n):
    return None

@app.callback(
    Output('sc_dd_size', 'value'),
    Input('sc_size_reset', 'n_clicks'),
    prevent_initial_call=True
)
def sc_size_reset(n):
    return None

@app.callback(
    Output('para_dd_color', 'value'),
    Input('para_color_reset', 'n_clicks'),
    prevent_initial_call=True
)
def para_color_reset(n):
    return None

@app.callback(
    Output('sc_dd_color', 'value'),
    Input('sc_color_reset', 'n_clicks'),
    prevent_initial_call=True
)
def sc_color_reset(n):
    return None

#%% Theme change callback
@app.callback(
    Output('theme_store', 'data'),
    Output('map_fig', 'figure',allow_duplicate=True),
    Output('sc_fig', 'figure', allow_duplicate=True), 
    Output('para_fig', 'figure', allow_duplicate=True),   
    Output('ts_fig', 'figure', allow_duplicate=True),   

    Input(ThemeChangerAIO.ids.radio("theme"), "value"),
    State('map_fig', 'figure'),
    State('sc_fig', 'figure'),  
    State('para_fig', 'figure'), 
    State('ts_fig', 'figure'), 
    prevent_initial_call=True
)
def update_theme(theme_url, fig_map, fig_sc, fig_para, fig_ts):

    theme_str = template_from_url(theme_url)

    t0=time.time()
    load_figure_template(theme0)
    # load_figure_template(themes)
    t1=time.time()
    # print(f'load template(s): {t1-t0:.3f} s')  

    fig_map = replace_none_colors(fig_map)
    fig_map = go.Figure(fig_map).update_layout(template=theme_str)

    fig_sc = replace_none_colors(fig_sc)
    fig_sc = go.Figure(fig_sc).update_layout(template=theme_str)

    fig_para = go.Figure(fig_para).update_layout(template=theme_str)
    fig_ts = go.Figure(fig_ts).update_layout(template=theme_str)    

    return theme_str, fig_map, fig_sc, fig_para, fig_ts

# opens sodir.no field page on click
@app.callback(
    Output('dummy_output', 'children'), # dummy output
    Input('map_fig', 'clickData'),
    State('checkbox_URL','value'),
    prevent_initial_call=True
)
def open_FactPageUrl(clickData, open):
    "open the field's page on factpages.sodir.no/en"
    # print(clickData)
    if clickData is not None and open==True:
        url=clickData['points'][0]['customdata'][0]
        url = url[0] if isinstance(url,list) else url
        webbrowser.open(url)
    return ""


@app.callback(
    Output('para_fig', 'figure'),
    Output('para_store_df', 'data'),   
    Input('update_para', 'n_clicks'),
    Input('para_dd_color','value'),
    Input('select_para_colorscale','value'),
    Input('switch_reverse_para_cs','value'),
    State('mtable', 'selected_rows'),    
    State('mtable', 'data'),        
    State('ptable', 'data'),  
    State('theme_store', 'data'),
    prevent_initial_call=True
)
def update_para(n, color, colorscale, reverse_colorscale, 
                sel_rows, mrecords, precords,  theme):

    if sel_rows is None or sel_rows == []: raise PreventUpdate
    df = pd.DataFrame(data=mrecords)
    df = df.loc[sel_rows,:]
    pdf = pd.DataFrame(data=precords)
    pdf = pdf.dropna(subset='parameter')
    params = pdf['parameter'].to_list()  
    df = df.dropna(subset=params)
    if reverse_colorscale: colorscale += "_r"

    new_params = []
    for i in pdf.index:
        p = pdf.loc[i, 'parameter']
        p_new = p
        x = df[p].copy()
        if pdf.loc[i, 'log10']:  
            x=np.log10(x)
            p_new = f"log10({p})" 
            
        if pdf.loc[i, 'reverse']:  
            x = -x
            p_new = '-' + p_new

        if pdf.loc[i,'normalize']:
            x=normalize_series(x)
            p_new += "*"

        new_params.append(p_new)
        df[p_new] = x
    
    # old_labels_dict = dict(zip(params,new_params))
    # new_labels_dict = dict(zip(new_params,params))
    # pstore['old_labels_dict'] = old_labels_dict
    # pstore['new_labels_dict'] = new_labels_dict    

    fig = px.parallel_coordinates(
        df, dimensions=new_params, # labels=old_labels_dict, 
        color=color, color_continuous_scale=colorscale, template=theme
        )
    return fig, df.to_dict('records')

@app.callback(
    Output("para_selected", "children"),  
    Output("para_store_ranges", "data"),  
    #
    Input('update_para', 'n_clicks'),
    Input("para_fig", "restyleData"),
    Input("para_fig", "figure"),    
    State("para_store_ranges", "data"),
    State("para_store_df", "data"),
    prevent_initial_call=True
)
def display_selected_traces(n,restyleData, fig,  ranges, records):

    df = pd.DataFrame(data=records) 
    df = df.set_index('field')
    
    # fetch axes' labels
    labels = []
    for i in fig['data'][0]['dimensions']:
        labels.append(i['label'])

    
    # getting axes' selected ranges
    if restyleData is None: return None, ranges 

    key0 = list(restyleData[0].keys())[0]
    nn = extract_number(key0)
    # print('restyleData: ',restyleData)
    if nn is not None:
        ranges[labels[nn]] = restyleData[0][key0]
    # print(ranges)
    
    mask = pd.Series(np.ones(df.shape[0], dtype=bool), index=df.index) 
    for p,v in ranges.items():
        # print(v)  
        mask2 = pd.Series(np.zeros(df.shape[0], dtype=bool), index=df.index)          
        if v is None: 
            # mask *= mask2
            continue
        for lims in v: 
            # lu = lower/upper boundaries
            if isinstance(lims[0], list):
                for lu in lims:
                    mask2 += (df[p]>=lu[0]) * (df[p]<=lu[1])
            else: 
                mask2 += (df[p]>=lims[0]) * (df[p]<=lims[1])

        mask *= mask2

    out = list(df[mask == True].index)
    out = f"selected: {out}"
    # # "all-selected=>None-printed" way to treat
    # if mask.all():
    #     out = f"selected: ALL"
    # else:
    #     
    #     out = f"selected: {out}"

    return out, ranges


@app.callback(
    Output('mtable', 'data'),
    Output('ts_fig', 'figure'),    
    Input('update_ts', 'n_clicks'),    
    State('mtable', 'data'),
    State('mtable', 'selected_rows'),    
    State('wtable', 'data'),    
    State('theme_store', 'data'),
    prevent_initial_call=True
)
def update_ts(n, records, sel_rows, wrecords, theme):
    '''calculates total score column, updates the chart'''
    df = pd.DataFrame(data=records) # main df

    wdf = pd.DataFrame(data=wrecords)  # weights etc.
    wdf = wdf.dropna(subset='parameter')
    cdf = df[['field']].copy() # to store contributions
    # print(wdf)
    weight_sum=wdf['weight'].abs().sum()
    params = wdf['parameter'].to_list()

    df['total score'] = 0
    for i in wdf.index:
        p = wdf.loc[i, 'parameter']
        if p is None: continue
        
        x = df[p].copy()
        weight = wdf.loc[i, 'weight']

        if wdf.loc[i, 'log10']:  
            x=np.log10(x)

        if wdf.loc[i,'normalize']:
            x=normalize_series(x)
            
         # to correctly hangle weights<0, i.e. parameters to be minimized
        x = x*np.sign(weight)
        cdf[p] = 100*np.abs(weight)*(x - x.min())/(x.max()-x.min())/ weight_sum
        cdf[p] = cdf[p].round(1)
        df['total score'] += cdf[p]

    # df['total score'] = df['total score'].apply(round_to_sign_digits)
    df['total score'] = df['total score'].round(1)
    
    if sel_rows is None or sel_rows == []: raise PreventUpdate  
    cdf = cdf.loc[sel_rows, :]

    fig = None
    cdf['total score'] = df['total score'].round(1)
    cdf = cdf.dropna()
    cdf = cdf.sort_values(by='total score', ascending=False)
    cdf = cdf[:min(15,cdf.shape[0])]
    # print(cdf)

    fig = None
    fig = px.bar(cdf, x=params, y="field", orientation='h',log_x=False,
                 hover_data=['total score'], template=theme
                 )
    fig.update_layout(
        barmode='stack',
        yaxis=dict(autorange='reversed'), 
        xaxis=dict(title='total score and its components', side='top', 
                   range=(0,100)),
        )
    return df.to_dict('records'), fig

if __name__ == '__main__':
    if DEBUG:
        app.run(debug=True)  # should be False for deployment
    else:
        app.run(debug=False)  # should be False for deployment