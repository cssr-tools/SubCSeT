DEBUG=False # switch for many parameters, should be False for deployment
# DEBUG=True

import pandas as pd
import numpy as np
import io
# plotly & dash
import plotly.express as px
from plotly.io import write_html, write_image, to_json, to_html, to_image
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
from utils import generate_rainbow_colors
import webbrowser

# %% constants
tooltip_delay={'show': 750, 'hide': 0}

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
    elif method=='mean':
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
    config={'displayModeBar': True,'scrollZoom': True}
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
    dbc.Button(
        # html.I(className="bi bi-box-arrow-down-left"),
        html.I(className="bi bi-arrow-90deg-down"),
        id='b_chart_selection', size='md',
        outline=True, color="dark",
        ),
    dbc.Button(html.I(className="bi bi-gear-fill"),
               outline=True, color="dark",
               id='b_settings', size='md',
               ),  
    dbc.Button(html.I(className="bi bi-question-lg"),
            #    outline=True, 
               color="info",
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
        html.H2('Settings'),
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
        dbc.Row([
            dbc.Col(
                dbc.InputGroup([
                    dbc.InputGroupText("discrete colors",style={'width': '45%'}),
                    dbc.Select(
                        id='select_dclrs', value='Rainbow',
                        options=['Rainbow',
                                 'Plotly', 'D3', 'G10', 'T10', 'Alphabet',
                                 'Dark24','Light24','Vivid']
                        )
                    ]), 
            ),
            dbc.Col([],width=3), 
        ]),   
        dbc.Row(
            dbc.Col([
                html.Div('configure map and scatter tooltips:', 
                         style={'padding-top': '1vh'}),
                dcc.Dropdown(
                    value=['depth','grad_p0'], id='dd_configure_tooltips', 
                    multi=True,
                    placeholder='add/remove parameters in map and scatter tooltips'
                )
            ])
        ),
        # dbc.Switch(label='mousewheel/two-finger scroll to zoom the map',
        #            value=True,id='map_zoom_switch',
        #            style={'padding-top': '0.75vh'}),         
        dbc.Checkbox(
            id='checkbox_URL', 
            label="click on a field to open its page on factpages.sodir.no "+\
                "(may not work in the cloud)", 
            # style={'alignSelf': 'center'},
            value=False),
        dbc.Checkbox(
            id='checkbox_CO2_BAA', 
            label="show CO2 exploration and storage licenses ",
            # style={'alignSelf': 'center'},
            value=True),   
        html.Hr(),
        html.H2('Additional Tools'),   
        dbc.InputGroup([
            dbc.Button('save picture as',id='b_save_fig', color="primary"),
            dbc.Input(id="input_save_fig", placeholder="enter a name to override the default",
                    #   style={'width': '50%'}
                      ),
            dcc.Dropdown(id='select_save_fig_format',
                       options=['.png', '.jpeg', '.webp', '.svg', '.pdf', '.html'], 
                       value='.png', style={'width': '5vw'}, clearable=False), 
            dcc.Dropdown(id='select_save_fig_scale',
                       options=[{'label': 'x1', 'value': 1.0},\
                                {'label': 'x2', 'value': 2.0},
                                {'label': 'x3', 'value': 3.0},
                                ],
                       value=1.0, style={'width': '3vw'}, clearable=False),   
            dbc.Tooltip('scale', target='select_save_fig_scale',
                        delay=tooltip_delay),                       
        ]),
        dcc.Download("download"),
    ]),
    id='settings', is_open=False, scrollable=True,
    style={'width': '37vw'}
)

@app.callback(
    Output('help', 'is_open'),
    Input('b_help', 'n_clicks'),
    prevent_initial_call=True,
)
def open_import_help(n):
    return True

# @app.callback(
#     Output('map_fig', 'config'),
#     Input('map_zoom_switch', 'value'),
# )
# def set_map_zoom_mode(v):
#     '''mousewheel/two-finger scroll to zoom the map'''
#     return {'displayModeBar': True, 'scrollZoom': v}
    
@app.callback(
    Output('para_table_div', 'is_open'),
    Input('hide_para_table', 'n_clicks'),
    State('para_table_div', 'is_open'),
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
            ], style={'width': '45%'}
        ),
        dbc.InputGroup([
            dbc.InputGroupText('color'),
            dbc.Select(id='map_dd_color', value='q_resv'),
            # dbc.Button(html.I(className="bi bi-x-square"),
            #            size='md', outline=True, id='map_color_reset'),            
            ], style={'width': '45%'}
        ),         
        ], gap=2, direction="horizontal"),
    c_map, # map itself
])

c_toolbar = dbc.Stack([
    c_toolbar,    
    dbc.Tooltip('add all filtered rows to selection', target='b_select',
                delay=tooltip_delay),
    dbc.Tooltip('deselect all', target='b_deselect',
                delay=tooltip_delay),
    dbc.Tooltip('add selected in chart', target='b_chart_selection',
                delay=tooltip_delay),
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
        'height': '85vh',
    },
    config={'displayModeBar': True}
)

c_sc_tab = html.Div([
    dbc.Stack([
        c_sc_b_update,
        dbc.Stack([
            dbc.InputGroup([
                dbc.InputGroupText('X'),
                dbc.Select(id='sc_dd_x',value='depth'),
                ], # style={'width': '20%'}
            ),  
            dbc.InputGroup([
                dbc.InputGroupText('Y'),
                dbc.Select(id='sc_dd_y',value='CO2 density RC'),
                ], #style={'width': '20%'}
            ),  
        ]),
        dbc.Stack([
            dbc.Switch(label='log10',value=False,id='sc_x_log10',
                       style={'padding-top': '0.75vh'}
                       ), 
            dbc.Switch(label='log10',value=False,id='sc_y_log10',
                    #    style={'padding-top': '0.5vh'}
                       ),  
        ]),
        dbc.Stack([
            dbc.InputGroup([
                dbc.InputGroupText('size', style={'width': '20%'}),
                dbc.Select(id='sc_dd_size',value='CO2 SC'),
                dbc.Button(html.I(className="bi bi-x-square"), size='md', 
                        outline=True, color="dark", id='sc_size_reset'), 
                ], #style={'width': '25%'}
                ),
            dbc.InputGroup([
                dbc.InputGroupText('color', style={'width': '20%'}),
                dbc.Select(id='sc_dd_color',value='grad_p0'),
                dbc.Button(html.I(className="bi bi-x-square"), size='md', 
                        outline=True, color="dark", id='sc_color_reset'),
                ], #style={'width': '25%'}
                ),   
        ]),
        ], gap=2, direction="horizontal"),
    c_sc
])
#%% Parallel plot
c_para_tab = html.Div([
    dbc.Stack([
        dbc.Button(
            'update', id='para_update', n_clicks=0,
            color='danger', className="me-1", size='md',
        ),   
        dbc.ButtonGroup([
            dbc.Button(
                html.I(className="bi bi-plus"),
                size='md', id='para_plus',
                outline=True, color="dark",
                # className="me-1",
               ),
            dbc.Button(
                html.I(className="bi bi-table"),             
                id='hide_para_table', n_clicks=0,
                # color='danger',
                color="dark", outline=True, 
                # className="me-1", 
                size='md',
            ),  
            dbc.Button(
                html.I(className="bi bi-dash"), 
                size='md', id='para_minus',
                outline=True, color="dark",
                # className="me-1",
               ),
        ]),
        dbc.InputGroup([
            dbc.InputGroupText('color'),
            dbc.Select(id='para_dd_color', value='p0'),
            dbc.Button(html.I(className="bi bi-x-square"),
                        size='md', outline=True, color="dark",
                        id='para_color_reset'),
        ], style={'width': '40%'}), 
    ], direction="horizontal", gap=2),
    dbc.Collapse([], id='para_table_div',is_open=True), 
    dcc.Graph(
        id='para_fig', style={'height': '53vh'},
        config={'displayModeBar': True}),
    dbc.Container(
        id='para_selected', 
        style={'width': '95%','maxHeight':'12vh', 
               "overflowY": "auto", #    "border": "2px solid"
               }),           
])
#%% total score

c_ts_table = DataTable(id='ts_table', columns=[], data=[], editable=True)
c_ts_table_div = html.Div(c_ts_table, id='ts_table_div')

c_ts_tab=html.Div([
    dbc.Stack([
        dbc.Button(
            'update', id='ts_update', n_clicks=0,
            color='danger', className="me-1", size='md',
        ), 
        dbc.ButtonGroup([
            dbc.Button(
                html.I(className="bi bi-plus"),
                size='md', id='ts_plus',
                outline=True, color="dark",
                # className="me-1",
               ),
            dbc.Button(
                html.I(className="bi bi-table"),             
                id='hide_ts_table', n_clicks=0,
                color="dark", outline=True, 
                # className="me-1", 
                size='md',
            ),  
            dbc.Button(
                html.I(className="bi bi-dash"), 
                size='md', id='ts_minus',
                outline=True, color="dark",
                # className="me-1",
               ),
        ]),        
        dbc.Switch(label='only selected rows', value=True, id='ts_switch',
                   style={'padding-top': '0.5vh'}),
    ], direction='horizontal', gap=2,
    ),
    c_ts_table_div,
    dcc.Graph(id='ts_fig', style={'height': '65vh'},)
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
        c_ts_tab, label='TOTAL SCORE', tab_id='tab_ts',
        active_tab_style={"fontWeight": "bold"}
        ) 
    ], id='all_tabs', active_tab='tab_map',
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
    dcc.Store(id='units_info_store', data={}),
    dcc.Store(id='shape_store', data={}),    
    html.Div(id='dummy_output', hidden=True)
],
style={
    # 'display': 'grid', 
    # to account for scrollbars
    'padding-right': '0.25vw',
    # 'padding-left':  '0.25vw',
    'width': 'calc(100vw - (100vw - 100%))',
    'height': 'calc(100vh - (100vh - 100%))',
    },
className="dbc"
)


@app.callback(
    Output('mtable_div', 'children'),
    Output('para_table_div', 'children'),
    Output('ts_table_div', 'children'),    
    Output('update_map', 'n_clicks'),
    Output('update_sc', 'n_clicks'),
    Output('para_update', 'n_clicks'),    
    Output('ts_update', 'n_clicks'),    
    Output('map_dd_size', 'options'),
    Output('map_dd_color', 'options'),
    Output('sc_dd_x', 'options'),
    Output('sc_dd_y', 'options'),
    Output('sc_dd_size', 'options'),
    Output('sc_dd_color', 'options'),
    Output('dd_configure_tooltips', 'options'),
    Output('para_dd_color','options'),
    Output('help_markdown','children'),
    Output('units_info_store', 'data'),
    Output('shape_store', 'data'),    
    #
    Input('inp_fldr', 'value'),  
    State(ThemeChangerAIO.ids.radio("theme"), "value"), 
)
def initial_setup(path2csv, theme_url):

    with open(r'./assets/_main_columns.json', 'r') as f:
        CLMNS = json.load(f)

    with open(r'./assets/_help_columns.json', 'r') as f:
        HELP_CLMNS = json.load(f)
    
    # uploading shapes to visualize CO2 expl. and storage licenses
    with open(r'./assets/baa_shapes.json', "r") as f:
        SHAPES  = json.load(f)      
    # keeping only CO2 licenses ...
    SHAPES = [f for f in SHAPES if f['baaName'][:3] in ['EL0', 'EXL']]

    df = pd.read_csv(path2csv)
    
    # all columns
    all_clmns = list(df.columns)

    for v in ['reservoir', 'FactPageUrl']: all_clmns.remove(v)

    # only numerical columns
    num_clmns = df.select_dtypes(include=['number','bool']).columns
    
    #%% preparing help content and options for dropdowns
    # loading ...
    with open(r'./assets/_help.md', 'r') as file:
        markdown_help = file.read()  
    # ... adding columns to Glossary and ...
    # creating options for the dropdowns
    _num_clmns = []
    _all_clmns = []    
    for key, value in HELP_CLMNS.items():
        nn = CLMNS[key][-1]
        markdown_help += f"{nn}. **{key}**: {value}  \n" 
        foo = {'label': f'{nn}. {key}', 'value': key}
        _all_clmns.append(foo)
        if key in num_clmns:
            _num_clmns.append(foo)
    # saving columns units and descriptions for further use
    UNITS_INFO={}
    for key, value in CLMNS.items():
        UNITS_INFO[key] = {'info': HELP_CLMNS[key], 'unit': CLMNS[key][2]}

    #%% adding another field column in the end to improve readability
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

    # preselected_rows = df[df['sea'] == 'NORTH'].index.to_list()
    preselected_rows = df.index.to_list()

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
        tooltip_header[k] = ['',tooltip_header[k], '', tooltip_header[k]]
        # tooltip_header[k] = tooltip_header[k]

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
            'recovery', 'fldID', 'PL/BAA',
            'size', 'lat', 'lon',
            'depth min',  'depth mean', 'depth median', 'depth max',
            'peak year', 'peak OE YPR', 'peak oil YPR', 'peak gas YPR',
            'q_gas','qi_gas', 'q_gas2', 'qi_resv',
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
    #%% para_table
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
    dropdowns['parameter']['options']=_num_clmns
    # dropdowns['parameter']['options']=\
    #     [{'label': i, 'value': i} for i in num_clmns]
    
    dropdowns['normalize']={}
    dropdowns['normalize']['options']=\
        [{'label': i, 'value': i} for i in ['min-max','median',
                                            'mean', 'z-score']]  
    
    dropdowns['log10']={}
    dropdowns['log10']['options']=\
        [{'label': 'Yes', 'value': True}, {'label': 'No', 'value': False}]    
    dropdowns['log10']['clearable'] = False  
    
    dropdowns['reverse']={}
    dropdowns['reverse']['options']=\
        [{'label': 'Yes', 'value': True}, {'label': 'No', 'value': False}]
    dropdowns['reverse']['clearable'] = False

    pdata=[\
        {'#': 1, 'parameter': 'CO2 SC','normalize': None, 'log10': True,
         'reverse': False}, 
        {'#': 2, 'parameter': 'q_resv','normalize': None, 'log10': True,
         'reverse': False}, 
        {'#': 3, 'parameter': 'depth', 'normalize': None, 'log10': False,
         'reverse': True},          
        # {'#': 4, 'parameter': None, 'normalize': None, 'log10': False,
        #  'reverse': True},  
        # {'#': 5, 'parameter': None, 'normalize': None, 'log10': False,
        #  'reverse': True},             
        ]

    para_table = DataTable(
        id='para_table', columns=clmns, data=pdata,
        dropdown=dropdowns, editable=True,
        style_table={
            'width': '47vw',
            'maxHeight': '23vh',
            'overflowY': 'auto'
            },
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
    dropdowns['parameter']['options']=_num_clmns
    # dropdowns['parameter']['options']=\
    #     [{'label': i, 'value': i} for i in num_clmns]
    
    dropdowns['normalize']={}
    dropdowns['normalize']['options']=\
        [{'label': i, 'value': i} for i in ['min-max','median',
                                            'mean', 'z-score']]  
    dropdowns['normalize']['clearable'] = False
    
    dropdowns['log10']={}
    dropdowns['log10']['options']=\
        [{'label': 'Yes', 'value': True}, {'label': 'No', 'value': False}] 
    dropdowns['log10']['clearable'] = False

    wdata=[\
        {'#': 1, 'parameter': 'CO2 SC','normalize': 'min-max', 'log10': True,
         'weight': 1}, 
        {'#': 2, 'parameter': 'q_resv','normalize': 'min-max', 'log10': True,
         'weight': 1}, 
        {'#': 3, 'parameter': 'depth', 'normalize': 'min-max', 'log10': False,
         'weight': -1},          
        # {'#': 4, 'parameter': None, 'normalize': 'min-max', 'log10': False,
        #  'weight': 1},  
        # {'#': 5, 'parameter': None, 'normalize': 'min-max', 'log10': False,
        #  'weight': 1},   
        ]

    ts_table = DataTable(
        id='ts_table', columns=clmns, data=wdata,
        dropdown=dropdowns, editable=True,
        style_table={'width': '45vw'},
        style_cell={'fontSize': 14, 
                    'textAlign': 'center',
                    'whiteSpace': 'normal'} 
    ) 

    out = (
        mtable, para_table, ts_table,  #
        1, 1, 1, 1, # initializes the plots
        _num_clmns,  # options for map's size dropdown
        _all_clmns,  # options for map's color dropdown
        _num_clmns, # options for scatter's X dropdown
        _num_clmns, # options for scatter's Y dropdown
        _num_clmns, # options for scatter's size dropdown
        _all_clmns, # options for scatter's color dropdown
        _all_clmns, # configure tooltips in maps and scatter
        _num_clmns, # options for para's color dropdown          
        markdown_help,  # help text
        UNITS_INFO, # dictionary: {'column': {'info': '...', unit: '...'}}
        SHAPES # shapes of BAA inc. CO2 expl. and storage licenses
    )
    return out
        
# %% select/deselect
@app.callback(
    Output('mtable', 'selected_rows'),
    Input('b_deselect', 'n_clicks'),
    Input('b_chart_selection', "n_clicks"),    
    Input('b_select', 'n_clicks'),    
    State('mtable', 'data'), 
    State('mtable', 'selected_rows'),
    State('mtable', 'derived_virtual_data'),
    State("all_tabs", 'active_tab'),
    State('map_fig', 'selectedData'),
    State('sc_fig', 'selectedData'),    
    State('para_selected', 'children'),     
    prevent_initial_call=True
)
def select_deselect(m, b, n, 
                    records, selected_rows, filtered_rows,
                    active_tab, sel_map, sel_sc, sel_para):

    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'b_deselect' in changed_id:
        selected_rows = []
    elif 'b_select' in changed_id:
        selected_rows += [row['#'] for row in filtered_rows]
        selected_rows.sort()
        selected_rows = list(set(selected_rows))  # adds to previous selection
    elif 'b_chart_selection' in changed_id:
        sel = []
        if active_tab == 'tab_map':
            if sel_map is None: return selected_rows
            for p in sel_map['points']:
                sel.append(p['customdata'][1])
        elif active_tab == 'tab_sc': 
            if sel_sc is None: return selected_rows
            for p in sel_sc['points']:
                sel.append(p['customdata'][0])
        elif active_tab == 'tab_para':
            sel=eval(sel_para[10:])
        else:
            pass  
        # geting # from field names
        df = pd.DataFrame(data=records)[['field','#']]
        df = df.set_index('field')
        selected_rows += df.loc[sel,'#'].to_list()
    else:
        pass
    selected_rows.sort()
    return selected_rows


@app.callback(
    Output('map_fig', 'figure'),
    Input('update_map', 'n_clicks'),
    Input('map_dd_color', 'value'),
    Input('map_dd_size', 'value'),
    Input('select_map_colorscale', 'value'),  
    Input('switch_reverse_map_cs', 'value'),  
    Input('select_map_style', 'value'),      
    Input('select_dclrs', 'value'),  
    Input('dd_configure_tooltips', 'value'),        
    Input('checkbox_CO2_BAA', 'value'),      
    State('map_fig', 'figure'),    
    State('mtable', 'selected_rows'),
    State('mtable', 'data'),
    State('theme_store', 'data'),
    State('units_info_store','data'),
    State('shape_store', 'data'),

    # prevent_initial_call=True
)
def update_map(n, color, size, 
               colorscale, reverse_colorscale, 
               map_style, dclrs, add_to_tooltips, show_co2_baa,
               fig0, sel_rows, records, theme, info_units, SHAPES):

    fig = go.Figure()
    if sel_rows is None or sel_rows == []: return fig

    if add_to_tooltips is None: add_to_tooltips=[]

    df = pd.DataFrame(data=records)
    df = df.loc[sel_rows, :]
    if reverse_colorscale: colorscale += "_r"

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
        dclrs = None
    else:
        df = df.sort_values(by=color, ascending=True, kind='stable')
        if dclrs=='Rainbow':
            dclrs = generate_rainbow_colors(len(df[color].unique()))
        else:
            dclrs = eval(f'px.colors.qualitative.{dclrs}')
        
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
    labels ={}
    for i in [size,color,*add_to_tooltips]:
        unit = info_units[i]['unit']
        labels[i] = i if unit in [''] else f"{i} ({unit})"

    fig=px.scatter_mapbox(
        df, lat='lat', lon='lon',size=_size, color=color,
        # hover_data=['field','lat','lon',size,color,'size', *add_to_tooltips],
        hover_data=['field',size,color,*add_to_tooltips],
        size_max=size_max, 
        custom_data=['FactPageUrl'],
        color_continuous_scale=colorscale,
        # color_discrete_sequence=px.colors.qualitative.G10,
        color_discrete_sequence=dclrs, labels=labels
        )

    if (fig0 is not None) and (fig0['layout'].get('mapbox') is not None):
        # print(fig0['layout']['mapbox'])
        zoom = fig0['layout']['mapbox']['zoom']
        ref_lat, ref_lon = \
            fig0['layout']['mapbox']['center']['lat'], \
            fig0['layout']['mapbox']['center']['lon']
    else:
        # these limits are configured to show all fields in the North sea
        zoom = 5.5  
        ref_lat, ref_lon = df.loc[df['sea']=='NORTH', ['lat','lon']].mean().values
        # these limits are configures to show all NCS fields
        # ref_lat, ref_lon, zoom = 65.7, 8.280232, 3.7
    
    center = {'lat': ref_lat, 'lon': ref_lon}
    # center = go.layout.mapbox.Center(lat=ref_lat, lon=ref_lon)


    if show_co2_baa:
        for shape in SHAPES:
            name = shape['baaName']
            # skipping petroleum BAAs ...
            if not name[:3] in ['EL0', 'EXL']: continue

            coords = shape['coordinates']
            lons, lats = zip(*coords)
            lons += (lons[0],)
            lats += (lats[0],)

            fig.add_trace(go.Scattermapbox(
                lon=lons, lat=lats, mode='lines', name=name,
                hovertext=(
                    f"<b>{shape['baaName']}</b><br>"
                    f"{shape['baaKind']}<br>"
                    f"Operator: {shape.get('cmpLongName', 'N/A')}<br>"
                ),            
                hoverinfo='text',
                # customdata=[shape["baaFactPageUrl"]],
                showlegend=False,
                line=dict(width=2, color='rgba(125, 125, 125, 0.15)'),
                fill='toself',
                fillcolor='rgba(125, 125, 125, 0.15)'  # semi-transparent fill
            ))

    fig.update_layout(
        template=theme,
        mapbox={
            'style': map_style,
            'center': center,
            'zoom': zoom,
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
    Input('select_dclrs', 'value'),  
    Input('sc_x_log10', 'value'),  
    Input('sc_y_log10', 'value'),  
    #
    State('mtable', 'selected_rows'),
    State('mtable', 'data'),
    State('theme_store', 'data'),
    State('units_info_store','data'),
    Input('dd_configure_tooltips', 'value'),
    # prevent_initial_call=True
)
def update_sc(n, x, y, color, size, colorscale, reverse_colorscale, dclrs,
              log10_x, log10_y,
              sel_rows, records, theme, info_units, add_to_tooltips):
    
    df = pd.DataFrame(data=records)
    if reverse_colorscale: colorscale += "_r"

    if sel_rows is None or sel_rows == []:
        return go.Figure()

    df = df.loc[sel_rows, :]  
    not_none_clms=[i for i in [x, y, size] if i is not None]
    df = df.dropna(subset=not_none_clms)    
    labels ={}
    for i in [*not_none_clms,*add_to_tooltips]:
        unit = info_units[i]['unit']
        labels[i] = i if unit in [''] else f"{i} ({unit})"

    if not color is None:
        labels[color]=f"{color}<br>({info_units[color]['unit']})"
        if df[color].dtype in ['float', 'int64', 'int32', 'int16', 'int8']:
            dclrs = None
        else:
            df = df.sort_values(by=color, ascending=True, kind='stable')
            if dclrs=='Rainbow':
                dclrs = generate_rainbow_colors(len(df[color].unique()))
            else:
                dclrs = eval(f'px.colors.qualitative.{dclrs}')
    else:
        dclrs = None
    
    size_max = 10
    _size = None
    hover_data=['field', x, y, color, size]
    if size is not None:  
        hover_data += ['s']        
        df['s'] = df[size]
        # # df.s = df.s**0.5
        s_max, s_min = df.s.max(), df.s.min()
        df.s = 1+99*(df.s - s_min)/(s_max - s_min)
        df.s = df.s.round(3)
        size_max = 50
        _size = df.s        

    if add_to_tooltips is None: add_to_tooltips=[]
    hover_data += add_to_tooltips

    fig=px.scatter(
        df, x=x, y=y, color=color, size=_size, size_max=size_max,
        template=theme, color_continuous_scale=colorscale,
        hover_data = hover_data,
        log_x=log10_x, log_y=log10_y,
        color_discrete_sequence=dclrs,
        labels = labels
        )    

    fig.update_layout(
        font_size=14,
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
    if clickData is not None and open==True and \
        clickData['points'][0].get('customdata',False):
        url=clickData['points'][0]['customdata'][0]
        url = url[0] if isinstance(url,list) else url
        webbrowser.open(url)
    return ""


@app.callback(
    Output('para_fig', 'figure'),
    Output('para_store_df', 'data'),   
    Input('para_update', 'n_clicks'),
    Input('para_dd_color','value'),
    Input('select_para_colorscale','value'),
    Input('switch_reverse_para_cs','value'),
    State('mtable', 'selected_rows'),    
    State('mtable', 'data'),        
    Input('para_table', 'data'),  
    State('theme_store', 'data'),
    prevent_initial_call=True
)
def para_update(n, color, colorscale, reverse_colorscale, 
                sel_rows, mrecords, precords,  theme):

    if (sel_rows is None) or (sel_rows == []): 
        return go.Figure(), []
    
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

        norm_method = pdf.loc[i,'normalize']
        if pdf.loc[i,'normalize'] is not None:
            x=normalize_series(x, norm_method)
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
    fig = fig.update_layout(font_size=14)

    return fig, df.to_dict('records')

@app.callback(
    Output('para_table', 'data'),    
    Input('para_plus', 'n_clicks'),
    Input('para_minus', 'n_clicks'),    
    State('para_table', 'data'), 
    prevent_initial_call=True  
)
def para_plus_minus(p,m, records):
    '''adds/removes rows in para-table'''
    nrows = len(records)
    # print(nrows, p, m, p-m)
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'para_plus' in changed_id:
        records.append({
            '#': nrows+1, 'parameter': None, 'normalize': None, 'log10': False,
            'reverse': False
            })
    if ('para_minus' in changed_id) and (nrows-1>=3): 
        records=records[:-1]

    return records

@app.callback(
    Output("para_selected", "children"),  
    Output("para_store_ranges", "data"),  
    #
    Input('para_update', 'n_clicks'),
    Input("para_fig", "restyleData"),
    Input("para_fig", "figure"),    
    State("para_store_ranges", "data"),
    State("para_store_df", "data"),
    prevent_initial_call=True
)
def para_display_selected_traces(n, restyleData, fig,  ranges, records):
    '''shows field names for the selected traces'''

    if records==[]: return None, {}
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
    Input('ts_update', 'n_clicks'),  
    Input('ts_switch', 'value'),      
    State('mtable', 'data'),
    State('mtable', 'selected_rows'),    
    Input('ts_table', 'data'),    
    State('theme_store', 'data'),
    prevent_initial_call=True
)
def ts_update(n, use_only_selected, 
              records, sel_rows, wrecords, theme):
    '''calculates total score column, updates the chart'''
    
    df = pd.DataFrame(data=records) # main df
    if not use_only_selected: 
        sel_rows = df.index.to_list()
    else:
        if sel_rows is None or sel_rows == []:
            raise PreventUpdate  

    wdf = pd.DataFrame(data=wrecords)  # weights etc.
    wdf = wdf.dropna(subset='parameter')
    cdf = df.loc[sel_rows,['field']].copy() # to store contributions
    weight_sum=wdf['weight'].abs().sum()
    params = wdf['parameter'].to_list()

    df['total score'] = df['total score'].astype(float)
    df.loc[sel_rows,'total score'] = 0.0
    for i in wdf.index:
        p = wdf.loc[i, 'parameter']
        if p is None: continue
        
        x = df.loc[sel_rows,p].copy()
        weight = wdf.loc[i, 'weight']
        # the series is mirrorred  if it should be minimized ... 
        if weight < 0: 
            x = x.max() - x + x.min()  
            weight = np.abs(weight)

        # utitlity function
        if wdf.loc[i, 'log10']:  x=np.log10(x)

        # to normalize
        x=normalize_series(x, method=wdf.loc[i,'normalize'])
        
        # to calculate contribution of the parameter to the total score
        cdf[p] = 100*weight*x/weight_sum
        cdf[p] = cdf[p].round(1)
        # adding the parameter's contribution to the total score 
        df.loc[sel_rows,'total score'] += cdf[p]

    # df['total score'] = df['total score'].apply(round_to_sign_digits)
    df.loc[sel_rows,'total score'] = df.loc[sel_rows,'total score'].round(1)

    # creating the histogram with the results
    cdf['total score'] = df['total score'].round(1)
    cdf = cdf.dropna()
    cdf = cdf.sort_values(by='total score', ascending=False)
    cdf = cdf[:min(15,cdf.shape[0])]

    fig = px.bar(cdf, x=params, y="field", orientation='h',log_x=False,
                 hover_data=['total score'], template=theme
                 )
    fig.update_layout(
        barmode='stack',
        font_size=14,
        yaxis=dict(autorange='reversed'), 
        xaxis=dict(title='total score and its components', side='top')
        )
    return df.to_dict('records'), fig


@app.callback(
    Output('ts_table', 'data'),    
    Input('ts_plus', 'n_clicks'),
    Input('ts_minus', 'n_clicks'),    
    State('ts_table', 'data'),   
    prevent_initial_call=True
)
def ts_plus_minus(p,m, records):
    '''adds/removes rows in ts-table'''
    nrows = len(records)
    # print(nrows, p, m, p-m)
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'ts_plus' in changed_id:
        records.append({
            '#': nrows+1, 'parameter': None, 'normalize': 'min-max', 
            'log10': False, 'weight': -1
            })
    if ('ts_minus' in changed_id) and (nrows-1>=3): 
        records=records[:-1]

    return records

@app.callback(
    Output("download", 'data'),    
    Input("b_save_fig", 'n_clicks'),    
    State("all_tabs", 'active_tab'),
    State("map_fig", 'figure'),
    State("sc_fig", 'figure'), 
    State("para_fig", 'figure'),  
    State("ts_fig", 'figure'),
    State("select_save_fig_format", 'value'),
    State("input_save_fig", 'value'),    
    State("select_save_fig_scale", 'value'),        
    prevent_initial_call=True
)
def save_current_figure(n, active_tab, map_fig, sc_fig, para_fig, ts_fig, 
                        format, _filename, scale):
    
    def clean_marker_colors(fig_dict, default_color="gray"):
        for trace in fig_dict.get("data", []):
            marker = trace.get("marker", {})
            color = marker.get("color", None)
            if isinstance(color, list):
                # Replace None values
                marker["color"] =\
                      [default_color if c is None else c for c in color]
        return fig_dict    
    
    width = 1100 
    height = 9.00/9.50*width
    if active_tab == 'tab_map':
        # replace None colors with grey
        fig = clean_marker_colors(map_fig)
        filename = 'subcset_map'
    elif active_tab == 'tab_sc':
        fig = clean_marker_colors(sc_fig)
        filename = 'subcset_scatter'
    elif active_tab == 'tab_para':
        filename = 'subcset_para'
        fig = para_fig
        width, height = width, 5/9.5*width
    elif active_tab == 'tab_ts':
        filename = 'subcset_total_score'
        fig = ts_fig
        width, height = width, 6/9.50*width 
    else:
        pass

    if _filename is None:
        now = datetime.now().strftime('%Y-%m-%d-%H%M%S')
        filename = f'{now}_{filename}{format}'
    else:
        filename = f'{_filename}{format}'
        
    # changed_id = [p['prop_id'] for p in callback_context.triggered][0]

    # this one works fine
    # return {'content': to_html(fig), 'filename': f'{now}_{filename}.html'}
    # this one works fine
    # buffer = io.BytesIO()
    # write_image(fig, buffer, format=format[1:], 
    #             scale=1, width=width, height=height
    #             )
    # buffer.seek(0)    
    # return dcc.send_bytes(buffer.read(), filename=filename)
    # this one works fine
    if format == '.html':
        return {'content': to_html(fig), 'filename': filename}
    else: 
        return dcc.send_bytes(
            to_image(fig, format=format[1:], scale=scale, 
                     width=width, height=height), 
            filename=filename)        


if __name__ == '__main__':
    if DEBUG:
        app.run(debug=True)  # should be False for deployment
    else:
        app.run(debug=False)  # should be False for deployment
