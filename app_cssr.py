import pandas as pd
import numpy as np
import npd_wraper as npd
# plotly & dash
import plotly.express as px
from plotly.io import write_html, to_json, from_json
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
# from flask_caching import Cache
from dash.exceptions import PreventUpdate
from dash.dash_table.Format import Format, Scheme
from dash.dash_table import DataTable
from dash import html
from dash_bootstrap_templates import load_figure_template, ThemeChangerAIO, template_from_url
import dash_bootstrap_components as dbc
from dash import callback_context
from dash import dcc
from dash import Dash
import time
import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
import cssr

# %% constants
themes_templates =\
    ["bootstrap", "cerulean", "cosmo", "cyborg", "darkly", "flatly",
     "journal", "litera", "lumen", "lux", "materia", "minty", "morph",
     "pulse", "quartz", "sandstone", "simplex", "sketchy", "slate",
     "solar", "spacelab", "superhero", "united", "vapor", "yeti", "zephyr"]

# THEME0 = np.random.choice(themes_templates)
THEME0 = "yeti"
# THEME0 = "cyborg"
THEME0 = THEME0.upper()
# %% Button to change the themes
c_theme = ThemeChangerAIO(
    aio_id="theme",
    radio_props={"value": eval('dbc.themes.'+THEME0)},
    button_props={
        # "color": "danger",
        "children": html.I(className="bi bi-palette"),
        'outline': True,
        'size': 'md',
        'style': {'width': '100%'}
    },
    offcanvas_props={"placement": "start",
                     "scrollable": True,
                     #  'style': {'width': '50vw'}
                     }
)

# %% app layout ----------------------------------------------------------------
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

# cache = Cache(app.server, config={
#     'CACHE_TYPE': 'filesystem',
#     'CACHE_DIR': 'cache-directory'
# })

# layout = [
#     dbc.Tabs([
#         dbc.Tab([dbc.Container(table_rv)], id='tab_table', label='overview'),
#         dbc.Tab([], id='tab_map', label='map'),
#     ])
#     # html.H1('CSSR screening'),
#     # html.P(id='text_output', children='placeholder'),
# ]

c_inp_fldr = dbc.Input(
    id='inp_fldr', type='text',
    value=r'./data/_main.csv'
)

c_mtable = DataTable(id='mtable', data=[], selected_rows=[])
c_mtable = html.Div(c_mtable, id='mtable_div')

# %% table with weights

c_wtable = DataTable(id='wtable', columns=[], data=[], editable=True)
c_wtable = html.Div(c_wtable, id='wtable_div')


# %%
map = dcc.Graph(
    id='map',
    style={
        'height': '90vh',
        'padding': '0.25vw'
    },
    config={'displayModeBar': True}
)

c_b_save = dbc.Button(
    'save selected', id='save_button', n_clicks=0,
    className="me-1", size='md',
)

c_b_update = dbc.Button(
    'update', id='update', n_clicks=0,
    color='danger',
    className="me-1", size='md',
    # style={'width': '30%'}
)

c_toolbar = dbc.ButtonGroup([
    dbc.Button(html.I(className="bi bi-check2-square"),
               #    outline=True,
               size='md', id='select'),
    dbc.Button(html.I(className="bi bi-x-square"),
               #    outline=True,
               size='md', id='deselect'),
    dbc.Button(html.I(className="bi bi-box-arrow-up-right"),
               #    outline=True,
               id='reopen', size='md',
               ),
])

# c_toolbar = dbc.Card(c_toolbar)

c_toolbar = dbc.Stack([
    c_toolbar,
    c_b_update,
    dbc.InputGroup([
        dbc.InputGroupText('size'),
        dbc.Select(id='dd_size'),
    ], style={'width': '30%'}),
    dbc.InputGroup([
        dbc.InputGroupText('color'),
        dbc.Select(id='dd_color'),
    ],
        style={'width': '30%'}
    ),
    dbc.Checkbox(id='chbx_invert', label="invert", value=False),
    dbc.Tooltip('add all filtered rows to selection', target='select'),
    dbc.Tooltip('deselect all', target='deselect'),
    dbc.Tooltip('reopen chart in new tab', target='reopen'),
],
    direction="horizontal"
)

c_tabs = dbc.Tabs([
    dbc.Tab(map, label='MAP', 
            active_tab_style={"fontWeight": "bold"},
            style={"flex": "1", "height": "100%"}
            ),
    dbc.Tab(label='SCATTER', 
            active_tab_style={"fontWeight": "bold"})
], 
# style={"flex": "1", "height": "100%"}
)

app.layout = html.Div([
    dbc.Row([
        dbc.Col([c_inp_fldr, 
                 c_wtable,
                 c_toolbar, 
                 c_mtable], width=6),
        dbc.Col(c_tabs, width=6)
    ],
        className="g-0",
        # style={"flex": "1", "height": "100%"}
    ),
    dcc.Store(id='store'),
],
    style={'display': 'grid', 
           # to account for scrollbars
           'width': 'calc(100vw - (100vw - 100%))',
           'height': 'calc(100vh - (100vh - 100%))',
           },
    className="dbc"
)


@app.callback(
    Output('mtable_div', 'children'),
    Output('wtable_div', 'children'),
    Output('update', 'n_clicks'),
    Output('dd_size', 'options'),
    Output('dd_size', 'value'),
    Output('dd_color', 'options'),
    Output('dd_color', 'value'),
    Input('inp_fldr', 'value'),  # ! temp
    # prevent_initial_call=True
)
def initial_setup(path2csv):

    with open(r'./data/_main_columns.json', 'r') as f:
        FANCY_CLMNS = json.load(f)

    df = pd.read_csv(path2csv)
    
    df['#'] = range(df.shape[0])
    # df = df.drop(columns={'reservoir','recovery','in-place liq./OE ratio'})
    if 'total score' in df.columns:
        df['total score'] = None

    all_clmns = {i: ["", i] for i in df.columns}
    all_clmns = {**all_clmns, **FANCY_CLMNS}

    # # Determine which DataFrame columns are not in the priority list
    # priority_columns = [i for i in FANCY_CLMNS if i in df.columns]     
    # other_columns = [col for col in df.columns if col not in priority_columns]
    # # Combine the lists, with priority columns first, then others
    # new_column_order = priority_columns + other_columns
    # df = df[new_column_order]

    clmn_names = list(all_clmns.keys())

    clmns = []
    for k, v in all_clmns.items():
        clmns.append({"name": v, "id": k, 'hideable': True,
                      #   'selectable': True
                      })

    for col in ['in-place liq./OE ratio',
                'rec. cond.', 'rec. NGL',
                'rem. NGL', 'rem. cond.',
                'RF oil', 'RF gas', 'gas FVF', 'oil FVF',
                'maturity oil', 'maturity gas', 'inj. score',
                'CO2 SC indicator', 'H2 SC indicator', 
                'produced oil PV', 'produced gas PV',
                'produced HCPV', 'CO2 SC indicator', 'H2 SC indicator'
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
        # fixed_columns={'headers': True, 'data': 1},
        page_action="native", page_current=0,
        hidden_columns=[
            '#','FactPageUrl', 'reservoir', 
            'res. quality','faulted',
            'recovery', 'fldID', 'GOR'
            'size', 'lat', 'lon',
            'depth min',  'depth mean', 'depth median', 'depth max',
            'year',
            'field',
            'net oil yearly pr.',
            'net gas yearly pr.'
            'net NGL yearly pr.',
            'net condensate yearly pr.',
            'net OE yearly pr.',
            'water yearly pr.',
            # 'inj. score'
            ],

        style_table={
            # 'height': '60vh',
            'height': '55vh',
            # 'height': 'auto',
            # 'width': 'auto',
            'width': '50vw', 
            'overflowX': 'auto', 
            'overflowY': 'auto'
        },
        # style_data_conditional=[
        #     # Highlighting the 'Name' column
        #     {
        #         'if': {'column_id': 'Name'},
        #         'backgroundColor': '#FFD2D2',
        #         'color': 'black'
        #     },
        #     # Highlighting the 'Age' column
        #     {
        #         'if': {'column_id': 'Age'},
        #         'backgroundColor': '#D2F0FF',
        #         'color': 'black'
        #     }
        # ],        
        style_cell={'fontSize': 14, 'whiteSpace': 'normal'}
    )


    clmns = [{'name': i, 'id': i} for i in ['parameter', 'weight']]
    clmns[0]['presentation'] = 'dropdown'
    clmns[1]['type'] = 'numeric'

    sel_columns = df.columns
    dropdowns = {}
    dropdowns['parameter']={}
    dropdowns['parameter']['options']=\
        [{'label': i, 'value': i} for i in sel_columns]

    wtable = DataTable(
        id='wtable', columns=clmns, 
        data=[{'parameter': None, 'weight': None}]*5,
        dropdown=dropdowns, editable=True
    )


    return mtable, wtable, 1, clmn_names, 'CO2 SC indicator', clmn_names, 'field'


# %% select/deselect
@app.callback(
    Output('mtable', 'selected_rows'),
    Input('deselect', 'n_clicks'),
    Input('select', 'n_clicks'),    
    State('mtable', 'selected_rows'),
    State('mtable', 'derived_virtual_data'),
    prevent_initial_call=True
)
def select_deselect(m, b, selected_rows, filtered_rows):

    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'deselect' in changed_id:
        selected_rows = []
    elif 'select' in changed_id:
        selected_rows += [row['#'] for row in filtered_rows]
        selected_rows.sort()
        selected_rows = list(set(selected_rows))  # adds to previous selection
    else:
        # raise PreventUpdate
        pass
    selected_rows.sort()
    return selected_rows


@app.callback(
    Output("reopen", "n_clicks"),
    Input("reopen", "n_clicks"),
    State('map', 'figure'),
    prevent_initial_call=True
)
def reopen_current_chart(n, fig):

    if fig is not None:
        fig = go.Figure(fig)
        fig.show(renderer='browser')

    return n


@app.callback(
    Output('map', 'figure'),
    Output('mtable', 'data'),
    Input('update', 'n_clicks'),
    Input('dd_color', 'value'),
    Input('dd_size', 'value'),
    Input('chbx_invert', 'value'),    
    State('mtable', 'selected_rows'),
    State('mtable', 'data'),
    State('wtable', 'data'),
    # prevent_initial_call=True
)
def update(n, color, size, invert, sel_rows, records, records2):

    fig = go.Figure()
    df = pd.DataFrame(data=records)
    # df.set_index('index',inplace=True)

    df2 = pd.DataFrame(data=records2)

    # print(df2)

    df['total score'] = 0
    for i in df2.index:
        column = df2.loc[i, 'parameter']
        if column is None: continue
        weight = df2.loc[i, 'weight']
        df['total score'] += weight*df[column]
    
    df0 = df.copy()

    if sel_rows is None or sel_rows == []:
        print('PreventUpdate!')
        # raise PreventUpdate
        return fig

    df = df.loc[sel_rows, :]
    # df.reset_index(inplace=True)

    df['color'] = 'grey'  # preallocating
    if df[color].dtype in ['float', 'int64', 'int32', 'int16', 'int8']:
        df = df.sort_values(by=color, ascending=False, kind='stable')
        df['color']=cssr.generate_rainbow_colors(df[color]*(-1 if invert else 1))
    else:
        df = df.sort_values(by=color, ascending=True, kind='stable')
        # all other datatypes are treated as categoricals
        foo = df[color].unique()
        clrs = cssr.generate_rainbow_colors(len(foo))
        if invert: clrs = clrs[::-1]
        for i, clr in zip(foo, clrs):
            ind = df[color] == i
            df.loc[ind, 'color'] = clr

    df['size'] = df[size]**0.5
    df['size'] = 5 + 95*(df['size'] - df['size'].min())/(df['size'].max())

    for i in df.index:
        row = df.loc[i, :]
        url=row['FactPageUrl']
        fig.add_trace(
            go.Scattermapbox(
                lat=[row['lat']],
                lon=[row['lon']],
                mode='markers',
                name=row['field'],
                # text=f"{row[color]}</b>{size}: {row[size]}"+\
                #     f"</b>{color}:{row[color]}",
                legendgroup='fields',
                legendgrouptitle_text='fields',
                # hoverinfo=['name',color,size],
                # hoverinfo='text',
                hovertemplate=  # f"%{name}<br>"+
                f"{row['field']}<br>" +
                f"{size}: {row[size]}<br>" +
                f"{color}: {row[color]}<br>" +
                # rf"URL:  {url}"+
                "<extra></extra>",
                marker={'size': row['size'], 'opacity': .5,
                        'symbol': 'circle',
                        # 'line': {'color': 'black'},
                        'color': row['color']}
            )
        )

    styles = ['open-street-map', 'carto-positron', 'carto-darkmatter']
    ref_lat, ref_lon = df.loc[:, ['lat', 'lon']].mean().values

    fig.update_layout(
        mapbox={
            # 'style': 'open-street-map',
            # 'style': 'carto-positron', 
            'style': 'carto-darkmatter',
            'center': go.layout.mapbox.Center(lat=ref_lat, lon=ref_lon),
            # 'fitbounds': 'locations',
            'zoom': 5.5,
            # 'maxzoom': 10, 'minzoom':
        },
        legend=dict(groupclick="toggleitem"),
        modebar_orientation='v',
        margin={"r": 10, "t": 0, "l": 0, "b": 0},
        modebar_add=['toggleHover', 'drawline', 'drawopenpath',
                     'drawclosedpath', 'drawcircle', 'drawrect',
                     'eraseshape', 'toggleSpikelines'],
    )
    fig.update_geos(fitbounds="locations")

    return fig, df0.to_dict('records')


if __name__ == '__main__':
    app.title = 'CSSR screening'
    app.run_server(debug=True)
    # app.run_server(debug=False)  # should be False for deployment
# %%
