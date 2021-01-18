# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import plotly.figure_factory as ff
import numpy as np
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

external_stylesheets_boostrap=[dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

layout_settings = {
                  'transition_duration': 1000,
                  'transition_easing': "cubic-in-out",
                  'font_family':"Courier New",
                  'font_size':14,
                  'height': 720
                }

AirBnB_data = "./data/cleaned-AirBnB.csv"

df = pd.read_csv(AirBnB_data, sep=',', decimal='.', header=None,
  names =['id', 'description', 'neighborhood_overview', 'host_id', 'host_since',
       'host_location', 'host_about', 'host_response_time',
       'host_response_rate', 'host_acceptance_rate', 'host_is_superhost',
       'host_neighbourhood', 'host_listings_count',
       'host_total_listings_count', 'host_verifications',
       'host_has_profile_pic', 'host_identity_verified', 'neighbourhood',
       'neighbourhood_cleansed', 'Borough', 'latitude',
       'longitude', 'property_type', 'room_type', 'accommodates',
       'bathrooms_text', 'bedrooms', 'beds', 'amenities', 'price',
       'minimum_nights', 'maximum_nights', 'minimum_minimum_nights',
       'maximum_minimum_nights', 'minimum_maximum_nights',
       'maximum_maximum_nights', 'minimum_nights_avg_ntm',
       'maximum_nights_avg_ntm', 'has_availability', 'availability_30',
       'availability_60', 'availability_90', 'availability_365',
       'number_of_reviews', 'number_of_reviews_ltm', 'number_of_reviews_l30d',
       'first_review', 'last_review', 'review_scores_rating',
       'review_scores_accuracy', 'review_scores_cleanliness',
       'review_scores_checkin', 'review_scores_communication',
       'review_scores_location', 'review_scores_value', 'instant_bookable',
       'calculated_host_listings_count',
       'calculated_host_listings_count_entire_homes',
       'calculated_host_listings_count_private_rooms',
       'calculated_host_listings_count_shared_rooms', 'reviews_per_month',
       'encodePrivacy', 'StringPropType', 'encoded prop_type',
       'room_type_rank', 'bathrooms_Count', 'bathrooms_share',
       'amenities_clean', 'availability_30_nml', 'availability_60_nml',
       'availability_90_nml', 'availability_365_nml',
       'review_scores_rating_nml', 'host_since_days', 'host_year_exp',
       'host_year', 'number_of_amenities', 'string_amenities'])


# Create price array of each neighbourhood_group_cleansed

group_labels=["Manhattan", "Brooklyn", "Queens", "Staten Island", "Bronx"]

# NY map figure

# get year list
years = df["host_year"].unique()
years = list(sorted(years.astype(str)))
years_dict = {i:i for i in years}

px.set_mapbox_access_token("pk.eyJ1IjoiaHVuZ3RoZXpvcmJhIiwiYSI6ImNrYnE4d3o4OTE3MjEydnB0cG44b3JiZzEifQ.M5kWTwGGuqA7WGBAdsIurQ")
@app.callback(
    [dash.dependencies.Output('map-graph', 'figure'),
     dash.dependencies.Output('airbnb-num','children')],
    [dash.dependencies.Input('year-slider', 'value')])
def update_map(year_value):

    df_selected = df[df['host_year'] <= year_value]

    map_figure = px.scatter_mapbox(df_selected, lat=df_selected['latitude'], lon=df_selected['longitude'],
                                     color="Borough",
                                     labels="Borough",
                                    zoom=9, mapbox_style="mapbox://styles/hungthezorba/ckjwq978c106e17mynq5zyx9q",
                                    center=dict(lat=40.730610, lon=-73.935242),
                                    height=720,

                                   hover_data=["price"],
                                    color_continuous_midpoint=1,
                                    )
    map_figure.update_layout(font_family="Courier New",
                             font_size=14,
                             title={
                                    'text': "Map of AirBnB in New York City",
                                    'x':0.45,
                                    'xanchor': 'center',
                                    'yanchor': 'top'
                                    },
                             transition_duration=1
                             )
    data = [html.Div(children=[
        html.P(children=[
            "Number of ",
            html.Span(className="airbnb-span", children="AirBnB "),
            "in New York City in: ", str(year_value), ": ", str(len(df_selected))]
        )])]
    # data = "Number of AirBnB in New York City in " + str(year_value) + ": " + str(len(df_selected))
    return map_figure, data

# 1. First plot: box plot callback
@app.callback(
    dash.dependencies.Output('hist-graph-1', 'figure'),
    [dash.dependencies.Input('x-box-dropdown', 'value')])
def update_output(x_value):
    # boxplot figure
    fig = px.box(df, x=x_value,
                 color="Borough" ,height=622,
                 title="Distributions in New York's Borough",
                 labels={
                     x_value: " ".join(x_value.split("_")).capitalize(),
                 }
                )
    fig.update_layout(layout_settings)

    return fig

# 2. Second plot: scatter plot callback
@app.callback(
    dash.dependencies.Output('scatter-graph-1', 'figure'),
    [dash.dependencies.Input('x-scatter-dropdown', 'value'),
     dash.dependencies.Input('y-scatter-dropdown', 'value'),
     dash.dependencies.Input('color-scatter-dropdown', 'value')])
def update_scatter(x_value, y_value, color_value):
    # scatter figure
    if (color_value == "host_is_superhost"):
    	 df['host_is_superhost'] = df['host_is_superhost'].replace({1:'T',0:'F'})

    fig = px.scatter(
        df, x=x_value, y=y_value,
        color=color_value,
        hover_data=['price'],
        title="Scatter plot visualization AirBnB in New York city",
        labels={
            x_value: " ".join(x_value.split("_")).capitalize(),
            y_value: " ".join(y_value.split("_")).capitalize(),
        }
    )

    fig.update_layout(layout_settings)

    return fig

app.layout = html.Div(id="homepage", children=[
    html.Section(id="header", children=[
    html.Div(className="header-box", children=[
        html.Div(children=[
            html.H1(id="project-title", children="AirBnB Case Study")

            ]),

        html.Div(children=[
            html.H3(id="course-title", children="COSC2789 - Practical Data Science")

            ])

        ])
    ]),
    html.Section(id="main", children=[

    	# AirBnB in New York city's map
        html.Article(id="airbnb-map", children=[

            html.Div(className="heading-name", children=[

                    html.H4(className="section-title",children="I. Map of AirBnB in New York City"),

                ]),

                html.Div(className="map-container", children=[
						html.P(id="airbnb-num"),
                        dcc.Slider(
                            id='year-slider',
                            min=int(years[0]),
                            max=int(years[-1]),
                            value=int(years[-1]),
                            marks=years_dict,
                            included=False
                        ),
                        dcc.Graph(
                            id="map-graph",

                            )
                    ])

            ]),

        # First plot visualization
        html.Article(id="price-distribution", children=[

                dbc.Row(className="plot-container", children=[

                    dbc.Col(width=4,lg=4, xs=12, className="left-container", children=[

                            html.Div(className="plot-name", children=[

                                    html.H4(className="section-title", children="II. Distributions in New York's Borough")
                                ]),
                            html.Div(className="plot-description", children=[

                                    html.P(children="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aliquam pellentesque nulla sed leo blandit egestas. Quisque tempus, turpis non finibus pellentesque, tellus arcu consequat elit, tincidunt pellentesque libero eros porta dui. Fusce vitae dui quis justo lobortis tristique sit amet ac orci. Praesent tincidunt enim at facilisis rutrum.")
                                ]),
                            html.Div(className="plot-selectors", children=[
                                    html.Div(className="box dropdown-custom", children=[
                                        html.P(children="X-axis"),
                                        dcc.Dropdown(id='x-box-dropdown',
                                            options=[

                                                {'label': 'Price', 'value': 'price'},
                                                {'label': 'Number of reviews', 'value': 'number_of_reviews'},
                                                {'label': 'Accomodates', 'value': 'accommodates'},

                                            ],
                                                value='price'),
                                    ]),
                                ])

                        ]),
                        dbc.Col(width=8, lg=8, xs=12,className="right-container", children=[
                            html.Div(className="hist-plot", children=[
                                    dcc.Graph(
                                        id='hist-graph-1',

                                    )
                                ])
                            ])
                    ]),
            ]),

        # Second plot visualization
        html.Article(id="scatter-exploration", children=[
            dbc.Row(className="plot-container", children=[
                dbc.Col(width=4, lg=4, xs=12, children=[
                    html.Div(className="plot-name", children=[

                        html.H4(className="section-title", children="III. Scatter plot exploration in AirBnB")
                    ]),
                    html.Div(className="plot-description", children=[

                        html.P(
                            children="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aliquam pellentesque nulla sed leo blandit egestas. Quisque tempus, turpis non finibus pellentesque, tellus arcu consequat elit, tincidunt pellentesque libero eros porta dui. Fusce vitae dui quis justo lobortis tristique sit amet ac orci. Praesent tincidunt enim at facilisis rutrum.")
                    ]),
                    html.Div(className="plot-selectors", children=[
                        dbc.Row(children=[

                                dbc.Col(className="box dropdown-custom", children=[
                                    html.P(children="X-axis"),
                                    dcc.Dropdown(id='x-scatter-dropdown',
                                                 options=[

                                                     {'label': 'Price', 'value': 'price'},
                                                     {'label': 'Number of reviews', 'value': 'number_of_reviews'},
                                                     {'label': 'Accomodates', 'value': 'accommodates'},
                                                     {'label': 'Minimum nights', 'value': 'minimum_nights'},
                                                     {'label': 'Number of amenities', 'value': 'number_of_amenities'},
                                                     {'label': 'Number of bathrooms', 'value': 'bathrooms_Count'},


                                                 ],
                                                 value='price'),
                                ]),

                                dbc.Col(className="box dropdown-custom", children=[
                                    html.P(children="Y-axis"),
                                    dcc.Dropdown(id='y-scatter-dropdown',
                                                 options=[

                                                     {'label': 'Price', 'value': 'price'},
                                                     {'label': 'Number of reviews', 'value': 'number_of_reviews'},
                                                     {'label': 'Accomodates', 'value': 'accommodates'},
                                                     {'label': 'Minimum nights', 'value': 'minimum_nights'},
                                                     {'label': 'Number of amenities', 'value': 'number_of_amenities'},
                                                     {'label': 'Number of bathrooms', 'value': 'bathrooms_Count'},
                                                 ],
                                                 value='price'),
                                ]),
                        ]),
                        html.Div(className="box dropdown-custom", children=[
                            html.P(children="Color"),
                            dcc.Dropdown(id='color-scatter-dropdown',
                                         options=[
                                            {'label': 'Superhost', 'value': 'host_is_superhost'},
                                            {'label': 'Borough', 'value': 'Borough'},
                                            {'label': 'Room type', 'value': 'room_type'}
                                         ],
                                         value='host_is_superhost'),
                        ])
                    ])
                ]),

                dbc.Col(width=8, lg=8, xs=12, children=[
                    html.Div(className="scatter-plot", children=[
                        dcc.Graph(
                            id="scatter-graph-1",
                        )
                    ])
                ])
            ])
        ]),
        html.Article(id="classification-model", children=[
            dbc.Row(className="plot-container", children=[
                dbc.Col(width=4, lg=4, xs=12, children=[
                    html.Div(className="plot-name", children=[

                        html.H4(className="section-title", children="IV. Property classification in AirBnB")
                    ]),
                    html.Div(className="plot-description", children=[

                        html.P(
                            children="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aliquam pellentesque nulla sed leo blandit egestas. Quisque tempus, turpis non finibus pellentesque, tellus arcu consequat elit, tincidunt pellentesque libero eros porta dui. Fusce vitae dui quis justo lobortis tristique sit amet ac orci. Praesent tincidunt enim at facilisis rutrum.")
                    ]),
                ]),
            ])
        ])
    ]),
    html.Section(id="footer", children=[
        html.Div(className="footer-box", children=[

        ])
    ])
])

if __name__ == '__main__':
    app.title = "AirBnB Dashboard"
    app.run_server(debug=True)
