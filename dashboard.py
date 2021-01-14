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

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

AirBnB_data = "./cleaned-AirBnB.csv"

df = pd.read_csv(AirBnB_data, sep=',', decimal='.', header=None, 
  names =['description', 'neighborhood_overview', 'host_since', 'host_location',
   'host_about', 'host_response_time', 'host_response_rate', 'host_acceptance_rate',
    'host_is_superhost', 'host_neighbourhood', 'host_listings_count', 'host_total_listings_count',
     'host_verifications', 'host_has_profile_pic', 'host_identity_verified', 'neighbourhood', 'neighbourhood_cleansed', 
     'Borough', 'latitude', 'longitude', 'property_type', 'room_type', 'accommodates', 'bathrooms_text',
      'bedrooms', 'beds', 'amenities', 'price', 'minimum_nights', 'maximum_nights', 'minimum_minimum_nights', 'maximum_minimum_nights',
       'minimum_maximum_nights', 'maximum_maximum_nights', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'has_availability', 'availability_30',
        'availability_60', 'availability_90', 'availability_365', 'number_of_reviews', 'number_of_reviews_ltm', 'number_of_reviews_l30d', 'first_review',
        'last_review', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication',
         'review_scores_location', 'review_scores_value', 'instant_bookable', 'calculated_host_listings_count', 'calculated_host_listings_count_entire_homes',
          'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms', 'reviews_per_month', 'string_amenities'])


# Create price array of each neighbourhood_group_cleansed

g1 = df[df['Borough'] == "Manhattan"]['price'].to_list()
g2 = df[df['Borough'] == "Brooklyn"]['price'].to_list()
g3 = df[df['Borough'] == "Queens"]['price'].to_list()
g4 = df[df['Borough'] == "Staten Island"]['price'].to_list()
g5 = df[df['Borough'] == "Bronx"]['price'].to_list()


priceData = [g1, g2, g3, g4, g5]


group_labels=["Manhattan", "Brooklyn", "Queens", "Staten Island", "Bronx"]

#NY map figure

map_figure = px.scatter_mapbox(df, lat=df['latitude'], lon=df['longitude'],
                                 color="Borough",
                                 labels="Borough",
                                zoom=10, mapbox_style="open-street-map",
                                center=dict(lat=40.730610, lon=-73.935242),
                                width=1048, height=622,
                                size="price"
                                )


fig = ff.create_distplot(priceData, group_labels, bin_size=.2, show_curve=False)

app.layout = html.Div(children=[
    html.Section(className="header", children=[
    html.Div(className="heading", children=[
        html.Div(children=[
            html.H1(children="AirBnB Case Study")

            ]),

        html.Div(children=[
            html.H3(children="COSC2789 - Practical Data Science")

            ])

        ])
    ]),
    html.Section(className="main", children=[

        html.Div(id="airbnb-map", children=[

            html.Div(className="heading-name", children=[

                    html.H4(children="Map of AirBnB in New York City")

                ]),

                html.Div(className="map-container", children=[

                        dcc.Graph(
                            id="map-graph",
                            figure=map_figure

                            )

                    ])

            ]),


        html.Div(id="price-distribution", children=[

                html.Div(className="plot-container", children=[

                    html.Div(className="left-container", children=[

                            html.Div(className="plot-name", children=[

                                    html.H4(children="Price distributions")
                                ]),
                            html.Div(className="plot-description", children=[

                                    html.P(children="Price distribution is visualized by histogram.")
                                ])

                        ])    

                    ]),

                    html.Div(className="right-container", children=[
                        html.Div(className="hist-plot", children=[
                                dcc.Graph(
                                    id='example-graph',
                                    figure=fig
                                )
                            ])
                        ])

            ]),
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True)