import plotly.express as px
import pandas as pd
import numpy as np

df = pd.read_csv('dataset/boxplot/original.csv')

df.replace('-', np.nan, inplace=True)

df = df.astype(float)

# Create a box plot using Plotly Express
fig = px.box(df, y=df.columns, title='Box Plot for Scores', labels={'variable': 'Nature', 'value': 'Score'})
fig.show()
