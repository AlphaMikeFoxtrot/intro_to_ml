import plotly
from plotly.graph_objs import Scatter, Layout, Annotation,Marker,Font,XAxis,YAxis

# Scientific libraries
from scipy import stats

xi = [elem for elem in range(0,9)]

# (Almost) linear sequence
y = [19, 20, 20.5, 21.5, 22, 23, 23, 25.5, 24]

# Generated linear fit
slope, intercept, r_value, p_value, std_err = stats.linregress(xi,y)
line = [slope*x+intercept for x in xi]
prediction = slope*(xi[-1]+1)+intercept

# Creating the dataset, and generating the plot
trace1 = Scatter(
                  x=xi, 
                  y=y, 
                  mode='markers',
                  marker=Marker(color='rgb(255, 127, 14)'),
                  name='Data'
                  )

trace2 = Scatter(
                  x=xi, 
                  y=line, 
                  mode='line',
                  marker=Marker(color='rgb(31, 119, 180)'),
                  name='Fit'
                  )

trace3 = Scatter(
                  x=[xi[-1]+1], 
                  y=[prediction], 
                  mode='markers',
                  marker=Marker(color='rgb(31, 119, 180)'),
                  name='Fit'
                  )

annotation = Annotation(
                  x=3.5,
                  y=23.5,
                  text='$R^2 = 0.9551,\\Y = 0.716X + 19.18$',
                  showarrow=False,
                  font=Font(size=16)
                  )
layout = Layout(
                title='Linear Fit in Python',
                plot_bgcolor='rgb(229, 229, 229)',
                  xaxis=XAxis(zerolinecolor='rgb(255,255,255)', gridcolor='rgb(255,255,255)'),
                  yaxis=YAxis(zerolinecolor='rgb(255,255,255)', gridcolor='rgb(255,255,255)'),
                  annotations=[annotation]
                )

to_scatter = [trace1,trace2,trace3]
plotly.offline.plot({
    "data":to_scatter,
    "layout":layout
})
