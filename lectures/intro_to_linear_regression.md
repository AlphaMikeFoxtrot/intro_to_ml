#Linear regression

Linear Regression is a powerful and useful technique, that is easy to interpret and use.  The classical method of using Linear Regression is called ordinary least squares - where you take the square of the distance of each point of data and compare it against some line that you fit to the data.  This line is initialized naively and then using some method (typically gradient descent) the distance between the line and all points are minimized.  This creates a "linearized" fit of the data points.  This linearized fit is a mathematical abstraction of the data set in question.  From it we can use all of the grade school math we learned to better understand our approximation of the data.  And thus geometry, trigonometry, algebra and many other basic disciplines find relevance in the world of work.

##A worked example

Will start with as minimal a python example as we can.  We'll be making use of plotly's linear fit for this and standard data.  Let's start with installation:

You'll need `pip` python's standard package manager.  We'll be using Python3, scipy and plotly for this example.

Let's start with the import statement - 

`import plotly`
`from scipy import stats`

```
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
```
