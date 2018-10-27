
import plotly
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import charts as c
import plotly.tools as tls
import plotly.plotly as py

def testPlotly():
    plotly.tools.set_credentials_file(username='daveperry10', api_key='VW3xYEyVXvJOh1ca2gio')
    trace0 = go.Scatter(x=[1, 2, 3, 4], y=[10, 15, 13, 17])
    trace1 = go.Scatter(x=[1, 2, 3, 4],y=[16, 5, 11, 9])
    data = [trace0, trace1]
    #py.plot(data, filenam='basic-line', fileopt='overwrite', auto_open=False)
    plotly.offline.plot(data, filenam='basic-line', fileopt='overwrite', auto_open=False)

def testPlotlyfromMPL():
    chart = c.Chart(2,1)
    chart.chartBasic(pd.Series([0,1,4,2,3,4]),(0,1), kind='line')
    mpl_fig = chart.fig
    plotly_fig = tls.mpl_to_plotly(mpl_fig)
    unique_url = py.plot(plotly_fig, fileopt='overwrite', auto_open=False)
    print(unique_url)

def testSubPlots1():
    """ Use go.Figure()"""
    plotly.tools.set_credentials_file(username='daveperry10', api_key='VW3xYEyVXvJOh1ca2gio')
    trace1 = go.Scatter(x=[1, 2, 3],y=[2, 3, 4], name='Dave')
    trace2 = go.Scatter(x=[20, 30, 40],y=[5, 5, 5],xaxis='x2',yaxis='y')
    trace3 = go.Scatter(x=[2, 3, 4],y=[600, 700, 800],xaxis='x',yaxis='y3')
    trace4 = go.Scatter(x=[25, 28, 32],y=[7000, 8000, 9000],xaxis='x2',yaxis='y4')
    data = [trace1, trace2, trace3, trace4]
    # Order:  Right to left from bottom

    layout = go.Layout( xaxis=dict(domain=[0, 0.45]), yaxis=dict(domain=[0, 0.45]),
        xaxis2=dict(domain=[0.55, 1]), xaxis4=dict(domain=[0.55, 1],anchor='y4'),
        yaxis3=dict(domain=[0.55, 1]), yaxis4=dict(domain=[0.55, 1],anchor='x4'))

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='shared-axes-subplots')

def testSubPlots2():
    """ use plotly.tools.make_subplots() """

    trace1 = go.Scatter(x=[1, 2, 3], y=[4, 5, 6], name='Dave 1')
    trace2 = go.Scatter(x=[2, 3, 4], y=[5, 2, 3], name='Dave 2')
    trace3 = go.Scatter(x=[300, 400, 500], y=[600, 700, 800], name='Dave 3')
    trace4 = go.Scatter(x=[4000, 5000, 6000], y=[7000, 8000, 9000], name='Dave 4')

    fig = plotly.tools.make_subplots(rows=2, cols=2, subplot_titles=('Plot 1', 'Plot 2', 'Plot 3', 'Plot 4'))
    fig['layout'].update(height=600, width=600, title='Subplot for Kin')
    
    # Order is right to left from top
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 1)
    fig.append_trace(trace3, 2, 1)
    fig.append_trace(trace4, 2, 2)

    #fig['layout'].update(height=600, width=600, title='Subplot for Kin')
    py.iplot(fig, filename='make-subplots-multiple-with-titles', fileopt='overwrite', auto_open=False)
testSubPlots2()

def testPandasLine():
    N = 500
    x = np.linspace(0, 1, N)
    y = np.random.randn(N)
    df = pd.DataFrame({'x': x, 'y': y})
    df.head()
    df.name = "YO"
    # Do it with data
    data = [go.Scatter(x=df['x'], y=df['y'], name='2007')]
    url = py.plot(data, filename='passing data object directly', fileopt='overwrite', auto_open=False)

    # Do it with fig
    layout = go.Layout(
        title='passing fig instead of data',
        yaxis=dict(title='y axis label'),
        xaxis=dict(title='x axis label'))
    fig = go.Figure(data=data, layout=layout)
    url = py.plot(fig, filename='passing fig object', fileopt='overwrite', auto_open=False)

def testPandasHistogram():
    ser = pd.Series(30 * np.random.randn(500))
    data = [go.Histogram(x=ser, orientation='v', opacity=0.2)]
    url = py.plot(data, filename='pandas-simple-histogram', auto_open=False)
testPandasHistogram()


def testPandasRibbon():
    import urllib
    import numpy as np

    # url = "https://raw.githubusercontent.com/plotly/datasets/master/spectral.csv"
    # f = urllib.request(url)
    # spectra = np.loadtxt(f, delimiter=',')

    spectra = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/spectral.csv")
    traces = []
    y_raw = spectra.iloc[:, 0]  # wavelength
    sample_size = spectra.shape[1] - 1
    for i in range(1, sample_size):
        z_raw = spectra.iloc[:, i]
        x = []
        y = []
        z = []
        ci = int(255 / sample_size * i)  # ci = "color index"
        for j in range(0, len(z_raw)):
            z.append([z_raw[j], z_raw[j]])
            y.append([y_raw[j], y_raw[j]])
            x.append([i * 2, i * 2 + 1])
        traces.append(dict(
            z=z,
            x=x,
            y=y,
            colorscale=[[i, 'rgb(%d,%d,255)' % (ci, ci)] for i in np.arange(0, 1.1, 0.1)],
            showscale=False,
            type='surface',
        ))

    fig = {'data': traces, 'layout': {'title': 'Ribbon Plot'}}
    py.plot(fig, filename='ribbon-plot-python')
    return

#testPandasRibbon()
