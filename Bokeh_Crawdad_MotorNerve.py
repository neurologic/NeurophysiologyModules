#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 16:14:33 2021

@author: kperks
"""

import sys
from pathlib import Path

import scipy
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.signal import find_peaks
from copy import deepcopy
import math
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from bokeh.layouts import layout, row, column, gridplot, widgetbox
from bokeh.plotting import figure, show
from bokeh.io import output_file, curdoc
from bokeh.models import ColumnDataSource, HoverTool, CategoricalColorMapper, Column, Band
from bokeh.models import Button, RangeSlider, TextInput, CheckboxGroup
from bokeh.models.widgets import Tabs, Panel, Spinner
from bokeh.models.annotations import Label
from bokeh.models import MultiLine, Line, Range1d
from bokeh.palettes import Spectral6
from bokeh.themes import Theme
import yaml


#################
# tab 1 import data and explore
###################

def button_callback():
    sys.exit()  # Stop the server

def import_data():
    """
    function called when either filepath or fs are changed

    ToDo: check if file size is too big
    """
    # filepath = "/Users/kperks/mnt/PerksLab_rstore/neurophysiology_lab/CockroachLeg/CockroachLeg_20K2021-07-04T09_31_20.bin"
    # fs = 40000
    print('uploading data... this may take a moment. smile and relax')
    f_ = filepath.value.strip()
    
    nchan_ = int(nchan.value.strip())
    displaychan_ = int(displaychan.value.strip())
    
    #file_input is "new"
    fs_ = int(fs.value.strip())

    y_data = np.fromfile(Path(f_), dtype = np.float64)
    y_data = y_data.reshape(-1,nchan_)
    y_data = y_data[:,displaychan_]
    
    # 1 channel
    
    y_data = y_data - np.median(y_data)

    x_data = np.linspace(0,len(y_data)/fs_,len(y_data))
	
    max_val_slider = len(y_data)/fs_

    data = {'y' : y_data,'x' : x_data}
    new_data = ColumnDataSource(data = data)
    src_data.data.update(new_data.data)
	
    range_slider.update(end=max_val_slider)

    start_ = 0 #range_slider.value[0]
    stop_ = 1 #range_slider.value[1]
    range_selected = [start_,stop_]
    range_slider.update(value=(start_,stop_))
    new_selection = select_data(range_selected)
    data_selected.data.update(new_selection.data)
    print('data uploaded')

def select_data(range_selected):
	fs_ = int(fs.value.strip())
	y = src_data.data['y'][int(range_selected[0]*fs_):int(range_selected[1]*fs_)]
	x = src_data.data['x'][int(range_selected[0]*fs_):int(range_selected[1]*fs_)]
	
	data = {'y' : y,
		   'x' : x}
	return ColumnDataSource(data = data)
	
def button_plot_range_callback():
	start_ = range_slider.value[0]
	end_ = range_slider.value[1]
	new_selection = select_data([start_,end_])
	data_selected.data.update(new_selection.data)



# create exit button
button_exit = Button(label="Exit", button_type="success",width=100)
button_exit.on_click(button_callback)

# create text input for data file path
filepath = TextInput(title="path to data file",value="/Users/kperks/mnt/PerksLab_rstore/neurophysiology_lab/CrayfishNerve3/TelsonStim.bin",width=800)


# create import data button
button_import_data = Button(label="Import Data", button_type="success",width=100)
button_import_data.on_click(import_data)



# create plot range button
button_plot_range = Button(label="Plot X-Range", button_type="success",width=100)
button_plot_range.on_click(button_plot_range_callback)


# create text inpot for sampling rate
fs = TextInput(title="sampling rate",value='30000',width=100)

# flexible number of channels recorded in case also did intracell
nchan = TextInput(title="number of channels recorded in Bonsai",value='1',width=100)

displaychan = TextInput(title="which channel to display/analyze",value='0',width=100)

# create hover tool
hover = HoverTool(tooltips=[('mV', '@y'), ('time', '@x')])

# create figure
p = figure(plot_width=1000, plot_height=500,
				   tools=[hover,'pan','box_zoom','wheel_zoom','reset','save'],
				   x_axis_label = 'seconds',y_axis_label='Volts')

p.xaxis.major_label_text_font_size = "18pt"
p.xaxis.axis_label_text_font_size = "18pt"
p.yaxis.major_label_text_font_size = "18pt"
p.yaxis.axis_label_text_font_size = "18pt"

# initialize a range to plot
range_selected = [0,1]

# create range slider
range_slider = RangeSlider(
    title="Adjust x-axis range", # a title to display above the slider
    start=0,  # set the minimum value for the slider
    end=1,  # set the maximum value for the slider 
    step=1,  # increments for the slider
    value=(range_selected[0],range_selected[1]),  # initial values for slider (range_selected[0],range_selected[1])
    width=800
    )

# range_slider.js_link("value", p.x_range, "start", attr_selector=0)
# range_slider.js_link("value", p.x_range, "end", attr_selector=1)
# range_slider.on_change("value",update_plot1_slider)

# initialize data
src_data = ColumnDataSource({'y':[]})
data = {'x':[],'y':[]}
sumall_data = ColumnDataSource(data)
data_selected = ColumnDataSource(data)


# plot data within selected range as line
line = p.line('x','y',source=data_selected,line_color='black')

# collect controls
controls = [button_exit,fs,filepath,nchan,displaychan,button_import_data,range_slider,button_plot_range]

# layout controls
inputs = column(*controls)
# show(column(range_slider,p))

# layout all elements together
l = column(inputs, p)

# create tab
tab1 = Panel(child=l,title='import data and explore')

#############
# tab 2 look at clustered spike waveforms
#############


def cluster_spikes():
    threshold_ = float(spk_thresh.value.strip())
    polarity_ = float(polarity.value.strip())
    fs_ = float(fs.value.strip())
    
    
    nerve = src_data.data['y']
    
    max_val_slider = len(nerve)/fs_	
    range_slider2.update(end=max_val_slider)
    
    peaks,props = find_peaks(polarity_ * nerve,height=threshold_, 
                         prominence = threshold_, distance=(0.0005*fs_))
    peaks_t = peaks/fs_
    # peaks,props = find_peaks(polarity * nerve,height=0.01,prominence=0.01,distance=(0.001*fs))

    df = pd.DataFrame({
        'height': props['peak_heights'],
        'r_prom' : -nerve[peaks]+nerve[props['right_bases']],
        'l_prom' : -nerve[peaks]+nerve[props['left_bases']]
        # 'widths' : props['widths']/fs
            })
    
    #normalize data in dataframe for PCA
    df_normalized=(df - df.mean()) / df.std()
    pca = PCA(n_components=df.shape[1])
    pca.fit(df_normalized)
    
    X_pca=pca.transform(df_normalized)
    
    kmeans = KMeans(n_clusters=7).fit(X_pca[:,0:2])
    
    df['peaks_t'] = peaks_t
    df['cluster'] = kmeans.labels_
    
    #also use this function to plot the resulting scatter
    
    n,bins = np.histogram(props['peak_heights'],bins = 100)
    data_hist.data.update(
        {'n': n, 
         'bins' : bins[1:]
         })

    src_spkt.data = df
    
    start_ = 0 #range_slider.value[0]
    stop_ = 2 #range_slider.value[1]
    range_selected = [start_,stop_]
    range_slider2.update(value=(start_,stop_))
    new_selection = select_data2(range_selected)
    data_selected2.data.update(new_selection.data)
    
    
    df = df[((start_ < df['peaks_t']) & (df['peaks_t'] < stop_))]
    
    k=0
    bool_labels = df['cluster'].values==k
    data = {
        'x' : df['peaks_t'][bool_labels],
        'y' : polarity_ * df['height'][bool_labels]
        }
    data_spkt_0.data = data
    
    k=1
    bool_labels = df['cluster'].values==k
    data = {
        'x' : df['peaks_t'][bool_labels],
        'y' : polarity_ * df['height'][bool_labels]
        }
    data_spkt_1.data = data
    
    k=2
    bool_labels = df['cluster'].values==k
    data = {
        'x' : df['peaks_t'][bool_labels],
        'y' : polarity_ * df['height'][bool_labels]
        }
    data_spkt_2.data = data
    
    k=3
    bool_labels = df['cluster'].values==k
    data = {
        'x' : df['peaks_t'][bool_labels],
        'y' : polarity_ * df['height'][bool_labels]
        }
    data_spkt_3.data = data
    
    k=4
    bool_labels = df['cluster'].values==k
    data = {
        'x' : df['peaks_t'][bool_labels],
        'y' : polarity_ * df['height'][bool_labels]
        }
    data_spkt_4.data = data
    
    k=5
    bool_labels = df['cluster'].values==k
    data = {
        'x' : df['peaks_t'][bool_labels],
        'y' : polarity_ * df['height'][bool_labels]
        }
    data_spkt_5.data = data
    
    k=6
    bool_labels = df['cluster'].values==k
    data = {
        'x' : df['peaks_t'][bool_labels],
        'y' : polarity_ * df['height'][bool_labels]
        }
    data_spkt_6.data = data
    
    # plot first cluster (kluster 0 )
    # include all waveforms from recording
    windur = 0.001
    winsamps = int(windur * fs_)
    
    k = 0
    df = pd.DataFrame(src_spkt.data)
    bool_labels = df['cluster'].values==k
    spkt = df[bool_labels]['peaks_t'].values
    
    spkwav = [nerve[(int(t*fs_)-winsamps):(int(t*fs_)+winsamps)] for t in spkt]
    xtime = [np.linspace(-windur,windur,winsamps*2)*1000 for t in spkt]

    data_cluster.data = {
        'xs':xtime,
        'ys':spkwav}
    



def select_data2(range_selected):
	fs_ = int(fs.value.strip())
	y = src_data.data['y'][int(range_selected[0]*fs_):int(range_selected[1]*fs_)]
	x = src_data.data['x'][int(range_selected[0]*fs_):int(range_selected[1]*fs_)]
	
	data = {'y' : y,
		   'x' : x}
	return ColumnDataSource(data = data)
	
def button_plot_range_callback2():
    polarity_ = float(polarity.value.strip())
    start_ = range_slider2.value[0]
    stop_ = range_slider2.value[1]
    new_selection = select_data2([start_,stop_])
    data_selected2.data.update(new_selection.data)
    
    df = pd.DataFrame(src_spkt.data)
    df = df.loc[((start_ < df['peaks_t']) & (df['peaks_t'] < stop_))]
    
    k=0
    bool_labels = df['cluster'].values==k
    data = {
        'x' : df['peaks_t'][bool_labels],
        'y' : polarity_ * df['height'][bool_labels]
        }
    data_spkt_0.data = data
    
    k=1
    bool_labels = df['cluster'].values==k
    data = {
        'x' : df['peaks_t'][bool_labels],
        'y' : polarity_ * df['height'][bool_labels]
        }
    data_spkt_1.data = data
    
    k=2
    bool_labels = df['cluster'].values==k
    data = {
        'x' : df['peaks_t'][bool_labels],
        'y' : polarity_ * df['height'][bool_labels]
        }
    data_spkt_2.data = data
    
    k=3
    bool_labels = df['cluster'].values==k
    data = {
        'x' : df['peaks_t'][bool_labels],
        'y' : polarity_ * df['height'][bool_labels]
        }
    data_spkt_3.data = data
    
    k=4
    bool_labels = df['cluster'].values==k
    data = {
        'x' : df['peaks_t'][bool_labels],
        'y' : polarity_ * df['height'][bool_labels]
        }
    data_spkt_4.data = data
    
    k=5
    bool_labels = df['cluster'].values==k
    data = {
        'x' : df['peaks_t'][bool_labels],
        'y' : polarity_ * df['height'][bool_labels]
        }
    data_spkt_5.data = data
    
    k=6
    bool_labels = df['cluster'].values==k
    data = {
        'x' : df['peaks_t'][bool_labels],
        'y' : polarity_ * df['height'][bool_labels]
        }
    data_spkt_6.data = data
    
def update_cluster_waveplot(attr,old,new):
    k = int(plot_cluster.value)
    print('cluster number being plotted = ' + str(k))
    
    fs_ = float(fs.value.strip())
    windur = 0.001
    winsamps = int(windur * fs_)
    
    df = pd.DataFrame(src_spkt.data)
    spkt = df.loc[df['cluster']==k]['peaks_t'].values

    nerve = src_data.data['y']
    
    spkwav = [nerve[(int(t*fs_)-winsamps):(int(t*fs_)+winsamps)] for t in spkt]
    xtime = [np.linspace(-windur,windur,winsamps*2)*1000 for t in spkt]
    
    cols = ['red','blue','purple','green','orange','cyan','brown']
    display_nspk.update(value=str(len(spkt)) + ' spikes are assigned to this cluster; color = ' + cols[k])

    data_cluster.data = {
        'xs':xtime,
        'ys':spkwav}
    
def overlay_all_mean_spikes():
    # plot mean and sem for all spikes; each in a different color
    fs_ = float(fs.value.strip())
    windur = 0.002
    winsamps = int(windur * fs_)
    
    df = pd.DataFrame(src_spkt.data)
    nerve = src_data.data['y']
    
    xtime = np.linspace(-windur,windur,winsamps*2)*1000
    
    
    k=0
    spkt = df.loc[df['cluster']==k]['peaks_t'].values
    spkwav = np.asarray([nerve[(int(t*fs_)-winsamps):(int(t*fs_)+winsamps)] for t in spkt])
    wav_u = np.mean(spkwav,0)
    wav_std = np.std(spkwav,0)
    cluster_mean_0.data = {
        'x':xtime,
        'y':wav_u,
        'upper':wav_u+wav_std,
        'lower':wav_u-wav_std
        }

    k=1
    spkt = df.loc[df['cluster']==k]['peaks_t'].values
    spkwav = np.asarray([nerve[(int(t*fs_)-winsamps):(int(t*fs_)+winsamps)] for t in spkt])
    wav_u = np.mean(spkwav,0)
    wav_std = np.std(spkwav,0)
    cluster_mean_1.data = {
        'x':xtime,
        'y':wav_u,
        'upper':wav_u+wav_std,
        'lower':wav_u-wav_std
        }
    
    k=2
    spkt = df.loc[df['cluster']==k]['peaks_t'].values
    spkwav = np.asarray([nerve[(int(t*fs_)-winsamps):(int(t*fs_)+winsamps)] for t in spkt])
    wav_u = np.mean(spkwav,0)
    wav_std = np.std(spkwav,0)
    cluster_mean_2.data = {
        'x':xtime,
        'y':wav_u,
        'upper':wav_u+wav_std,
        'lower':wav_u-wav_std
        }
    
    k=3
    spkt = df.loc[df['cluster']==k]['peaks_t'].values
    spkwav = np.asarray([nerve[(int(t*fs_)-winsamps):(int(t*fs_)+winsamps)] for t in spkt])
    wav_u = np.mean(spkwav,0)
    wav_std = np.std(spkwav,0)
    cluster_mean_3.data = {
        'x':xtime,
        'y':wav_u,
        'upper':wav_u+wav_std,
        'lower':wav_u-wav_std
        }
    
    k=4
    spkt = df.loc[df['cluster']==k]['peaks_t'].values
    spkwav = np.asarray([nerve[(int(t*fs_)-winsamps):(int(t*fs_)+winsamps)] for t in spkt])
    wav_u = np.mean(spkwav,0)
    wav_std = np.std(spkwav,0)
    cluster_mean_4.data = {
        'x':xtime,
        'y':wav_u,
        'upper':wav_u+wav_std,
        'lower':wav_u-wav_std
        }
    
    k=5
    spkt = df.loc[df['cluster']==k]['peaks_t'].values
    spkwav = np.asarray([nerve[(int(t*fs_)-winsamps):(int(t*fs_)+winsamps)] for t in spkt])
    wav_u = np.mean(spkwav,0)
    wav_std = np.std(spkwav,0)
    cluster_mean_5.data = {
        'x':xtime,
        'y':wav_u,
        'upper':wav_u+wav_std,
        'lower':wav_u-wav_std
        }
    
    k=6
    spkt = df.loc[df['cluster']==k]['peaks_t'].values
    spkwav = np.asarray([nerve[(int(t*fs_)-winsamps):(int(t*fs_)+winsamps)] for t in spkt])
    wav_u = np.mean(spkwav,0)
    wav_std = np.std(spkwav,0)
    cluster_mean_6.data = {
        'x':xtime,
        'y':wav_u,
        'upper':wav_u+wav_std,
        'lower':wav_u-wav_std
        }
    

    
    
# create hover tool
hover = HoverTool(tooltips=[('mV', '@y'), ('time', '@x')])

# create figure for tab 4
p2a = figure(plot_width=1000, plot_height=500,
				   tools=[hover,'pan','box_zoom','wheel_zoom','reset','save'],
				   y_axis_label='number of spikes', x_axis_label='Volts',
				   title='histogram of spike amplitudes (height above 0 V; 100 evenly spaced bins from min to max voltage)')
p2a.xaxis.major_label_text_font_size = "18pt"
p2a.xaxis.axis_label_text_font_size = "18pt"
p2a.yaxis.major_label_text_font_size = "18pt"
p2a.yaxis.axis_label_text_font_size = "18pt"

# create figure for tab 4
p2 = figure(plot_width=1000, plot_height=500,
				   tools=[hover,'pan','box_zoom','wheel_zoom','reset','save'],
				   y_axis_label='V', x_axis_label='time (milliseconds)',
				   title='spike detections and categorical assignment')
p2.xaxis.major_label_text_font_size = "18pt"
p2.xaxis.axis_label_text_font_size = "18pt"
p2.yaxis.major_label_text_font_size = "18pt"
p2.yaxis.axis_label_text_font_size = "18pt"

# create figure 2 for tab 4
p2b = figure(plot_width=1000, plot_height=500,
				   tools=[hover,'pan','box_zoom','wheel_zoom','reset','save'],
				   y_axis_label='V', x_axis_label='time from spike peak (milliseconds)',
				   title='overlay of all waveforms from chosen spike cluster')
p2b.xaxis.major_label_text_font_size = "18pt"
p2b.xaxis.axis_label_text_font_size = "18pt"
p2b.yaxis.major_label_text_font_size = "18pt"
p2b.yaxis.axis_label_text_font_size = "18pt"

# create figure 2 for tab 4
p2c = figure(plot_width=1000, plot_height=800,
				   tools=[hover,'pan','box_zoom','wheel_zoom','reset','save'],
				   y_axis_label='V', x_axis_label='time from spike peak (milliseconds)',
				   title='overlay of mean waveform from each spike cluster (+/- std)')
p2c.xaxis.major_label_text_font_size = "18pt"
p2c.xaxis.axis_label_text_font_size = "18pt"
p2c.yaxis.major_label_text_font_size = "18pt"
p2c.yaxis.axis_label_text_font_size = "18pt"

# create plot range button
button_plot_range2 = Button(label="Plot X-Range", button_type="success",width=100)
button_plot_range2.on_click(button_plot_range_callback2)

# initialize a range to plot
range_selected = [0,1]

# create range slider
range_slider2 = RangeSlider(
    title="Adjust x-axis range", # a title to display above the slider
    start=0,  # set the minimum value for the slider
    end=1,  # set the maximum value for the slider 
    step=1,  # increments for the slider
    value=(range_selected[0],range_selected[1]),  # initial values for slider (range_selected[0],range_selected[1])
    width=800
    )

# create text input for spike threshold
spk_thresh = TextInput(title="Spike Threshold (from examining raw data; seconds)",value='', width=100)

# create text input for spike threshold
polarity = TextInput(title="Polarity for spike detections: -1 if detections are better for negative peaks; 1 if upward",value='', width=100)

detect_spikes = Button(label="Detect Spikes and Cluster Identity", button_type="success",width=100)
detect_spikes.on_click(cluster_spikes)
# overlay_all_button.on_click(overlay_all_mean_spikes)

# need to add in ability to select a portion of the data (in case recording messed up, etc)

# which spike to plot
plot_cluster = Spinner(title="spike cluster to plot",low=0,high=6, value=0, width=100)
plot_cluster.on_change("value",update_cluster_waveplot)

display_nspk = TextInput(title="Number of Spikes in Selected Cluster",value='', width=800)
# p2b.add_layout(display_nspk)

overlay_all_button = Button(label="Plot All Mean Spike Waveforms", button_type="success",width=100)
overlay_all_button.on_click(overlay_all_mean_spikes)


# initialize data_spikes ColumnDataSource - spike datamat to plot from
data_cluster = ColumnDataSource(data = {
    'xs':[],
	'ys':[]
	})

# initialize all of the cluster spike times for scatter
src_spkt = ColumnDataSource(pd.DataFrame({
    'height' : [],
    'r_prom' : [],
    'l_prom' : [],
    'peaks_t' : [],
    'cluster' : []
    }))
    

data_spkt_0 = ColumnDataSource(data = {
	'y':[],
	'x':[]
	})
data_spkt_1 = ColumnDataSource(data = {
	'y':[],
	'x':[]
	})
data_spkt_2 = ColumnDataSource(data = {
	'y':[],
	'x':[]
	})
data_spkt_3 = ColumnDataSource(data = {
	'y':[],
	'x':[]
	})
data_spkt_4 = ColumnDataSource(data = {
	'y':[],
	'x':[]
	})
data_spkt_5 = ColumnDataSource(data = {
	'y':[],
	'x':[]
	})
data_spkt_6 = ColumnDataSource(data = {
	'y':[],
	'x':[]
	})

cluster_mean_0 = ColumnDataSource(data = {
	'y':[],
	'x':[]
	})
cluster_mean_1 = ColumnDataSource(data = {
	'y':[],
	'x':[]
	})
cluster_mean_2 = ColumnDataSource(data = {
	'y':[],
	'x':[]
	})
cluster_mean_3 = ColumnDataSource(data = {
	'y':[],
	'x':[]
	})
cluster_mean_4 = ColumnDataSource(data = {
	'y':[],
	'x':[]
	})
cluster_mean_5 = ColumnDataSource(data = {
	'y':[],
	'x':[]
	})
cluster_mean_6 = ColumnDataSource(data = {
	'y':[],
	'x':[]
	})

data_overlay_allspkwav = ColumnDataSource(data = {
	'xs':[],
	'ys':[]
	})

data_overlay_individspkwav = ColumnDataSource(data = {
	'y':[],
	'x':[]
	})

data_selected2 = ColumnDataSource(data = {
    'y':[],
    'x':[]
    })

data_hist = ColumnDataSource(data = {
    'n':[],
    'bins':[]
    })


# initialize line plot for spike waveform
p2a.vbar(x='bins',top='n',source=data_hist,line_color='black',fill_color='gray',width=0.0001)

p2.line(x = 'x', y='y',source =data_selected2,line_color='black')

cols = ['red','blue','purple','green','orange','cyan','brown']
p2.circle(x='x',y='y',source=data_spkt_0,color=cols[0],size=6,level='overlay')
p2.circle(x='x',y='y',source=data_spkt_1,color=cols[1],size=6,level='overlay')
p2.circle(x='x',y='y',source=data_spkt_2,color=cols[2],size=6,level='overlay')
p2.circle(x='x',y='y',source=data_spkt_3,color=cols[3],size=6,level='overlay')
p2.circle(x='x',y='y',source=data_spkt_4,color=cols[4],size=6,level='overlay')
p2.circle(x='x',y='y',source=data_spkt_5,color=cols[5],size=6,level='overlay')
p2.circle(x='x',y='y',source=data_spkt_6,color=cols[6],size=6,level='overlay')


glyph = MultiLine(xs='xs',ys='ys',line_alpha=0.5)
p2b.add_glyph(data_cluster,glyph)


p2c.line(x = 'x', y='y',source =cluster_mean_0,line_color=cols[0],line_width=2)
p2c.line(x = 'x', y='y',source =cluster_mean_1,line_color=cols[1],line_width=2)
p2c.line(x = 'x', y='y',source =cluster_mean_2,line_color=cols[2],line_width=2)
p2c.line(x = 'x', y='y',source =cluster_mean_3,line_color=cols[3],line_width=2)
p2c.line(x = 'x', y='y',source =cluster_mean_4,line_color=cols[4],line_width=2)
p2c.line(x = 'x', y='y',source =cluster_mean_5,line_color=cols[5],line_width=2)
p2c.line(x = 'x', y='y',source =cluster_mean_6,line_color=cols[6],line_width=2)

band_0 = Band(base='x', lower='lower', upper='upper', source=cluster_mean_0, fill_alpha=0.15,fill_color = cols[0])
band_1 = Band(base='x', lower='lower', upper='upper', source=cluster_mean_1, fill_alpha=0.15,fill_color = cols[1])
band_2 = Band(base='x', lower='lower', upper='upper', source=cluster_mean_2, fill_alpha=0.15,fill_color = cols[2])
band_3 = Band(base='x', lower='lower', upper='upper', source=cluster_mean_3, fill_alpha=0.15,fill_color = cols[3])
band_4 = Band(base='x', lower='lower', upper='upper', source=cluster_mean_4, fill_alpha=0.15,fill_color = cols[4])
band_5 = Band(base='x', lower='lower', upper='upper', source=cluster_mean_5, fill_alpha=0.15,fill_color = cols[5])
band_6 = Band(base='x', lower='lower', upper='upper', source=cluster_mean_6, fill_alpha=0.15,fill_color = cols[6])

p2c.add_layout(band_0)
p2c.add_layout(band_1)
p2c.add_layout(band_2)
p2c.add_layout(band_3)
p2c.add_layout(band_4)
p2c.add_layout(band_5)
p2c.add_layout(band_6)



# # initialize line plot for all spikes overlay
# # use multiline to plot data
# glyph = MultiLine(xs='xs',ys='ys')
# p4b.add_glyph(data_overlay4,glyph)
# """
# ADD show average rate
# """
# p4b.line(x = 'x', y='y',source =data_overlay4_mean,line_color='orange',line_width=6,alpha=0.5)

# collect controls and layout all elements
controls2 = [spk_thresh,polarity,detect_spikes]
inputs2 = column(*controls2)
l2 = column(inputs2,p2a,range_slider2,button_plot_range2,p2,plot_cluster,display_nspk,p2b,overlay_all_button,p2c)

# create tab
tab2 = Panel(child=l2,title='cluster spikes')



##############
# create app
##


tabs = Tabs(tabs=[tab1,tab2])
curdoc().add_root(tabs)