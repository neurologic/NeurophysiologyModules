import sys
from pathlib import Path

import scipy
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.signal import find_peaks
from copy import deepcopy
import math

from bokeh.layouts import layout, row, column, gridplot, widgetbox
from bokeh.plotting import figure, show
from bokeh.io import output_file, curdoc
from bokeh.models import ColumnDataSource, HoverTool, CategoricalColorMapper, Column 
from bokeh.models import Button, RangeSlider, TextInput
from bokeh.models.widgets import Tabs, Panel, Spinner
from bokeh.models import MultiLine, Line, Range1d
from bokeh.palettes import Spectral6
from bokeh.themes import Theme
import yaml


#################
# tab 1 import data and explore
###################

def button_callback():
	sys.exit()  # Stop the server

def import_data(attr,old,new):
	"""
	function called when either filepath or fs are changed

	ToDo: check if file size is too big
	"""
	# filepath = "/Users/kperks/mnt/PerksLab_rstore/neurophysiology_lab/CockroachLeg/CockroachLeg_20K2021-07-04T09_31_20.bin"
	# fs = 40000
	print('uploading data... this may take a moment. smile and relax')
	f_ = filepath.value.strip()

	#file_input is "new"
	fs_ = int(fs.value.strip())
	
	y_data = np.fromfile(Path(f_), dtype = np.float64)
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
	
def update_plot1_slider(attr,old,new):
	start_ = range_slider.value[0]
	end_ = range_slider.value[1]
	new_selection = select_data([start_,end_])
	data_selected.data.update(new_selection.data)


# create exit button
button = Button(label="Exit", button_type="success",width=100)
button.on_click(button_callback)

# create text input for data file path
filepath = TextInput(title="path to data file",value="PathToFile",width=800)
filepath.on_change("value", import_data)

# create text inpot for sampling rate
fs = TextInput(title="sampling rate",value='40000',width=100)
fs.on_change("value",import_data)

# create hover tool
hover = HoverTool(tooltips=[('V', '@y'), ('time', '@x')])

# create figure
p = figure(plot_width=1000, plot_height=500,
				   tools=[hover,'pan','box_zoom','wheel_zoom','reset','save'])

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

range_slider.js_link("value", p.x_range, "start", attr_selector=0)
range_slider.js_link("value", p.x_range, "end", attr_selector=1)
range_slider.on_change("value",update_plot1_slider)

# initialize data
data = {'x':[],'y':[]}
src_data = ColumnDataSource(data)
data_selected = ColumnDataSource(data)


# plot data within selected range as line
line = p.line('x','y',source=data_selected,line_color='black')

# collect controls
controls = [fs,filepath,range_slider,button]

# layout controls
inputs = column(*controls)
# show(column(range_slider,p))

# layout all elements together
l = column(inputs, p)

# create tab
tab1 = Panel(child=l,title='import data and explore')



################
# tab3 interspike interval and raster
#################

def update_plot3(attr,old,new):
	print('calculating instantaneous rate and updating plots')
	fs_ = int(fs.value.strip())

	trial_dur = float(trial_duration3.value.strip()) # sec

	trial_time = float(trial_times3.value.strip()) #trial times list text input must not have spaces

	spk_thresh_ = float(spk_thresh3.value.strip())

	#create a new dictionary to store new data to plot temporarily
	datamat={'y':[],'x':[]}

	
	win0 = int(trial_time*fs_)
	win1 = np.min([win0+int(trial_dur*fs_),len(dict(src_data.data)['y'])])
	y = deepcopy(dict(src_data.data)['y'][win0:win1])
	x = deepcopy(dict(src_data.data)['x'][win0:win1])

	data_trial3.data = {'x' : x,'y' : deepcopy(dict(src_data.data)['y'][win0:win1])}

	y[y<=spk_thresh_] = 0

	#find peaks of y (don't allow EODs within 1ms of each other)
	peaks,props = find_peaks(y,distance = int(fs_*0.001))

	peak_t = np.asarray([x[p] for p in peaks])

	rate = 1/np.diff(peak_t)
	
	datamat['y'] = rate #np.asarray(ys)
	datamat['x'] = peak_t[1:] #np.asarray(xs)

	data_spktimes3.data = {'x':peak_t,'y':np.zeros(len(peak_t))}

	data_scatter.data = datamat

	ydr3.start= -10
	ydr3.end= np.max(rate)+10
	xdr3.start = win0/fs_
	xdr3.end = win1/fs_

xdr3 = Range1d(start=0,end=5)
ydr3 = Range1d(start=-10,end=50)


# create figure for scatter of rate on tab 3
p3 = figure(plot_width=1000, plot_height=500,
				   tools=['hover','pan','box_zoom','lasso_select','wheel_zoom','reset','save'],
				   y_axis_label='EOD rate', x_axis_label='time of EOD',
				   x_range=xdr3,y_range=ydr3)

# create figure for EOD detections tab 3
p3b = figure(plot_width=1000, plot_height=500,
				   tools=['hover','pan','box_zoom','wheel_zoom','reset','save'],
				   title = 'Raw data to show spike detection',
				   x_axis_label = 'seconds')


# create text input for trial duration
trial_duration3 = TextInput(title="Duration of analysis window",value='1', width=100)
trial_duration3.on_change("value",update_plot3)

# create text input for trial time
trial_times3 = TextInput(title="Start Time for IPI analysis", width=100)
trial_times3.on_change("value",update_plot3)

# create text input for spike threshold
spk_thresh3 = TextInput(title="Detection Threshold (from examining raw data; V)",value='0.04', width=100)
spk_thresh3.on_change("value",update_plot3)

# initialize data_overlay ColumnDataSource
data_scatter = ColumnDataSource(data = {
	'y':[],
	'x':[]
	})

# initialize data_overlay ColumnDataSource
data_trial3 = ColumnDataSource(data = {
	'y':[],
	'x':[]
	})

data_spktimes3 = ColumnDataSource(data = {
	'y':[],
	'x':[]
	})


# plot ipi scatter
p3.circle(x='x',y='y',source=data_scatter,color='black')

# use multiline to plot data
p3b.line(x = 'x', y='y',source =data_trial3,line_color='black')
p3b.circle(x='x',y='y',source=data_spktimes3,color='red',size=6,level='overlay')

# collect controls and layout all elements
controls3 = [trial_times3,trial_duration3,spk_thresh3]
inputs3 = column(*controls3)
l3 = column(inputs3,p3,p3b)

# create tab
tab3 = Panel(child=l3,title='IPI scatter plot')


#############
# tab 4 scroll through spike waveforms
#############

def update_data4(attr,old,new):
	"""
	creates a matrix with each column a different spike waveform
	rows are time/samples
	"""
	fs_ = int(fs.value.strip())
	# filtert = int(0.01*fs_)
	windur = float(plot_duration4.value.strip())
	offset_t = windur/2 # 500 msec include a pre-stimulus onset
	
	trialdur = float(trial_duration4.value.strip()) # sec

	trial_times_ = trial_times4.value.split(',') #trial times list text input must not have spaces
	trial_times_ = [float(t) for t in trial_times_]

	spk_thresh_ = float(spk_thresh4.value.strip())

	#create a new dictionary to store new data to plot temporarily
	datamat={'ys':[]}
	xtime = np.linspace(0,trialdur,int((trialdur)*fs_))

	ys = []
	xs = []
	spks = []
	for i,t in enumerate(trial_times_):
		win0 = int(t*fs_)
		win1 = win0+int(trialdur*fs_)
		y_trial = dict(src_data.data)['y'][win0:win1]
		y = deepcopy(dict(src_data.data)['y'][win0:win1])	
		y[y<=spk_thresh_] = 0

		#find peaks of y
		peaks,props = find_peaks(y,distance = int(fs_*0.0005))

		peak_t = np.asarray([xtime[p] for p in peaks])
		spks.extend(peak_t)


		for j,s in enumerate(spks):

			if (((s-offset_t)>0) & ((s+offset_t)<trialdur)):
				win0_ = int((s-offset_t)*fs_)
				win1_ = int((s+offset_t)*fs_)

				ys.append(deepcopy(y_trial[win0_:win1_]))

	datamat['ys'] = np.asarray(ys).T
	data_spikes.data = datamat

	print('total spikes = len(ys) = ' + str(len(ys)))

	data_plot4.data = {
		'x':np.linspace(-offset_t*1000,offset_t*1000,int((windur)*fs_)),
		'y':deepcopy(dict(data_spikes.data)['ys'][:,0])
		}

	plot_spike4.update(high=len(ys)-1)
	plot_spike4.update(value=0)

	spklist4.update(value='0')
	spklist4.update(value='all')


def update_plot4(attr,old,new):
	spknum = int(plot_spike4.value)
	print('spike number being plotted = ' + str(spknum))

	x = data_plot4.data['x']
	data_plot4.data = {
		'x':x,
		'y':deepcopy(dict(data_spikes.data))['ys'][:,spknum]}

def update_overlay4(attr,old,new):
	print('overlaying specified spikes')
	spks_to_overlay = spklist4.value.split(',')

	print(spks_to_overlay)

	if spks_to_overlay[0]=='all':
		spks_to_overlay = np.arange(np.shape(data_spikes.data['ys'])[1])
		print('plotting all spikes; total number = ')
		print(np.shape(data_spikes.data['ys'])[1])

	spks_to_overlay = [int(i) for i in spks_to_overlay]

	x = data_plot4.data['x']

	xs = []
	ys = []
	for i in spks_to_overlay:
		xs.append(x)
		ys.append(deepcopy(dict(data_spikes.data))['ys'][:,i])

	print('updating overlay data')
	data_overlay4.data = {
		'xs':xs,
		'ys':ys 
		}

	print('updating mean of overlay')
	data_overlay4_mean.data = {
		'x':x,
		'y':np.mean(np.asarray(ys),0)
	}

hover = HoverTool(tooltips=[('V', '@y'), ('time', '@x')],mode='vline',point_policy='snap_to_data')
# create figure for tab 4
p4 = figure(plot_width=1000, plot_height=500,
				   tools=[hover,'pan','box_zoom','wheel_zoom','reset','save'],
				   y_axis_label='V', x_axis_label='time from spike peak (milliseconds',
				   title='individual spikes')


# create figure 2 for tab 4
p4b = figure(plot_width=1000, plot_height=500,
				   tools=['pan','box_zoom','wheel_zoom','reset','save'],
				   y_axis_label='V', x_axis_label='time from spike peak (milliseconds',
				   title='overlay of chosen spikes')


# create text input for trial duration
trial_duration4 = TextInput(title="Duration of trials (seconds)",value='1', width=100)
trial_duration4.on_change("value",update_data4)

# create text input for trial times
trial_times4 = TextInput(title="List of Trial Start times (comma-separated no spaces; seconds)", width=800)
trial_times4.on_change("value",update_data4)

# create text input for spike threshold
spk_thresh4 = TextInput(title="Spike Threshold (from examining raw data; seconds)",value='0.05', width=100)
spk_thresh4.on_change("value",update_data4)

plot_duration4 = TextInput(title="Duration of analysis window (seconds)",value='0.005', width=100)
plot_duration4.on_change("value",update_data4)

plot_spike4 = Spinner(title="spike number to plot",low=0, width=100,value=0)
plot_spike4.on_change("value",update_plot4)

spklist4 = TextInput(title="spike indices to overlay (either 'all' or a comma-separated list of indices)", width=800, value = 'all')
spklist4.on_change("value",update_overlay4)

# create a button to export current waveform to concatenate with an h5 file
""" each waveform exported must have same duration """
# button_exportTOh5 = Button(label="Export Waveform", button_type="success")
# button_exportTOh5.on_click(button_exportwaveformtoh5file)

# initialize data_spikes ColumnDataSource - spike datamat to plot from
data_spikes = ColumnDataSource(data = {
	'ys':[]
	})

# initialize data_plot4 ColumnDataSource - spike waveform to plot
data_plot4 = ColumnDataSource(data = {
	'y':[],
	'x':[]
	})

data_overlay4 = ColumnDataSource(data = {
	'xs':[],
	'ys':[]
	})

data_overlay4_mean = ColumnDataSource(data = {
	'y':[],
	'x':[]
	})

# initialize line plot for spike waveform
p4.line(x = 'x', y='y',source =data_plot4,line_color='black')

# initialize line plot for all spikes overlay
# use multiline to plot data
glyph = MultiLine(xs='xs',ys='ys')
p4b.add_glyph(data_overlay4,glyph)
"""
ADD show average rate
"""
meanline = p4b.line(x = 'x', y='y',source =data_overlay4_mean,line_color='orange',line_width=6,alpha=0.5)
p4b.add_tools(HoverTool(tooltips=[("msec","@x"),("V","@y")],
	renderers=[meanline],mode='vline',point_policy='snap_to_data'))

# collect controls and layout all elements
controls4 = [trial_times4,trial_duration4,spk_thresh4,plot_duration4,plot_spike4,spklist4]
inputs4 = column(*controls4)
l4 = column(inputs4,p4,p4b)

# create tab
tab4 = Panel(child=l4,title='plot EOD waveforms')


#######
# create tabs layout
######

tabs = Tabs(tabs=[tab1,tab3,tab4])

curdoc().add_root(tabs)

