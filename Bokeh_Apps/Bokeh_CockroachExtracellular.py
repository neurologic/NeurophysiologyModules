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
from bokeh.models import Button, RangeSlider, TextInput, CheckboxGroup
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

# def import_data(attr,old,new):
# 	"""
# 	function called when either filepath or fs are changed

# 	ToDo: check if file size is too big
# 	"""
# 	# filepath = "/Users/kperks/mnt/PerksLab_rstore/neurophysiology_lab/CockroachLeg/CockroachLeg_20K2021-07-04T09_31_20.bin"
# 	# fs = 40000
# 	print('uploading data... this may take a moment. smile and relax')
# 	f_ = filepath.value.strip()

# 	#file_input is "new"
# 	fs_ = int(fs.value.strip())
# 	
# 	y_data = np.fromfile(Path(f_), dtype = np.float64)
# 	y_data = y_data - np.median(y_data)
# 	x_data = np.linspace(0,len(y_data)/fs_,len(y_data))

# 	max_val_slider = len(y_data)/fs_

# 	data = {'y' : y_data,'x' : x_data}
# 	new_data = ColumnDataSource(data = data)
# 	src_data.data.update(new_data.data)
# 	
# 	range_slider.update(end=max_val_slider)

# 	start_ = 0 #range_slider.value[0]
# 	stop_ = 1 #range_slider.value[1]
# 	range_selected = [start_,stop_]
# 	new_selection = select_data(range_selected)
# 	data_selected.data.update(new_selection.data)
# 	print('data uploaded')
    
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
    # nervechan_ = int(nervechan_.value.strip())
    # synapchan_ = int(synapchan_.value.strip())
    # simultaneous post and pre synaptic recording so two channels
    # nchan_=2
    
    #file_input is "new"
    fs_ = int(fs.value.strip())

    y_data = np.fromfile(Path(f_), dtype = np.float64)
    y_data = y_data.reshape(-1,nchan_)
    y_data = y_data[:,displaychan_]
    
    # 1 channel
    
    y_data = y_data - np.median(y_data,0)

    x_data = np.linspace(0,np.shape(y_data)[0]/fs_,np.shape(y_data)[0])
	
    max_val_slider = len(y_data)/fs_
    data = {'y' : y_data, 'x' : x_data}
    # data = {'y_syn' : y_data[:,synapchan_], 'y_nerve' : ydata[:,nervechan_], 'x' : x_data}
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
	
# def update_plot1_slider(attr,old,new):
# 	start_ = range_slider.value[0]
# 	end_ = range_slider.value[1]
# 	new_selection = select_data([start_,end_])
# 	data_selected.data.update(new_selection.data)
def button_plot_range_callback():
    print('processing range')
    start_ = range_slider.value[0]
    end_ = range_slider.value[1]
    new_selection = select_data([start_,end_])
    data_selected.data.update(new_selection.data)
    print('plot updated')


# create exit button
button_exit = Button(label="Exit", button_type="success",width=100)
button_exit.on_click(button_callback)

# PathToFile = "/Users/kperks/OneDrive - wesleyan.edu/Teaching/Neurophysiology/Data/CockroachSensoryPhysiology/40kHz/RepeatedStimulation2021-08-27T18_37_10.bin"

# filepath = TextInput(title="path to data file",value="PathToFile",width=800)
filepath = TextInput(title="path to data file",value="PathToFile",width=800)

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

# initialize data
data = {'x':[],'y':[]}
src_data = ColumnDataSource(data)
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


#################
# tab 2 overlay trials
#################
# def button_saveas():
# 	#sys.exit()  # Stop the server
# 	print('will save dataframe for overlaid data when add function')
# 	# convert ColumnDataSource to dataframe

# 	# save dataframe as h5

'''
spont 0.5,1,1.5,2,2.5,3,3.5,4,24.5,24
tarsa 6.38,7.972,9.666,11.432,13.024,14.746,16.52,18.38,20.246,22.4
barb 26.725,28.926,30.818,32.649,34.561,36.446,38.366,40.305,42.442,44.325

'''

def update_overlay():
	
	fs_ = int(fs.value.strip())
	filtert = int(0.01*fs_)
	offset_t = 0.5 # 500 msec include a pre-stimulus onset
	windur = float(trial_duration2.value.strip()) # sec

	trial_times_ = trial_times2.value.split(',')
	trial_times_ = [float(t) for t in trial_times_]


	#create a new dictionary to store new data to plot temporarily
	datamat={'ys':[],'xs':[]}
	xtime = np.linspace(-offset_t,windur,int((windur+offset_t)*fs_))
	ys = []
	xs = []
	for i,t in enumerate(trial_times_):
		xs.append(xtime)
		win0 = int((t-offset_t)*fs_)
		win1 = win0+int((windur+offset_t)*fs_)
		y = src_data.data['y'][win0:win1]
		y = y - np.mean(y)
		y = np.abs(y)
		y = ndimage.gaussian_filter(y,filtert)
		ys.append(y)
	datamat['ys'] = ys
	datamat['xs'] = xs

	data_overlay.data = datamat

	if do_average.active:
		if do_average.active[0]==0:
			data_mean.data = {'x':np.mean(np.asarray(xs),0),'y':np.mean(np.asarray(ys),0)}

	if not do_average.active:
		data_mean.data = {'x':[],'y':[]}

	################
	#create a new dictionary to store raw data to get spikes
	###############

# create save button
# button_save = Button(label="Save", button_type="success", width=100)
# button_save.on_click(button_saveas)

# create text input for trial times
trial_times2 = TextInput(title="List of Trial Start times (comma-separated; seconds)", width=800)
# trial_times2.on_change("value",update_overlay)

# check whether to plot overlay or not
labels = ["plot average across trials"]
do_average = CheckboxGroup(labels=labels, active=[0])
# do_average.on_change("active",update_overlay)

# create text input for trial times
trial_duration2 = TextInput(title="Duration of plot window (seconds)",value='1', width=100)
# trial_duration2.on_change("value",update_overlay)

button_update_overlay2 = Button(label="Update Plot", button_type="success",width=100)
button_update_overlay2.on_click(update_overlay)

ymin = TextInput(title="Duration of plot window (seconds)",value='1', width=100)

# hover = HoverTool(tooltips=[('mV', '@y'), ('time', '@x')])
p2 = figure(plot_width=1000, plot_height=500,
				   tools=['hover','pan','box_zoom','wheel_zoom','reset','save'],
				   x_axis_label='time from stimulus onset (seconds)',y_axis_label='amplitude (arbitrary units)')
p2.xaxis.major_label_text_font_size = "18pt"
p2.xaxis.axis_label_text_font_size = "18pt"
p2.yaxis.major_label_text_font_size = "18pt"
p2.yaxis.axis_label_text_font_size = "18pt"
# p2.x_range = Range1d(20, 25)
# p2.y_range = Range1d(-0.1, 0.1)

# get fs_ from text input on tab1
fs_ = int(fs.value.strip())

# hard-coded values for window duration and offset currently also in update function
offset_t = 0.5 # 500 msec include a pre-stimulus onset
windur = float(trial_duration2.value.strip()) # sec

# initialize xtime
xtime = np.linspace(-offset_t,windur,int((windur+offset_t)*fs_))

# initialize data_overlay ColumnDataSource
data_overlay = ColumnDataSource(data = {
	'ys':[np.zeros(int((windur+offset_t)*fs_))],
	'xs':[xtime]
	})

data_mean = ColumnDataSource(data = {
	'y':[np.zeros(int((windur+offset_t)*fs_))],
	'x':[xtime]
	})

# use multiline to plot data
glyph = MultiLine(xs='xs',ys='ys')
p2.add_glyph(data_overlay,glyph)
"""
ADD show average rate
"""
p2.line(x = 'x', y='y',source =data_mean,line_color='red',line_width=4)

#########
# is there a way to have each line a different color?  --yes with a colors list (needs to change with size datamat)
# or when hover on a line it is highlighted? 
#########

# collect controls and layout all elements
controls2 = [trial_times2,do_average,trial_duration2,button_update_overlay2]#,button_save]
inputs2 = column(*controls2)
l2 = column(inputs2,p2)

# create tab
tab2 = Panel(child=l2,title='overlay trials')


################
# tab3 spike counts and raster
#################
# def button_expfit():
# 	#sys.exit()  # Stop the server
# 	print('will do an exponential fit on selected data')

def update_plot3():
	print('calculating average rate and updating plots')
	fs_ = int(fs.value.strip())
	filtert = int(0.01*fs_)
	offset_t = 0.5 # 500 msec include a pre-stimulus onset
	windur = float(trial_duration3.value.strip()) # sec

	trial_times_ = trial_times3.value.split(',') #trial times list text input must not have spaces
	trial_times_ = [float(t) for t in trial_times_]

	spk_thresh_ = float(spk_thresh3.value.strip())

	xtime = np.linspace(-offset_t,windur,int((windur+offset_t)*fs_))
	win0 = int((trial_times_[0]-offset_t)*fs_)
	win1 = win0+int((windur+offset_t)*fs_)
	y = deepcopy(dict(src_data.data)['y'][win0:win1])
	data_trial3.data = {'x' : xtime,'y' : deepcopy(dict(src_data.data)['y'][win0:win1])}
	y[y<=spk_thresh_] = 0
	#find peaks of y
	peaks,props = find_peaks(y,distance = int(fs_*0.0005))
	peak_t = np.asarray([xtime[p] for p in peaks])
	data_spktimes3.data = {'x':peak_t,'y':np.zeros(len(peak_t))}

	#create a new dictionary to store new data to plot temporarily
	datamat={'y':[],'x':[]}
	binsize = float(bin_size3.value.strip())

	spks = []
	for i,t in enumerate(trial_times_):
		win0 = int(t*fs_)
		win1 = win0+int((windur+offset_t)*fs_)
		y = deepcopy(dict(src_data.data)['y'][win0:win1])
		xtime = deepcopy(dict(src_data.data)['x'][win0:win1])-t
		y[y<=spk_thresh_] = 0
		#find peaks of y
		peaks,props = find_peaks(y,distance = int(fs_*0.0005))
		peak_t = np.asarray([xtime[p] for p in peaks])
		spks.extend(peak_t)
	bins = np.arange(0,windur+binsize,binsize)
	h,bin_edges = np.histogram(spks,bins)
	avg_rate_response = h/binsize/len(trial_times_) #(number of spikes per bin divided by duration of bin divided by number of trials)

	#now get hist for pre-stim 
	spks = []
	for i,t in enumerate(trial_times_):
		win0 = int((t-offset_t)*fs_)
		win1 = int(t*fs_)
		y = deepcopy(dict(src_data.data)['y'][win0:win1])
		xtime = deepcopy(dict(src_data.data)['x'][win0:win1])-t
		y[y<=spk_thresh_] = 0
		peaks,props = find_peaks(y,distance = int(fs_*0.0005))
		peak_t = np.asarray([xtime[p] for p in peaks])
		spks.extend(peak_t)
	bins = np.arange(-offset_t,0+binsize,binsize)
	h,bin_edges_base = np.histogram(spks,bins)
	avg_rate_base = h/binsize/len(trial_times_)
	print('n trials = ')
	print(len(trial_times_))
	print('n spks baseline = ')
	print(len(peak_t))
	

	datamat['y'] = np.concatenate([avg_rate_base,avg_rate_response]) #np.asarray(ys)
	# datamat['x'] = bins[0:-1] #np.asarray(xs)
	datamat['x'] = np.concatenate([bin_edges_base[0:-1],bin_edges[0:-1]])

	data_scatter.data = datamat

	ydr3.start= -10
	ydr3.end= np.max(np.concatenate([avg_rate_base,avg_rate_response]))+10
	xdr3.start = -offset_t
	xdr3.end = windur

xdr3 = Range1d(start=-0.5,end=1)
ydr3 = Range1d(start=-10,end=1000)

hover = HoverTool(tooltips=[('mV', '@y'), ('time', '@x')])
# create figure for tab 3
p3 = figure(plot_width=1000, plot_height=500,
				   tools=[hover,'pan','box_zoom','lasso_select','wheel_zoom','reset','save'],
				   y_axis_label='average spike rate per bin', x_axis_label='time from stimulus onset (seconds)',
				   x_range=xdr3,y_range=ydr3)
p3.xaxis.major_label_text_font_size = "18pt"
p3.xaxis.axis_label_text_font_size = "18pt"
p3.yaxis.major_label_text_font_size = "18pt"
p3.yaxis.axis_label_text_font_size = "18pt"

# hover = HoverTool(tooltips=[('mV', '@y'), ('time', '@x')])
p3b = figure(plot_width=1000, plot_height=500,
				   tools=[hover,'pan','box_zoom','wheel_zoom','reset','save'],
				   title = 'Example trial (first trial listed) to show spike detection',
				   x_axis_label = 'seconds')
p3b.xaxis.major_label_text_font_size = "18pt"
p3b.xaxis.axis_label_text_font_size = "18pt"
p3b.yaxis.major_label_text_font_size = "18pt"
p3b.yaxis.axis_label_text_font_size = "18pt"


# create exp fit button
# button_dofit = Button(label="Fit Data", button_type="success", width=100)
# button_dofit.on_click(button_expfit)

# create text input for trial duration
trial_duration3 = TextInput(title="Duration of plot window",value='1', width=100)
# trial_duration3.on_change("value",update_plot3)

# create text input for trial times
trial_times3 = TextInput(title="List of Trial Start times (comma-separated no spaces; seconds)", width=800)
# trial_times3.on_change("value",update_plot3)

# create text input for spike threshold
spk_thresh3 = TextInput(title="Spike Threshold (from examining raw data; seconds)",value='0.04', width=100)
# spk_thresh3.on_change("value",update_plot3)

# create text input for bin size of histogram for spike rate
bin_size3 = TextInput(title="Bin Width to Calculate Spike Rate (seconds)",value='0.01', width=100)
# bin_size3.on_change("value",update_plot3)

button_update_plot3 = Button(label="Update Plot", button_type="success",width=100)
button_update_plot3.on_click(update_plot3)

# hard-coded values for offset currently also in update function
offset_t = 0.5 # 500 msec include a pre-stimulus onset
windur = float(trial_duration3.value.strip()) # sec

# initialize xtime
xtime = np.linspace(-offset_t,windur,int((windur+offset_t)*fs_))

# initialize data_overlay ColumnDataSource
data_scatter = ColumnDataSource(data = {
	'y':np.zeros(int((windur+offset_t)*fs_)),
	'x':xtime
	})

# use multiline to plot data
p3.circle(x='x',y='y',source=data_scatter,color='black')

# initialize data_overlay ColumnDataSource
data_trial3 = ColumnDataSource(data = {
	'y':[np.zeros(int((windur+offset_t)*fs_))],
	'x':[xtime]
	})

data_spktimes3 = ColumnDataSource(data = {
	'y':np.zeros(int((windur+offset_t)*fs_)),
	'x':xtime
	})

# use multiline to plot data
p3b.line(x = 'x', y='y',source =data_trial3,line_color='black')
p3b.circle(x='x',y='y',source=data_spktimes3,color='red',size=6,level='overlay')


# collect controls and layout all elements
controls3 = [trial_times3,trial_duration3,spk_thresh3,bin_size3,button_update_plot3]#,button_dofit]
inputs3 = column(*controls3)
l3 = column(inputs3,p3,p3b)

# create tab
tab3 = Panel(child=l3,title='Spiking Response Histogram')


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
	print('windur =' )
	print(windur)
	offset_t = windur/2 # 500 msec include a pre-stimulus onset
	
	trialdur = float(trial_duration4.value.strip()) # sec
	print('trialdur =')
	print(trialdur)

	trial_times_ = trial_times4.value.split(',') #trial times list text input must not have spaces
	trial_times_ = [float(t) for t in trial_times_]
	print('trial_times_= ')
	print(trial_times_)
	spk_thresh_ = float(spk_thresh4.value.strip())
	print('spk_thresh_ =')
	print(spk_thresh_)

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


# create figure for tab 4
p4 = figure(plot_width=1000, plot_height=500,
				   tools=['hover','pan','box_zoom','wheel_zoom','reset','save'],
				   y_axis_label='V', x_axis_label='time from spike peak (milliseconds)',
				   title='individual spikes')


# create figure 2 for tab 4
p4b = figure(plot_width=1000, plot_height=500,
				   tools=['hover','pan','box_zoom','wheel_zoom','reset','save'],
				   y_axis_label='V', x_axis_label='time from spike peak (milliseconds)',
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

plot_spike4 = Spinner(title="spike number to plot",low=0, width=100)
plot_spike4.on_change("value",update_plot4)

spklist4 = TextInput(title="spike indices to overlay (either 'all' or a comma-separated list of indices", value='all', width=800)
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
p4b.line(x = 'x', y='y',source =data_overlay4_mean,line_color='orange',line_width=6,alpha=0.5)

# collect controls and layout all elements
controls4 = [trial_times4,trial_duration4,spk_thresh4,plot_duration4,plot_spike4,spklist4]
inputs4 = column(*controls4)
l4 = column(inputs4,p4,p4b)

# create tab
tab4 = Panel(child=l4,title='plot spike waveforms')


#######
# create tabs layout
######

tabs = Tabs(tabs=[tab1,tab2,tab3,tab4])

curdoc().add_root(tabs)

