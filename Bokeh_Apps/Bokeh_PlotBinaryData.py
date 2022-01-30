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
    x_data = np.linspace(0,len(y_data)/fs_,len(y_data))

    max_val_slider = len(y_data)/fs_

    data = {'y' : y_data,'x' : x_data}
    new_data = ColumnDataSource(data = data)
    src_data.data.update(new_data.data)
    
    range_slider.update(end=max_val_slider)

    start_ = range_slider.value[0]
    stop_ = range_slider.value[1]
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

# create hover tool
hover = HoverTool(tooltips=[('V', '@y'), ('time', '@x')])

# create figure
p = figure(plot_width=1000, plot_height=500,
                   tools=[hover,'pan','box_zoom','wheel_zoom','reset','save'],
                   x_axis_label = 'seconds',y_axis_label='Volts')

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

#################
# tab 2 plot Vm by distance
#################


def update_scatter(attr,old,new):

    distance_ = distance.value.split(',')
    distance_ = [float(d) for d in distance_]

    Vm_ = Vm.value.split(',')
    Vm_ = [float(d) for d in Vm_]

    datamat={'y':Vm_,'x': distance_}

    data_scatter2.data = datamat


# create save button
# button_save = Button(label="Save", button_type="success", width=100)
# button_save.on_click(button_saveas)

# create text input for trial times
distance = TextInput(title="List: distance from Voltage source to electrode (nodes)",value='0', width=800)
distance.on_change("value",update_scatter)

# create text input for trial times
Vm = TextInput(title="List: voltage recorded", value='0', width=800)
Vm.on_change("value",update_scatter)

# hover = HoverTool(tooltips=[('mV', '@y'), ('time', '@x')])
p2 = figure(plot_width=1000, plot_height=500,
                   tools=['pan','lasso_select','box_zoom','wheel_zoom','reset','save'],
                   x_axis_label='distance from voltage source to electrode',y_axis_label='Voltage measured')

# initialize data_overlay ColumnDataSource
data_scatter2 = ColumnDataSource(data = {
    'y':[0],
    'x':[0]
    })

circ = p2.circle(x = 'x', y='y',source =data_scatter2,color='black',size=20)
p2.line(x = 'x', y='y',source =data_scatter2,color='gray')

hover_tool = HoverTool(tooltips=[
            ('Amplitude', '@y'),
            ('Distance', '@x'),
        ], renderers=[circ],mode='vline')
p2.tools.append(hover_tool)

#########
# is there a way to have each line a different color?  --yes with a colors list (needs to change with size datamat)
# or when hover on a line it is highlighted? 
#########

# collect controls and layout all elements
controls2 = [distance,Vm]#,button_save]
inputs2 = column(*controls2)
l2 = column(inputs2,p2)

# create tab
tab2 = Panel(child=l2,title='Scatter Plot Results')

#######
# create tabs layout
######

tabs = Tabs(tabs=[tab1,tab2])

curdoc().add_root(tabs)
