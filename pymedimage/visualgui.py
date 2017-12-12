"""
Visualization of 3d numpy array for debugging and visualization
Based on code from internet
https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric-data#gs.v0bGR6M
"""
# TODO: Add orientation selection and automatic interpolation
import numpy as np
import matplotlib.pyplot as plt
from .rttypes import BaseVolume

def multi_slice_viewer(volume,_slice = None, cmap='viridis'):
    if isinstance(volume, BaseVolume):
        volume = volume.data
    if volume.ndim < 3: volume = np.expand_dims(volume, axis=0)

    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    if _slice is not None:
        _slice = int(_slice)
        if (_slice >= 0) and (_slice < volume.shape[0]):
            ax.index = int(_slice)
        else:
            raise ValueError("Invalid slice indexing %s out of %s"%(_slice,volume.shape[0]))
    else:
        ax.index = volume.shape[0] // 2
    ax.set_xlabel("Slice %s of %s"%(str(ax.index+1),volume.shape[0]))
    ax.imshow(volume[ax.index], cmap=cmap)
    fig.canvas.mpl_connect('key_press_event', process_key)
    fig.canvas.mpl_connect('scroll_event', process_scroll)
    plt.show()

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key in ['down', 'left', 'j']:
        previous_slice(ax)
    elif event.key in ['up', 'right', 'k']:
        next_slice(ax)
    fig.canvas.draw()

def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.set_xlabel("Slice %s of %s"%(str(ax.index+1),volume.shape[0]))
    ax.images[0].set_array(volume[ax.index])

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.set_xlabel("Slice %s of %s"%(str(ax.index+1),volume.shape[0]))
    ax.images[0].set_array(volume[ax.index])

def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

def process_scroll(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.button == 'up':
        previous_slice(ax)
    elif event.button == 'down':
        next_slice(ax)
    fig.canvas.draw()

