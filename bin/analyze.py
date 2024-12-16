import multiprocessing as mp
import pickle
import sys
from argparse import ArgumentParser
import numpy as np
import scipy.stats as stats
import mdtraj as md
import analysis
from analysis.frame import Frame
import copy as cp

def analyze_all(frame):

    results_per_frame = {}
    # Setting how many peaks we require based upon leaflets in the system. Since we are finding peaks and leaflet positions based on the midpoints of the COM locations, this changes the number of peaks we require.
    if frame.n_leaflets == 6:
        needed_peaks = 3
    elif frame.n_leaflets == 4:
        needed_peaks = 2
    elif frame.n_leaflets == 2:
        needed_peaks = 2

    # Unpack inputs
    frame.validate_frame()

    # Calculates directors for a given set of residues
    tail_info = analysis.utils.calc_all_directors(frame.xyz,
                                                  frame.masses,
                                                  frame.residuelist)

    directors = tail_info["directors"]
    coms = tail_info["coms"].squeeze()

    # Get the locations of each leaflet based on the
    # z_coord of the COMs
    z = coms[:,2]
    peaks,hist = analysis.height.calc_peaks(
                z, [np.min(z), np.max(z)],
                n_layers=frame.n_leaflets,
                prominence=0,
                distance=50,
                threshold=[0, frame.n_leaflets])

    if len(peaks) == needed_peaks:
        if len(peaks) == 2:

"bin/analyze.py" 304L, 11721B
