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
            # Here the midpoint between the 2 peaks shows the leaflet position.
            midpoints = (peaks[:-1] + peaks[1:]) *0.5
            all_leaflet_points = [-np.inf] + list(midpoints) + [np.inf]
            all_leaflet_points = np.sort(all_leaflet_points)
            leaflet_ranges = [(all_leaflet_points[i], all_leaflet_points[i+1])
                          for i in range(len(all_leaflet_points)-1)]

        # Essentially, the midpoints here refer to the leaflet boundary in a bilayer while the midpoints are the middle bilayer boundaries. We do not find the values of the top most and bottom most layer and instead use inf
        # We just have to make sure that the kwargs (especially distance) used to calculate peaks are well defined otherwise we might get really close together.
        elif found_leaflets == 6:
            midpoints = (peaks[:-1] + peaks[1:]) *0.5
            all_leaflet_points = [-np.inf] + list(midpoints)  + list(peaks) + [np.inf]
            all_leaflet_points = np.sort(all_leaflet_points)
            leaflet_ranges = [(all_leaflet_points[i], all_leaflet_points[i+1])
                          for i in range(len(all_leaflet_points)-1)]

        # Get the tilt angle and nematic order for each leaflet
        tilt = []
        s2 = []
        apt = []
        for lmin, lmax in leaflet_ranges:
            mask = np.logical_and(coms[:,2] > lmin,
                                  coms[:,2] < lmax)
            leaflet_directors = directors[mask]
            leaflet_tilt = analysis.utils.calc_tilt_angle(leaflet_directors)
            leaflet_apt = (frame.unitcell_lengths[0] * frame.unitcell_lengths[1] /
                           np.sum(mask))
            leaflet_s2 = analysis.utils.calc_order_parameter(leaflet_directors)
            tilt.append(np.mean(leaflet_tilt))  # Taking an average of tilt values per leaflet since it will be easier to process later
            s2.append(leaflet_s2)
            apt.append(leaflet_apt)

        # Calculate Area per Lipid: cross section / n_lipids
        apl = (frame.unitcell_lengths[0] * frame.unitcell_lengths[1] /
                len(frame.residuelist) * frame.n_leaflets)

        # Calculate the height -- uses the "head" atoms specified below
        if frame.cg:
            atomselection = "mhead2 oh1 oh2 oh3 oh4 oh5 amide chead head"
            atomselection = atomselection.split(' ')
            atoms = frame.select(names=atomselection)
        else:
            atomselection = [13.0, 100.0]
            atoms = frame.select(mass_range=atomselection)
            
        height = analysis.height.calc_height(frame, atoms)

        results_per_frame = {'tilt' :  np.array(tilt),
                    's2' : s2,
                    'apl' : apl,
                    'apt' : np.array(apt),
                    'height' : np.array(height)}
        print(results_per_frame)
    return results_per_frame


def calculate_averages_from_results(results):
    """
    Calculate averages from results loaded from the pickle file.

    Parameters:
    -----------
    results : list
        List of results dictionaries for each frame.

    Returns:
    --------
    averages : dict
        Dictionary of averaged properties across all frames.
    """
    # Initialize accumulators
    tilt_all = []
    s2_all = []
    apt_all = []
    apl_all = []
    height_all = []


    for frame_results in results:
        if not frame_results:
            continue  # Skip frames with invalid data
            tilt_all.append(frame_results.get('tilt', []))  # Default to empty if missing
        s2_all.append(frame_results.get('s2', []))
        apt_all.append(frame_results.get('apt', []))
        apl_all.append(frame_results.get('apl', np.nan))  # Default to NaN for scalar
        height_all.append(frame_results.get('height', []))

    if tilt_all:
        # Safely compute max_len only if tilt_all is non-empty
        max_len = max(len(x) for x in tilt_all if len(x) > 0)

        tilt_all = [list(x) + [np.nan] * (max_len - len(x)) if len(x) > 0 else [np.nan] * max_len for x in tilt_all]
        s2_all = [list(x) + [np.nan] * (max_len - len(x)) if len(x) > 0 else [np.nan] * max_len for x in s2_all]
        apt_all = [list(x) + [np.nan] * (max_len - len(x)) if len(x) > 0 else [np.nan] * max_len for x in apt_all]

    else:
        # Handle case where tilt_all is empty
        max_len = 0
        tilt_all = []
        s2_all = []
        apt_all = []

    # Calculate averages
    avg_tilt = [np.nanmean(tilt) for tilt in zip(*tilt_all)] if tilt_all else []
    error_tilt = [np.nanstd(tilt) for tilt in zip(*tilt_all)] if tilt_all else []

    avg_s2 = [np.nanmean(s2) for s2 in zip(*s2_all)] if s2_all else []
    error_s2 = [np.nanstd(s2) for s2 in zip(*s2_all)] if s2_all else []

    avg_apt = [np.nanmean(apt) for apt in zip(*apt_all)] if apt_all else []
    error_apt = [np.nanstd(apt) for apt in zip(*apt_all)] if apt_all else []

    avg_apl = np.nanmean(apl_all) if apl_all else np.nan
    error_apl = np.nanstd(apl_all) if apl_all else np.nan

    avg_height = np.nanmean(height_all, axis=0) if height_all else np.array([])
    error_height = np.nanstd(height_all, axis=0) if height_all else np.array([])


    return {
        "avg_tilt": avg_tilt,
        "err_tilt": error_tilt,
        "avg_s2": avg_s2,
        "err_s2": error_s2,
        "avg_apt": avg_apt,
        "err_apt": error_apt,
        "avg_apl": avg_apl,
        "err_apl": error_apl,
         "avg_height": avg_height,
        "err_height": error_height,
    }


def main():
    ## PARSING INPUTS
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", action="store", type=str,
                        default="")
    parser.add_argument("-c", "--conf", action="store", type=str)
    parser.add_argument("-o", "--output", action="store",
                        type=str, default="./")
    parser.add_argument("-n", "--nleaflets", action="store",
                        type=int, default=2)
    parser.add_argument("--cg", action="store_true", default=False)
    parser.add_argument("--reload", action="store_true", default=False)
    parser.add_argument("--min", action="store", type=float,
                        default=None)
    parser.add_argument("--max", action="store", type=float,
                        default=None)
    #pass a specific selection to parse from the trajectory
    #by default this will select everything not water
    parser.add_argument("-s", "--select", action="store", type=str,
                        default="all_lipids")

    options = parser.parse_args()

    trajfile = options.file
    topfile  = options.conf
    outputdir = options.output
    n_leaflets = options.nleaflets
    cg = options.cg
    reload_traj = options.reload
    z_min = options.min
    z_max = options.max
    selection_string  = options.select


    ## LOADING TRAJECTORIES
    # If cached traj exists:
    try:
        if reload_traj:
            raise ValueError("Ignoring cached trajectory (--reload)")
        frames = analysis.load.load_from_pickle(
            '{}/frames.p'.format(outputdir))

    # If previous traj isn't there load the files inputted via
    # command line
    except:
        traj = analysis.load.load_from_trajectory(
                    trajfile, topfile)

        # Get masses from hoomdxml
        if cg:
            masses = analysis.load.load_masses(cg, topfile=topfile)
        else:
            masses = analysis.load.load_masses(cg, topology=traj.top)
        print('Loaded masses')
# needs some work as that is not specific enough for me
        if selection_string == 'all_lipids':
            # keep only the lipids
            sel_atoms = traj.top.select("(not name water) and " +
                                        "(not resname tip3p " +
                                        "HOH SOL) and not resname DCF and not resname dcfh and not resname NA and not resname DEA")
        else:
            print('Parsing trajectory based on selection: ', selection_string)
            sel_atoms = traj.top.select("( " + selection_string + ")")

        # very simple check to ensure that the selection actually
        # contains some atoms atom_slice will fail if the array
        # size is 0, this will at least provide some more specific feedback
        # as to why
        if len(sel_atoms) == 0:
            raise ValueError("Error: selection does not include any atoms.")

        traj.atom_slice(sel_atoms, inplace=True)
        masses = masses.take(sel_atoms)

        # Load system information
        traj = analysis.load.get_standard_topology(traj, cg)

        # Extract atoms within a specified z range
        if (z_min or z_max):
            traj, masses = analysis.load.extract_range(
                            traj, masses, cg, z_min, z_max)

        # Convert to Frame/residuelist format
        residuelist = analysis.load.to_residuelist(traj.top, cg)
        residuelist = cp.deepcopy(residuelist)
        atomnames = [atom.name for atom in traj.top.atoms]
        frames = []
        for i in range(traj.n_frames):
            frame = Frame(xyz=np.squeeze(traj.xyz[i,:,:]),
                    unitcell_lengths=np.squeeze(
                            traj.unitcell_lengths[i,:]),
                    masses=masses, residuelist=residuelist,
                    atomnames=atomnames, n_leaflets=n_leaflets,
                    cg=cg)
            frames.append([cp.deepcopy(frame)])
        print('Created frame list')

        # Purge the old trajectory from memory
        del traj

        # Save a cached version of the frames list
        with open('{}/frames.p'.format(outputdir), 'wb') as f:
            pickle.dump(frames, f)

    # Get number of frames
    n_frames = len(frames)
    print('Loaded trajectory with {} frames'.format(n_frames))

    # Get parallel processes
    print('Starting {} parallel threads'.format(mp.cpu_count()))
    pool = mp.Pool(mp.cpu_count())
    chunksize = int(len(frames)/mp.cpu_count()) + 1
    results = pool.starmap(analyze_all, frames, chunksize=chunksize)

    # Dump pickle file of results
    with open('{}/results.p'.format(outputdir), 'wb') as f:
        pickle.dump(results, f)
    print('Finished!')

    averages = calculate_averages_from_results(results)
    print("Final Averages:")
    print(averages)


if __name__ == "__main__": main()



