from pathlib import Path
import hdf5_getters

directory = Path("MillionSongSubset")
h5_files = [path for path in directory.glob('**/*.h5') if path.is_file()]

h5 = hdf5_getters.open_h5_file_read(h5_files[0])
duration = hdf5_getters.get_duration(h5)
print(duration)
h5.close()