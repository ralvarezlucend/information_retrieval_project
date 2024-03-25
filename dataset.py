import pandas as pd
from pathlib import Path
import hdf5_getters


# Download the Million Song Subset: http://labrosa.ee.columbia.edu/~dpwe/tmp/millionsongsubset.tar.gz

# Add the path where the data was downloaded
path_to_data = ""

# careful with \
directory = Path("r" + path_to_data + "\millionsongsubset\MillionSongSubset")
h5_files = [path for path in directory.glob('**/*.h5') if path.is_file()]

data = []

# can choose other attributes
for h5_path in h5_files:
    with hdf5_getters.open_h5_file_read(h5_path) as h5:
        song_id = hdf5_getters.get_song_id(h5).decode('utf-8')
        tempo = hdf5_getters.get_tempo(h5)
        loudness = hdf5_getters.get_loudness(h5)
        artist_familiarity = hdf5_getters.get_artist_familiarity(h5)
        song_hotnesss = hdf5_getters.get_song_hotttnesss(h5)
        data.append({
            "song_id": song_id,
            "tempo": tempo,
            "loudness": loudness,
            "artist_familiarity": artist_familiarity,
            "song_hotttnesss": song_hotnesss
        })

df_h5 = pd.DataFrame(data)
df_h5.to_csv('data/10000_songs.csv', index=False)
