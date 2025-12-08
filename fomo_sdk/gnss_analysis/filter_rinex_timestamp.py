import georinex as gr
from datetime import datetime
import pandas as pd

# Load observation and navigation files
obs_file = "/Users/mbo/Desktop/fomo-data/logs/Reach_6BFD__20241023170621/Reach_6BFD__raw_20241023170621_RINEX_3_03/Reach_6BFD__raw_20241023170621.24O"  # Observation data
nav_file = "/Users/mbo/Desktop/fomo-data/logs/Reach_6BFD__20241023170621/Reach_6BFD__raw_20241023170621_RINEX_3_03/Reach_6BFD__raw_20241023170621.24P"

# Read RINEX observation data and filter by time
obs_data = gr.load(
    obs_file, tlim=["2024-10-23T18:52:30", "2024-10-23T19:14:44"]
)  # .sel(time=slice(start_timestamp, end_timestamp))

# Process or analyze obs_data as needed
print(obs_data)

with open("/Users/mbo/Desktop/filtered_data.O", "w") as f:
    # Write RINEX headers manually (refer to your original file for reference)
    header_lines = """\
    3.03           OBSERVATION DATA    M: Mixed            RINEX VERSION / TYPE
Emlid RGL 1.2.7                         20241023 192543 UTC PGM / RUN BY / DATE
format: u-blox                                              COMMENT
                                                           MARKER NAME
                                                           MARKER NUMBER
                                                           MARKER TYPE
                                                           OBSERVER / AGENCY
                   EMLID REACH RS+                         REC # / TYPE / VERS
                                                           ANT # / TYPE
 1399888.9584 -4099294.8080  4666704.0132                  APPROX POSITION XYZ
       0.0650        0.0000        0.0000                  ANTENNA: DELTA H/E/N
G    4 C1C L1C D1C S1C                                      SYS / # / OBS TYPES
R    4 C1C L1C D1C S1C                                      SYS / # / OBS TYPES
E    4 C1C L1C D1C S1C                                      SYS / # / OBS TYPES
J    4 C1C L1C D1C S1C                                      SYS / # / OBS TYPES
C    4 C1I L1I D1I S1I                                      SYS / # / OBS TYPES
 2024    10    23    17     6   39.9980000     GPS         TIME OF FIRST OBS
 2024    10    23    19    25   59.2030000     GPS         TIME OF LAST OBS
G                                                           SYS / PHASE SHIFT
R                                                           SYS / PHASE SHIFT
E                                                           SYS / PHASE SHIFT
J                                                           SYS / PHASE SHIFT
C                                                           SYS / PHASE SHIFT
 0                                                         GLONASS SLOT / FRQ #
C1C    0.000 C1P    0.000 C2C    0.000 C2P    0.000        GLONASS COD/PHS/BIS
                                                           END OF HEADER
"""
    f.write(header_lines)

    # Loop over filtered data and format each observation
    for time, observation in obs_data.groupby(
        "time"
    ):  # Assuming obs_data has time and observations
        # Convert numpy.datetime64 to a standard datetime object
        time_dt = pd.to_datetime(time).to_pydatetime()

        # Format time as needed for RINEX output
        formatted_time = time_dt.strftime(" %Y %m %d %H %M %S.%f")[:21] + "     GPS\n"
        f.write(formatted_time)

        # Format observations
        for satellite, values in observation.items():
            formatted_obs = (
                f"{satellite} " + " ".join(f"{v:.4f}" for v in values) + "\n"
            )
            f.write(formatted_obs)
