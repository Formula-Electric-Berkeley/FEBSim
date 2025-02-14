import QTrack
import QVehicle

# Parameters to edit
track_file = "Michigan_2022_AutoX.xlsx"
vehicle_file = "FEB_SN3_30kW.xlsx"

# Filepath constants
TRACK_FILES_FOLDER = "./track_files/"
VEHICLE_FILES_FOLDER = "./vehicle_files/"

track = QTrack.Track(TRACK_FILES_FOLDER + track_file, mesh_thickness=0.2)
vehicle = QVehicle.Vehicle.from_file(VEHICLE_FILES_FOLDER + vehicle_file)


mesh_speeds = [[] for i in range(track.mesh_count)]
