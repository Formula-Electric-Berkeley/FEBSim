import pandas as pd
import numpy as np


class MeshPoint:
    def __init__(self, segment_type, length, radius, partial_distance, track_segment_index):
        self.type = segment_type
        self.length = length
        self.radius = radius
        self.curvature = 1 / radius if radius != 0 else 0
        self.partial_distance = partial_distance
        self.track_segment_index = track_segment_index


class Track:
    def __init__(self, track_file, mesh_thickness=0.25):
        self.details = pd.io.excel.read_excel(track_file, sheet_name="Shape")
        self.details.columns = ["Type", "Length", "Corner Radius"]

        self.details["Partial Distance"] = self.details["Length"].cumsum().shift(1, fill_value=0)
        self.total_length = self.details["Partial Distance"].values[-1]

        self.mesh = self.generate_mesh(mesh_thickness)
        self.mesh_count = len(self.mesh)

        self.apex_indices = self.get_apex_indices(self.mesh)

    def generate_mesh(self, thickness):
        segments = []

        # For each intended mesh point, find the segment that it exists in
        for mesh_point_target in np.linspace(0, self.total_length, int(self.total_length // thickness)):
            active_track_segment = self.details[self.details["Partial Distance"] <= mesh_point_target].iloc[-1]
            mesh_point_segment = MeshPoint(*active_track_segment, active_track_segment.name) # using .name to access the index.

            segments.append(mesh_point_segment)
        
        return segments

    def get_apex_indices(self, mesh):
        meshes_per_track_segment = [0 for i in range(len(self.details))]


        for mesh_point in mesh:
            meshes_per_track_segment[mesh_point.track_segment_index] += 1

        print(meshes_per_track_segment)
        
        # Adjust cumsum so that it's offset by one, kinda
        cumsum = np.concatenate(([0], np.cumsum(meshes_per_track_segment)[:-1]))

        return [0] + [
           cumsum[i] + meshes_per_track_segment[i] // 2 for i in range(len(self.details))
        ]