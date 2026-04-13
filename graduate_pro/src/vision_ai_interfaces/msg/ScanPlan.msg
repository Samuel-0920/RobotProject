string strategy              # single_point or multi_point
float64 scan_height         # scan height in mm
float64 required_height     # calculated required height in mm
Waypoint[] waypoints        # waypoint list
geometry_msgs/Point[] scan_region  # scan region points
float64 object_height       # object height in mm
string mode                 # mode from request
float64 overlap_ratio