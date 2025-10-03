from metadrive.scenario.scenario_description import MetaDriveType


OBJECT_TYPE = {
    MetaDriveType.UNSET: 0,
    MetaDriveType.VEHICLE: 1,
    MetaDriveType.PEDESTRIAN: 2,
    MetaDriveType.CYCLIST: 3,
    MetaDriveType.OTHER: 4,
}

LANE_POLYLINES = [1, 2, 3]
ROADLINE_POLYLINES = [6, 7, 8, 9, 10, 11, 12, 13]
BOUNDARY_POLYLINES = [15, 16]
STOP_SIGN_POLYLINE = [17]
CROSSWALK_POLYLINE = [18, 19]

POLYLINE_TYPE = {
    # for lane
    MetaDriveType.LANE_FREEWAY: 1,
    MetaDriveType.LANE_SURFACE_STREET: 2,
    "LANE_SURFACE_UNSTRUCTURE": 2,
    MetaDriveType.LANE_BIKE_LANE: 3,
    # for roadline
    MetaDriveType.LINE_BROKEN_SINGLE_WHITE: 6,
    MetaDriveType.LINE_SOLID_SINGLE_WHITE: 7,
    "ROAD_EDGE_SIDEWALK": 7,
    MetaDriveType.LINE_SOLID_DOUBLE_WHITE: 8,
    MetaDriveType.LINE_BROKEN_SINGLE_YELLOW: 9,
    MetaDriveType.LINE_BROKEN_DOUBLE_YELLOW: 10,
    MetaDriveType.LINE_SOLID_SINGLE_YELLOW: 11,
    MetaDriveType.LINE_SOLID_DOUBLE_YELLOW: 12,
    MetaDriveType.LINE_PASSING_DOUBLE_YELLOW: 13,
    # for roadedge
    MetaDriveType.BOUNDARY_LINE: 15,
    MetaDriveType.BOUNDARY_MEDIAN: 16,
    # for stopsign
    MetaDriveType.STOP_SIGN: 17,
    # for crosswalk
    MetaDriveType.CROSSWALK: 18,
    # for speed bump
    MetaDriveType.SPEED_BUMP: 19,
}

TRAFFIC_LIGHT_STATE = {
    None: 0,
    MetaDriveType.LANE_STATE_UNKNOWN: 0,
    # // States for traffic signals with arrows.
    MetaDriveType.LANE_STATE_ARROW_STOP: 1,
    MetaDriveType.LANE_STATE_ARROW_CAUTION: 2,
    MetaDriveType.LANE_STATE_ARROW_GO: 3,
    # // Standard round traffic signals.
    MetaDriveType.LANE_STATE_STOP: 4,
    MetaDriveType.LANE_STATE_CAUTION: 5,
    MetaDriveType.LANE_STATE_GO: 6,
    # // Flashing light signals.
    MetaDriveType.LANE_STATE_FLASHING_STOP: 7,
    MetaDriveType.LANE_STATE_FLASHING_CAUTION: 8,
}
