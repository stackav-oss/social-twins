from collections import defaultdict

import numpy as np
from metadrive.scenario.scenario_description import MetaDriveType
from omegaconf import DictConfig
from scenarionet.common_utils import read_dataset_summary, read_scenario

from scenetokens.datasets.base_dataset import BaseDataset
from scenetokens.datasets.scenarionet_types import (
    BOUNDARY_POLYLINES,
    CROSSWALK_POLYLINE,
    LANE_POLYLINES,
    OBJECT_TYPE,
    POLYLINE_TYPE,
    ROADLINE_POLYLINES,
    STOP_SIGN_POLYLINE,
)
from scenetokens.utils import data_utils


default_value = 0
object_type = defaultdict(lambda: default_value, OBJECT_TYPE)
polyline_type = defaultdict(lambda: default_value, POLYLINE_TYPE)


class ScenarioNetDataset(BaseDataset):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)

    def get_dataset_summary(self, data_path: str) -> tuple[dict, dict]:
        _, summary_list, mapping = read_dataset_summary(data_path)
        return summary_list, mapping

    def read_scenario(self, data_path: str, mapping: dict, file_name: str) -> dict:
        return read_scenario(data_path, mapping, file_name)

    def repack_scenario(self, scenario: dict) -> dict:
        # Repack track information
        tracks = scenario["tracks"]
        track_infos = self.repack_tracks(tracks)
        scenario["metadata"]["ts"] = scenario["metadata"]["ts"][: self.total_steps]

        # Repack map information
        map_features = scenario["map_features"]
        map_infos = self.repack_map(map_features)

        # Repack dynamic map information
        dynamic_map_features = scenario["dynamic_map_states"]
        dynamic_map_infos = self.repack_dynamic_map(dynamic_map_features)

        preprocessed_scenario = {
            "track_infos": track_infos,
            "dynamic_map_infos": dynamic_map_infos,
            "map_infos": map_infos,
        }
        preprocessed_scenario.update(scenario["metadata"])
        preprocessed_scenario["timestamps_seconds"] = preprocessed_scenario.pop("ts")
        preprocessed_scenario["current_time_index"] = self.past_len - 1
        preprocessed_scenario["sdc_track_index"] = track_infos["object_id"].index(preprocessed_scenario["sdc_id"])

        tracks_to_predict = self.select_tracks_to_predict(preprocessed_scenario)
        preprocessed_scenario["tracks_to_predict"] = tracks_to_predict
        preprocessed_scenario["map_center"] = scenario["metadata"].get("map_center", np.zeros(3))[np.newaxis]
        preprocessed_scenario["track_length"] = self.total_steps
        return preprocessed_scenario

    def repack_tracks(self, track_features: dict) -> dict:
        start_frame = self.starting_frame
        end_frame = start_frame + self.total_steps
        trajectory_sample_interval = self.config.get("trajectory_sample_interval", 1)
        frequency_mask = data_utils.generate_mask(self.past_len - 1, self.total_steps, trajectory_sample_interval)

        track_infos = {
            "object_id": [],  # {0: unset, 1: vehicle, 2: pedestrian, 3: cyclist, 4: others}
            "object_type": [],
            "trajs": [],
        }

        for object_id, object_state in track_features.items():
            state = object_state["state"]
            for key, value in state.items():
                if len(value.shape) == 1:
                    state[key] = np.expand_dims(value, axis=-1)
            all_state = [
                state["position"],
                state["length"],
                state["width"],
                state["height"],
                state["heading"],
                state["velocity"],
                state["valid"],
            ]
            # type, x,y,z,l,w,h,heading,vx,vy,valid
            all_state = np.concatenate(all_state, axis=-1)
            # all_state = all_state[::sample_inverval]
            if all_state.shape[0] < end_frame:
                all_state = np.pad(all_state, ((end_frame - all_state.shape[0], 0), (0, 0)))
            all_state = all_state[start_frame:end_frame]

            assert all_state.shape[0] == self.total_steps, f"Error: {all_state.shape[0]} != {self.total_steps}"

            track_infos["object_id"].append(object_id)
            track_infos["object_type"].append(object_type[object_state["type"]])
            track_infos["trajs"].append(all_state)

        track_infos["trajs"] = np.stack(track_infos["trajs"], axis=0)
        # scenario['metadata']['ts'] = scenario['metadata']['ts'][::sample_inverval]
        track_infos["trajs"][..., -1] *= frequency_mask[np.newaxis]
        return track_infos

    def repack_lanes(self, map_state: dict) -> tuple[np.ndarray, dict]:
        lane_info = {}
        lane_info["speed_limit_mph"] = map_state.get("speed_limit_mph")
        lane_info["interpolating"] = map_state.get("interpolating")
        lane_info["entry_lanes"] = map_state.get("entry_lanes")
        try:
            lane_info["left_boundary"] = [
                {
                    "start_index": x["self_start_index"],
                    "end_index": x["self_end_index"],
                    "feature_id": x["feature_id"],
                    "boundary_type": "UNKNOWN",  # roadline type
                }
                for x in map_state["left_neighbor"]
            ]
            lane_info["right_boundary"] = [
                {
                    "start_index": x["self_start_index"],
                    "end_index": x["self_end_index"],
                    "feature_id": x["feature_id"],
                    "boundary_type": "UNKNOWN",  # roadline type
                }
                for x in map_state["right_neighbor"]
            ]
        except:
            lane_info["left_boundary"] = []
            lane_info["right_boundary"] = []
        polyline = map_state["polyline"]
        polyline = data_utils.interpolate_polyline(polyline)
        return polyline, lane_info

    def repack_polyline(self, map_state: dict, polyline_type: int) -> tuple[dict, np.ndarray, str]:
        polyline_info = {}
        polyline_info_type = map_state["type"]
        if polyline_type in LANE_POLYLINES:
            polyline_key = "lane"
            polyline, polyline_info = self.repack_lanes(map_state)
        elif polyline_type in ROADLINE_POLYLINES:
            polyline_key = "road_line"
            try:
                polyline = map_state["polyline"]
            except:
                polyline = map_state["polygon"]
            polyline = data_utils.interpolate_polyline(polyline)
        elif polyline_type in BOUNDARY_POLYLINES:
            polyline_key = "road_line"
            polyline = map_state["polyline"]
            polyline = data_utils.interpolate_polyline(polyline)
            polyline_info_type = 7
        elif polyline_type in STOP_SIGN_POLYLINE:
            polyline_key = "stop_sign"
            polyline_info["lane_ids"] = map_state["lane"]
            polyline_info["position"] = map_state["position"]
            polyline = map_state["position"][np.newaxis]
        elif polyline_type in CROSSWALK_POLYLINE:
            polyline_key = "crosswalk"
            polyline = map_state["polygon"]
        polyline_info["type"] = polyline_info_type
        return polyline, polyline_key, polyline_info

    def repack_map(self, map_features: dict) -> dict[str, list]:
        map_infos = {
            "lane": [],
            "road_line": [],
            "road_edge": [],
            "stop_sign": [],
            "crosswalk": [],
            "speed_bump": [],
        }
        polylines = []
        point_cnt = 0
        for k, v in map_features.items():
            polyline_type_ = polyline_type[v["type"]]
            if polyline_type_ == 0:
                continue

            cur_info = {"id": k}
            cur_info["type"] = v["type"]
            if polyline_type_ in LANE_POLYLINES:
                cur_info["speed_limit_mph"] = v.get("speed_limit_mph", None)
                cur_info["interpolating"] = v.get("interpolating", None)
                cur_info["entry_lanes"] = v.get("entry_lanes", None)
                try:
                    cur_info["left_boundary"] = [
                        {
                            "start_index": x["self_start_index"],
                            "end_index": x["self_end_index"],
                            "feature_id": x["feature_id"],
                            "boundary_type": "UNKNOWN",  # roadline type
                        }
                        for x in v["left_neighbor"]
                    ]
                    cur_info["right_boundary"] = [
                        {
                            "start_index": x["self_start_index"],
                            "end_index": x["self_end_index"],
                            "feature_id": x["feature_id"],
                            "boundary_type": "UNKNOWN",  # roadline type
                        }
                        for x in v["right_neighbor"]
                    ]
                except:
                    cur_info["left_boundary"] = []
                    cur_info["right_boundary"] = []
                polyline = v["polyline"]
                polyline = data_utils.interpolate_polyline(polyline)
                map_infos["lane"].append(cur_info)
            elif polyline_type_ in ROADLINE_POLYLINES:
                try:
                    polyline = v["polyline"]
                except:
                    polyline = v["polygon"]
                polyline = data_utils.interpolate_polyline(polyline)
                map_infos["road_line"].append(cur_info)
            elif polyline_type_ in BOUNDARY_POLYLINES:
                polyline = v["polyline"]
                polyline = data_utils.interpolate_polyline(polyline)
                cur_info["type"] = 7
                map_infos["road_line"].append(cur_info)
            elif polyline_type_ in STOP_SIGN_POLYLINE:
                cur_info["lane_ids"] = v["lane"]
                cur_info["position"] = v["position"]
                map_infos["stop_sign"].append(cur_info)
                polyline = v["position"][np.newaxis]
            elif polyline_type_ in CROSSWALK_POLYLINE:
                map_infos["crosswalk"].append(cur_info)
                polyline = v["polygon"]
            if polyline.shape[-1] == 2:  # noqa: PLR2004
                polyline = np.concatenate((polyline, np.zeros((polyline.shape[0], 1))), axis=-1)
            try:
                cur_polyline_dir = data_utils.get_polyline_dir(polyline)
                type_array = np.zeros([polyline.shape[0], 1])
                type_array[:] = polyline_type_
                cur_polyline = np.concatenate((polyline, cur_polyline_dir, type_array), axis=-1)
            except:
                cur_polyline = np.zeros((0, 7), dtype=np.float32)
            polylines.append(cur_polyline)
            cur_info["polyline_index"] = (point_cnt, point_cnt + len(cur_polyline))
            point_cnt += len(cur_polyline)

        try:
            polylines = np.concatenate(polylines, axis=0).astype(np.float32)
        except:
            polylines = np.zeros((0, 7), dtype=np.float32)
        map_infos["all_polylines"] = polylines
        return map_infos

    def repack_dynamic_map(self, dynamic_map: dict) -> dict:
        dynamic_map_infos = {"lane_id": [], "state": [], "stop_point": []}
        for dynamic_map_state in dynamic_map.values():  # (num_timestamp)
            lane_id, state, stop_point = [], [], []
            for cur_signal in dynamic_map_state["state"]["object_state"]:  # (num_observed_signals)
                lane_id.append(str(dynamic_map_state["lane"]))
                state.append(cur_signal)
                if isinstance(dynamic_map_state["stop_point"], list):
                    stop_point.append(dynamic_map_state["stop_point"])
                else:
                    stop_point.append(dynamic_map_state["stop_point"].tolist())
            lane_id = lane_id[: self.total_steps]
            state = state[: self.total_steps]
            stop_point = stop_point[: self.total_steps]
            dynamic_map_infos["lane_id"].append(np.array([lane_id]))
            dynamic_map_infos["state"].append(np.array([state]))
            dynamic_map_infos["stop_point"].append(np.array([stop_point]))
        return dynamic_map_infos

    def select_tracks_to_predict(self, scenario: dict) -> dict:
        track_infos = scenario["track_infos"]
        if self.config.only_train_on_ego:
            tracks_to_predict = {
                "track_index": [scenario["sdc_track_index"]],
                "difficulty": [0],
                "object_type": [MetaDriveType.VEHICLE],
            }
        else:
            sample_list = list(scenario["tracks_to_predict"].keys())  # + ret.get('objects_of_interest', [])
            sample_list = list(set(sample_list))
            tracks_to_predict = {
                "track_index": [
                    track_infos["object_id"].index(object_id)
                    for object_id in sample_list
                    if object_id in track_infos["object_id"]
                ],
                "object_type": [
                    track_infos["object_type"][track_infos["object_id"].index(object_id)]
                    for object_id in sample_list
                    if object_id in track_infos["object_id"]
                ],
            }
        return tracks_to_predict

    def get_agents_of_interest(
        self,
        track_index_to_predict: np.ndarray,
        obj_trajs_full: np.ndarray,
        current_time_index: int,
        obj_types: np.ndarray,
        scene_id: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        center_objects_list = []
        track_index_to_predict_selected = []
        selected_type = self.config.object_type
        selected_type = [object_type[x] for x in selected_type]
        for k in range(len(track_index_to_predict)):
            obj_idx = track_index_to_predict[k]
            if obj_trajs_full[obj_idx, current_time_index, -1] == 0:
                print(f"Warning: obj_idx={obj_idx} is not valid at time step {current_time_index}, scene_id={scene_id}")
                continue
            if obj_types[obj_idx] not in selected_type:
                continue

            center_objects_list.append(obj_trajs_full[obj_idx, current_time_index])
            track_index_to_predict_selected.append(obj_idx)
        if len(center_objects_list) == 0:
            print(f"Warning: no center objects at time step {current_time_index}, scene_id={scene_id}")
            return None, []
        center_objects = np.stack(center_objects_list, axis=0)  # (num_center_objects, num_attrs)
        track_index_to_predict = np.array(track_index_to_predict_selected)
        return center_objects, track_index_to_predict
