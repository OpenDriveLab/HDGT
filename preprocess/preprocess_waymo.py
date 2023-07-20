## Copyright 2022 Xiaosong Jia. All Rights Reserved.
## Check https://github.com/waymo-research/waymo-open-dataset/blob/656f759070a7b1356f9f0403b17cd85323e0626c/src/waymo_open_dataset/protos/map.proto and https://github.com/waymo-research/waymo-open-dataset/blob/656f759070a7b1356f9f0403b17cd85323e0626c/src/waymo_open_dataset/protos/scenario.proto for details about the data structure and data type
from typing_extensions import final
import os
os.sys.path.append('.')
import map_pb2
import scenario_pb2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pickle
from posixpath import basename
import numpy as np
import tensorflow as tf
from google.protobuf import text_format
import sys
import gc
import math
import shutil

## To obtain the index for interpolate missed frame of inputs
def get_all_break_point(total_lis, sub_lis):
    i = 0
    j = 0
    last_conti_index = -1
    break_point_i_lis = []
    break_point_j_lis = []
    while j < len(sub_lis):
        while total_lis[i] != sub_lis[j]:
            i += 1
        if last_conti_index == -1:
            last_conti_index = i
        elif i == last_conti_index + 1:
            last_conti_index += 1
        else:
            break_point_i_lis.append((last_conti_index, i))
            break_point_j_lis.append((j-1, j))
            last_conti_index = i
        j += 1
    return break_point_i_lis, break_point_j_lis

import scipy.interpolate as interp
## Interpolate polylines to the target number of points
def interpolate_polyline(polyline, num_points):
    if np.allclose(polyline[0], polyline[1]):
        return polyline[0][np.newaxis, :].repeat(num_points, axis=0)
    tck, u = interp.splprep(polyline.T, s=0, k=1)
    u = np.linspace(0.0, 1.0, num_points)
    return np.column_stack(interp.splev(u, tck))

def euclid(label, pred):
    return np.sqrt((label[..., 0]-pred[...,0])**2 + (label[...,1]-pred[...,1])**2)



## For interpolation
def transfer_poly_to_rectangle(poly_coor, yaw):
    cos_theta = np.cos(-yaw)
    sin_theta = np.sin(-yaw)
    poly_coor[..., 0], poly_coor[..., 1] = poly_coor[..., 0]*cos_theta - poly_coor[..., 1]*sin_theta, poly_coor[..., 1]*cos_theta + poly_coor[..., 0]*sin_theta
    p = Polygon(poly_coor)
    xmin, ymin, xmax, ymax = p.bounds
    poly_coor = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])
    cos_theta = np.cos(yaw)
    sin_theta = np.sin(yaw)
    poly_coor[..., 0], poly_coor[..., 1] = poly_coor[..., 0]*cos_theta - poly_coor[..., 1]*sin_theta, poly_coor[..., 1]*cos_theta + poly_coor[..., 0]*sin_theta
    return poly_coor

num_of_scene_per_folder = 84 ## parallel preprocessing
current_scene_index = 0 if len(sys.argv) == 1 else int(sys.argv[1])
parent_path = os.path.join(os.path.dirname(os.getcwd()), "dataset", "waymo")

data_version = "hdgt_waymo_dev_tmp" if len(sys.argv) == 1 else sys.argv[2]

if current_scene_index == 12: ## Validation Set
    data_split_type = "validation"
    saving_folder = os.path.join(parent_path, data_split_type, data_version+str(current_scene_index))

else: 
    data_split_type = "training"
    saving_folder = os.path.join(parent_path, data_split_type, data_version+str(current_scene_index))
if os.path.exists(saving_folder):
    shutil.rmtree(saving_folder)
os.mkdir(saving_folder)
print("Scene Index:", current_scene_index, "Data Split:", data_split_type, "Start!!!")
is_val = ("validation" == data_split_type)

case_cnt = 0
num_of_element_lis = [] ## The number of elements (agents + map) in each scene; For balanced batching; Otherwise, the GPU memory usage might vary a lot

fdir = os.listdir(os.path.join(parent_path, data_split_type)) ## The directory contains all tfrecord file
fdir.sort()
for fname in fdir:
    # if case_cnt > 5:
    #     break
    if "tfrecord" in fname:
        record_index = int(fname.split("-")[1])
        if not is_val and (record_index < current_scene_index * num_of_scene_per_folder or record_index >= (current_scene_index+1) * num_of_scene_per_folder): ## Not for this worker
            continue
        raw_dataset = tf.data.TFRecordDataset([os.path.join(parent_path, data_split_type, fname)])
        print("new file", fname,  flush=True)
        for raw_record_index, raw_record in enumerate(raw_dataset):
            proto_string = raw_record.numpy()
            proto = scenario_pb2.Scenario()
            proto.ParseFromString(proto_string)
            ## Agent Feature
            now_num_time_step = len(proto.timestamps_seconds)
            now_all_tracks = proto.tracks
            total_num_of_agent = len(now_all_tracks)
            
            ### Debug
            # if case_cnt > 5:
            #     break
            
            now_predict_agent_index = []
            now_difficulty = []
            for predict_track_info in proto.tracks_to_predict:
                now_predict_agent_index.append(predict_track_info.track_index)
                now_difficulty.append(predict_track_info.difficulty)
            
            now_difficulty = np.array(now_difficulty)
            now_other_agent_index = [_ for _ in range(total_num_of_agent) if _ not in now_predict_agent_index]
            all_object_index = now_predict_agent_index + now_other_agent_index


            all_predict_agent_feature = []
            all_label = []
            all_auxiliary_label = []
            all_label_mask = []
            all_predict_agent_type = []
            new_predict_index = [] ## After filering and reordering agents

            all_other_agent_type = []
            all_other_agent_feature = []
            all_other_label = []
            all_other_label_mask = []
            new_other_index = []

            now_index2id = {}
            pred_num = 0
            now_scene_id = proto.scenario_id
            for now_track_index_cnt, now_track_index in enumerate(all_object_index):
                now_track = now_all_tracks[now_track_index]
                now_mask = np.array([1] * 11)
                now_index2id[now_track_index] = now_track.id
                now_fea = []
                obs_end_index = None
                for timestep, timestep_state in enumerate(now_track.states):
                    if timestep_state.valid:
                        now_fea.append(np.array([int(timestep), float(timestep_state.center_x), float(timestep_state.center_y), float(timestep_state.center_z), float(timestep_state.velocity_x), float(timestep_state.velocity_y), float(timestep_state.heading), float(timestep_state.length), float(timestep_state.width), float(timestep_state.height),]))
                        if timestep <= 10:
                            obs_end_index = len(now_fea)
                now_fea = np.stack(now_fea, axis=0) #Num_Observed_Timestep, C: #T, x, y, z, vx, vy, psi, length, width, height
                ## The index of first timestep > 10 -> Not in the observation interval [0, 10] - 1.1 second
                if now_fea[0][0] > 10:
                    continue
                now_obs_fea = now_fea[:obs_end_index] #
                ## Not fully observable - interpolate with constant acceleratation assumption
                if now_obs_fea.shape[0] < 11:
                    start_step = int(now_obs_fea[0, 0])
                    end_step = int(now_obs_fea[-1, 0])
                    padded_fea = now_obs_fea[:, 1:7] ## We only need to interpolate x, y, z, vx, vy, psi

                    if now_obs_fea.shape[0] != end_step-start_step+1:
                        break_point_i_lis, break_point_j_lis = get_all_break_point(list(range(0, 11)), now_obs_fea[:, 0])
                        line_lis = [now_obs_fea[:break_point_j_lis[0][0]+1, 1:7]]
                        for bi in range(len(break_point_i_lis)):
                            line = interp.interp1d(x=[0.0, 1.0], y=now_obs_fea[break_point_j_lis[bi][0]:break_point_j_lis[bi][1]+1, 1:7], assume_sorted=True, axis=0)(np.linspace(0.0, 1.0, break_point_i_lis[bi][1]-break_point_i_lis[bi][0]+1))
                            v_xy = line[1:, 3:5]
                            cumsum_step = np.cumsum(v_xy/10.0, axis=0) + line[0, 0:2][np.newaxis, :]
                            line[1:, 0:2] = cumsum_step
                            now_mask[break_point_i_lis[bi][0]+1:break_point_i_lis[bi][1]] = 0
                            line_lis.append(line[1:-1, :])
                            if bi == len(break_point_i_lis) - 1:
                                line_lis.append(now_obs_fea[break_point_j_lis[bi][1]:, 1:7])
                            else:
                                line_lis.append(now_obs_fea[break_point_j_lis[bi][1]:break_point_j_lis[bi+1][0]+1, 1:7])
                        padded_fea = np.concatenate(line_lis, axis=0)
                    if start_step > 0:
                        v_xy = padded_fea[0, 3:5]
                        cumsum_step = np.cumsum((-v_xy/10.0)[np.newaxis, :].repeat(start_step, axis=0), axis=0)[::-1, :] + padded_fea[0, 0:2][np.newaxis, :]
                        padded_fea = np.concatenate([padded_fea[0][np.newaxis,:].repeat(start_step, axis=0), padded_fea], axis=0)
                        padded_fea[:start_step, :2] = cumsum_step
                        now_mask[:start_step] = 0
                    if end_step < 10:
                        v_xy = padded_fea[-1, 3:5]
                        cumsum_step = np.cumsum((v_xy/10.0)[np.newaxis, :].repeat(10-end_step, axis=0), axis=0) + padded_fea[-1, 0:2][np.newaxis, :]
                        padded_fea = np.concatenate([padded_fea, padded_fea[-1][np.newaxis,:].repeat(10-end_step, axis=0)], axis=0)
                        padded_fea[end_step+1:, :2] = cumsum_step
                        now_mask[end_step+1:] = 0
                    now_obs_fea_padded = np.concatenate([padded_fea, np.array([now_obs_fea[:, -3].mean()]*11)[:, np.newaxis], np.array([now_obs_fea[:, -2].mean()]*11)[:, np.newaxis], np.array([now_obs_fea[:, -1].mean()]*11)[:, np.newaxis], now_mask[:, np.newaxis]], axis=-1)
                else:
                    now_obs_fea_padded = np.concatenate([now_obs_fea[:, 1:],  now_mask[:, np.newaxis]], axis=-1)
                
                
                now_future_fea = now_fea[obs_end_index:]
                if now_future_fea.shape[0] != 80:
                    now_label_mask = np.array([0] * 80)
                    tmp_label = np.zeros((1, 80, 2))
                    tmp_auxiliary_label = np.zeros((1, 80, 3))
                    for label_time_index_i in range(now_future_fea.shape[0]):
                        label_time_index_in_lis = int(now_future_fea[label_time_index_i, 0]) - 11
                        now_label_mask[label_time_index_in_lis] = 1
                        tmp_label[0, label_time_index_in_lis, :] = now_future_fea[label_time_index_i, [1, 2]]
                        tmp_auxiliary_label[0, label_time_index_in_lis, :] = now_future_fea[label_time_index_i, [4, 5, 6]]
                else:
                    now_label_mask = np.array([1] * 80)
                    tmp_label = now_future_fea[:, 1:3][np.newaxis, :, :]
                    tmp_auxiliary_label = now_future_fea[:, 4:7][np.newaxis, :, :]
                
                if now_track_index_cnt >= len(now_predict_agent_index) or now_future_fea.shape[0] == 0:
                    ## Other Label
                    all_other_agent_feature.append(now_obs_fea_padded)
                    all_other_label.append(tmp_label)
                    all_other_label_mask.append(now_label_mask)
                    new_other_index.append(now_track_index)
                    all_other_agent_type.append(now_track.object_type)
                else:
                    pred_num += 1
                    all_predict_agent_feature.append(now_obs_fea_padded)
                    all_label.append(tmp_label)
                    all_auxiliary_label.append(tmp_auxiliary_label)
                    all_label_mask.append(now_label_mask)
                    new_predict_index.append(now_track_index)
                    all_predict_agent_type.append(now_track.object_type)

            all_object_id = [now_index2id[_] for _ in new_predict_index] + [now_index2id[_] for _ in new_other_index]
            all_agent_type = np.array(all_predict_agent_type + all_other_agent_type)
            all_agent_feature = np.stack(all_predict_agent_feature+all_other_agent_feature, axis=0) ## Num_agent, T_observed, C: x, y, z, vx, vy, heading, length, width, height, mask
        
            #all_agent_obs_final_v = np.sqrt(all_input_data[:, -1, 3]**2+all_input_data[:, -1, 4]**2)
            all_agent_map_size = np.ones(all_agent_feature.shape[0]) * 999.0 ## During preprocessing, we simply keep all map elements
            #all_agent_obs_final_v * 8.0 + np.vectorize(map_size_lis.__getitem__)(all_agent_type)
            all_dynamic_map_fea_dic = {}
            for time_step in range(11):
                for map_element_index in range(len(proto.dynamic_map_states[time_step].lane_states)):
                    now_tuple = (float(proto.dynamic_map_states[time_step].lane_states[map_element_index].stop_point.x), float(proto.dynamic_map_states[time_step].lane_states[map_element_index].stop_point.y), float(proto.dynamic_map_states[time_step].lane_states[map_element_index].stop_point.z), proto.dynamic_map_states[time_step].lane_states[map_element_index].lane)
                    if now_tuple not in all_dynamic_map_fea_dic:
                        all_dynamic_map_fea_dic[now_tuple] = [0] * 11 ## 0 represents unknown
                    all_dynamic_map_fea_dic[now_tuple][time_step] = proto.dynamic_map_states[time_step].lane_states[map_element_index].state
            all_unkown_traffic = [0] * 11
            traffic_light_info_to_remove = []
            for k, v in all_dynamic_map_fea_dic.items():
                if v == all_unkown_traffic:
                    traffic_light_info_to_remove.append(k)
            for k in traffic_light_info_to_remove:
                all_dynamic_map_fea_dic.pop(k)
            all_dynamic_map_fea = {int(k[3]):np.array([k[0], k[1], k[2]] + v)  for k, v in all_dynamic_map_fea_dic.items()}

            #id: (type, polygon)  type:0,1
            all_polygon_fea = []
            #id: (lane_id_lis, [x,y])
            all_stopsign_fea = {}
            #id: lane_info_dic
            all_lane_fea = {}
            all_road_edge = []
            all_road_line = []
            for now_map_fea in proto.map_features:
                if now_map_fea.HasField("crosswalk"):
                    now_polygon_fea = [0, np.array([[_.x, _.y, _.z]for _ in now_map_fea.crosswalk.polygon])]
                    all_polygon_fea.append(now_polygon_fea)
                if now_map_fea.HasField("speed_bump"):
                    now_polygon_fea = [1, np.array([[_.x, _.y, _.z]for _ in now_map_fea.speed_bump.polygon])]
                    all_polygon_fea.append(now_polygon_fea)
                if now_map_fea.HasField("stop_sign"):
                    all_stopsign_fea[int(now_map_fea.id)] = [list(now_map_fea.stop_sign.lane), [now_map_fea.stop_sign.position.x, now_map_fea.stop_sign.position.y, now_map_fea.stop_sign.position.z]]
                if now_map_fea.HasField("lane"):
                    all_lane_fea[int(now_map_fea.id)] = {}
                    all_lane_fea[int(now_map_fea.id)]["speed_limit"] = now_map_fea.lane.speed_limit_mph
                    all_lane_fea[int(now_map_fea.id)]["type"] = now_map_fea.lane.type #5 types
                    all_lane_fea[int(now_map_fea.id)]["xyz"] = np.array([[_.x, _.y, _.z]for _ in now_map_fea.lane.polyline])
                    ## A list of IDs for lanes that this lane may be entered from.
                    all_lane_fea[int(now_map_fea.id)]["entry"] = list(now_map_fea.lane.entry_lanes)
                    ## A list of IDs for lanes that this lane may exit to.
                    all_lane_fea[int(now_map_fea.id)]["exit"] = list(now_map_fea.lane.exit_lanes)
                    all_lane_fea[int(now_map_fea.id)]["left"] = []
                    for left_neighbor in now_map_fea.lane.left_neighbors:
                        boundary_type_lis = [int(_.boundary_type)  for _ in left_neighbor.boundaries]
                        ## For simplicity, we use the first appeared boundary type as the type
                        if 1 in boundary_type_lis:
                            boundary_type = 1
                        elif 2 in boundary_type_lis:
                            boundary_type = 2
                        elif 3 in boundary_type_lis:
                            boundary_type = 3
                        else:
                            boundary_type = 0
                        ##  ID -> Neighbor ID, self_start, self_end, neighbor_start, neighbor_end, type (4)
                        all_lane_fea[int(now_map_fea.id)]["left"].append([left_neighbor.feature_id, left_neighbor.self_start_index, left_neighbor.self_end_index, left_neighbor.neighbor_start_index, left_neighbor.neighbor_end_index, boundary_type])
                    all_lane_fea[int(now_map_fea.id)]["right"] = []
                    for right_neighbor in now_map_fea.lane.right_neighbors:
                        boundary_type_lis = [int(_.boundary_type)  for _ in right_neighbor.boundaries]
                        if 1 in boundary_type_lis:
                            boundary_type = 1
                        elif 2 in boundary_type_lis:
                            boundary_type = 2
                        elif 3 in boundary_type_lis:
                            boundary_type = 3
                        else:
                            boundary_type = 0
                        ##ID -> Neighbor ID, self_start, self_end, neighbor_start, neighbor_end, type
                        all_lane_fea[int(now_map_fea.id)]["right"].append([right_neighbor.feature_id, right_neighbor.self_start_index, right_neighbor.self_end_index, right_neighbor.neighbor_start_index, right_neighbor.neighbor_end_index, boundary_type])
                if now_map_fea.HasField("road_edge"):
                    road_edge_xy = np.array([[_.x, _.y, _.z]for _ in now_map_fea.road_edge.polyline])
                    if road_edge_xy.shape[0] > 2:
                        all_polygon_fea.append([now_map_fea.road_edge.type+2, road_edge_xy])
                if now_map_fea.HasField("road_line"):
                    road_line_xy = np.array([[_.x, _.y, _.z]for _ in now_map_fea.road_line.polyline])
                    if road_line_xy.shape[0] > 2:
                        all_polygon_fea.append([now_map_fea.road_line.type+2+3, road_line_xy]) ## 14 types
            
            ## Split Long Lane and interpolate to the same number of points (20) per polyline (20m). Then, we need to update the up/front/left/right relations of the splitted lanes
            length_per_polyline = 40.0 # 20 meters
            point_per_polyline = 21
            space = int(length_per_polyline // (point_per_polyline-1))

            new_lane_fea = []
            old_lane_id_to_new_lane_index_lis = {}

            for old_lane_id, old_lane_info in all_lane_fea.items():
                if old_lane_info["xyz"].shape[0] <= length_per_polyline:
                    old_lane_id_to_new_lane_index_lis[old_lane_id] = [len(new_lane_fea)]
                    new_lane_xy = old_lane_info["xyz"]
                    if new_lane_xy.shape[0] > 1:
                        new_lane_xy = interpolate_polyline(new_lane_xy, point_per_polyline)
                    else:
                        new_lane_xy = np.broadcast_to(new_lane_xy, (point_per_polyline, 3))
                    new_lane_fea.append({"xyz":new_lane_xy, "speed_limit":old_lane_info["speed_limit"], "type":old_lane_info["type"], "left":[], "right":[], "prev":[], "follow":[]   , "stop":[], "signal":[]})
                else:
                    num_of_new_lane = math.ceil(old_lane_info["xyz"].shape[0]/length_per_polyline)
                    now_lanelet_new_index_lis = list(range(len(new_lane_fea), len(new_lane_fea)+num_of_new_lane))
                    old_lane_id_to_new_lane_index_lis[old_lane_id] = now_lanelet_new_index_lis
                    new_lane_xy = []
                    for _ in range(num_of_new_lane-1):
                        tmp = old_lane_info["xyz"][int(_*length_per_polyline):int(_*length_per_polyline+length_per_polyline+1)]
                        new_lane_xy.append(tmp[::space, :])
                    tmp =  old_lane_info["xyz"][int((num_of_new_lane-1)*length_per_polyline):]
                    if tmp.shape[0] == 1:
                        tmp = np.concatenate([old_lane_info["xyz"][int((num_of_new_lane-1)*length_per_polyline-1)][np.newaxis, :], tmp], axis=0)
                    new_lane_xy.append(interpolate_polyline(tmp, point_per_polyline))
                    #tmp = tmp[::2, :]
                    for _ in range(len(new_lane_xy)):
                        new_lane_fea.append({"xyz":new_lane_xy[_], "speed_limit":old_lane_info["speed_limit"], "type":old_lane_info["type"], "left":[], "right":[], "prev":[], "follow":[], "stop":[], "signal":[]})

            ## Update relations
            for old_lane_id, new_lane_lis in old_lane_id_to_new_lane_index_lis.items():
                if len(new_lane_lis) > 0:
                    for j in range(1, len(new_lane_lis)):
                        prev_index = new_lane_lis[j-1]
                        next_index = new_lane_lis[j]
                        new_lane_fea[prev_index]["follow"].append([next_index, 0])
                        new_lane_fea[next_index]["prev"].append([prev_index, 1])
                ## Follow
                tmp_index = new_lane_lis[-1]
                for old_adj_index in  all_lane_fea[old_lane_id]["exit"]:
                    new_lane_fea[tmp_index]["follow"].append([old_lane_id_to_new_lane_index_lis[old_adj_index][0], 0])

                ## Prev
                tmp_index = new_lane_lis[0]
                for old_adj_index in all_lane_fea[old_lane_id]["entry"]:
                    new_lane_fea[tmp_index]["prev"].append([old_lane_id_to_new_lane_index_lis[old_adj_index][-1], 1])
                
                ## Left Right
                for edge_type in ["left", "right"]:
                    old_adj_info_lis = all_lane_fea[old_lane_id][edge_type]
                    ## ID, self_start, end, neighbor_start, end, type
                    for old_adj_info in old_adj_info_lis:
                        can_turn_new_lane_lis = new_lane_lis[int(old_adj_info[1]//length_per_polyline):int(old_adj_info[2]//length_per_polyline+1)]
                        can_turn_new_adj_lane_lis = old_lane_id_to_new_lane_index_lis[old_adj_info[0]][int(old_adj_info[3]//length_per_polyline):int(old_adj_info[4]//length_per_polyline+1)]
                        l1 = len(can_turn_new_lane_lis)
                        l2 = len(can_turn_new_adj_lane_lis)
                        boundary_type = old_adj_info[5]
                        if l1 == l2:
                            for tmp_index_i in range(l1):
                                tmp_index = can_turn_new_lane_lis[tmp_index_i]
                                new_lane_fea[tmp_index][edge_type].append([can_turn_new_adj_lane_lis[tmp_index_i], boundary_type+2])
                        elif l1 < l2:
                            ratio = int(math.ceil(float(l2)/float(l1)))
                            for tmp_index_i in range(l1):
                                tmp_index = can_turn_new_lane_lis[tmp_index_i]
                                ratio_index = 0
                                gap = ratio - 1
                                if l2%l1 == 0:
                                    gap += 1
                                while ratio_index < ratio and ratio_index + tmp_index_i * gap < l2:
                                    new_lane_fea[tmp_index][edge_type].append([can_turn_new_adj_lane_lis[int(ratio_index + tmp_index_i * gap)], boundary_type+2])
                                    ratio_index += 1                    
                        elif l1 > l2:
                            ratio = int(math.ceil(float(l1)/float(l2)))
                            for adj_index_i in range(l2):
                                tmp_adj_index = can_turn_new_adj_lane_lis[adj_index_i]
                                ratio_index = 0
                                gap = ratio - 1
                                if l1%l2 == 0:
                                    gap += 1
                                while ratio_index < ratio and ratio_index + adj_index_i * gap < l1:
                                    tmp_index = can_turn_new_lane_lis[ratio_index + adj_index_i * gap]
                                    new_lane_fea[tmp_index][edge_type].append([tmp_adj_index, boundary_type+2])
                                    ratio_index += 1
            for stop_sign_id in all_stopsign_fea:
                new_relate_to_stop_sign_id_lis = []
                for _ in all_stopsign_fea[stop_sign_id][0]:
                    new_relate_to_stop_sign_id_lis += old_lane_id_to_new_lane_index_lis[_]
                for _ in new_relate_to_stop_sign_id_lis:
                    new_lane_fea[_]["stop"].append(all_stopsign_fea[stop_sign_id][1])
            for old_lane_id in all_dynamic_map_fea:
                new_lane_id_lis = old_lane_id_to_new_lane_index_lis[old_lane_id]
                for _ in new_lane_id_lis:
                    new_lane_fea[_]["signal"].append(all_dynamic_map_fea[old_lane_id])
            for _ in range(len(new_lane_fea)):
                new_lane_fea[_]["yaw"] = np.arctan2(new_lane_fea[_]["xyz"][-1, 1]-new_lane_fea[_]["xyz"][0, 1], new_lane_fea[_]["xyz"][-1, 0]-new_lane_fea[_]["xyz"][0, 0])

            ##Split and Regularize Polygon fea
            ##20m per piece, 20 point
            new_polygon_fea = []
            for polygon_index in range(len(all_polygon_fea)):
                if all_polygon_fea[polygon_index][0] not in [0, 1]:
                    if len(all_polygon_fea[polygon_index][1]) > length_per_polyline:
                        num_of_piece = int(len(all_polygon_fea[polygon_index][1]) // length_per_polyline + 1)
                        length_per_piece = len(all_polygon_fea[polygon_index][1])//num_of_piece + 1
                        for _ in range(num_of_piece):
                            polygon_coor_of_current_piece = all_polygon_fea[polygon_index][1][int(_*length_per_piece):int((_+1)*length_per_piece)]
                            if polygon_coor_of_current_piece.shape[0] > 1:
                                new_polygon_fea.append([all_polygon_fea[polygon_index][0], polygon_coor_of_current_piece])
                    else:
                        if all_polygon_fea[polygon_index][1].shape[0] > 1:
                            new_polygon_fea.append(all_polygon_fea[polygon_index])
                else:
                    new_polygon_fea.append([all_polygon_fea[polygon_index][0], np.concatenate([all_polygon_fea[polygon_index][1], all_polygon_fea[polygon_index][1][0, :][np.newaxis, :]], axis=0)])
            
            all_polygon_fea = [[_[0], interpolate_polyline(_[1], point_per_polyline)] for _ in new_polygon_fea]

            num_of_agent = all_agent_feature.shape[0]
            # ## Split Too Much Agent
            new_dist_between_agent_lane = (euclid(all_agent_feature[:, -1, :2][:, np.newaxis, np.newaxis, :], np.stack([_["xyz"] for _ in new_lane_fea])[np.newaxis, :, :, :]).min(2) < all_agent_map_size[:, np.newaxis])
            if len(all_polygon_fea) > 0:
                new_dist_between_agent_polygon = (euclid(all_agent_feature[:, -1, [0,1]][:, np.newaxis, np.newaxis, :], np.stack([_[1] for _ in all_polygon_fea], axis=0)[np.newaxis, :, :, :]).min(2) < all_agent_map_size[:, np.newaxis])

        
            lane_new_index_to_final_index = {}
            for agent_index_i in range(new_dist_between_agent_lane.shape[0]):
                nearby_lane_new_index_lis = np.where(new_dist_between_agent_lane[agent_index_i, :])[0].tolist()
                for nearby_lane_new_index in nearby_lane_new_index_lis:
                    if nearby_lane_new_index not in lane_new_index_to_final_index:
                        lane_new_index_to_final_index[nearby_lane_new_index] = len(lane_new_index_to_final_index)
            final_lane_fea = [{} for _ in range(len(lane_new_index_to_final_index))]
            for lane_new_index in lane_new_index_to_final_index:
                for transfer_key in ["xyz", "speed_limit", "type", "stop", "signal", "yaw"]:
                    final_lane_fea[lane_new_index_to_final_index[lane_new_index]][transfer_key] = new_lane_fea[lane_new_index][transfer_key]
                for transfer_key in ["left", "right", "prev", "follow"]:
                    final_lane_fea[lane_new_index_to_final_index[lane_new_index]][transfer_key] = [[lane_new_index_to_final_index[_[0]], _[1]] for _ in new_lane_fea[lane_new_index][transfer_key] if _[0] in lane_new_index_to_final_index]
            polygon_new_index_to_final_index = {}
            if len(all_polygon_fea) > 0:
                for agent_index_i in range(new_dist_between_agent_polygon.shape[0]):
                    nearby_polygon_lis = np.where(new_dist_between_agent_polygon[agent_index_i, :])[0].tolist()
                    for nearby_polygon_new_index in nearby_polygon_lis:
                        if nearby_polygon_new_index not in polygon_new_index_to_final_index:
                            polygon_new_index_to_final_index[nearby_polygon_new_index] = len(polygon_new_index_to_final_index)
            final_polygon_fea = [[] for _ in range(len(polygon_new_index_to_final_index))]
            for polygon_new_index in polygon_new_index_to_final_index:
                final_polygon_fea[polygon_new_index_to_final_index[polygon_new_index]] = all_polygon_fea[polygon_new_index]
            
            ## Visualization
            # # os.environ['KMP_DUPLICATE_LIB_OK']= "True"
            # # from matplotlib import pyplot as plt
            # # plt.gca().axis('equal')
            # # plt.cla()
            # # plt.clf()

            # # for agent_index in range(len(all_agent_feature)):
            # #     plt.plot(all_agent_feature[agent_index, :, 0], all_agent_feature[agent_index, :, 1], "blue", zorder=20)
            # # for lane_index in range(len(final_lane_fea)):
            # #     plt.plot(final_lane_fea[lane_index]["xyz"][:, 0], final_lane_fea[lane_index]["xyz"][:, 1], "black")
            # # for polygon_index in range(len(final_polygon_fea)):
            # #     plt.plot(final_polygon_fea[polygon_index][1][:, 0], final_polygon_fea[polygon_index][1][:, 1], "black")
            
            # # ### Change this to increase resolution
            # # plt.xlim(950, 1000)
            # # plt.ylim(-2150, -2200)
            # # visualization_key = "left"
            # # for lane_index in range(len(final_lane_fea)):
            # #     if len(final_lane_fea[lane_index][visualization_key]) != 0:
            # #         for neighbor_lane_info in final_lane_fea[lane_index][visualization_key]:
            # #             plt.plot(final_lane_fea[lane_index]["xyz"][point_per_polyline//2, 0], final_lane_fea[lane_index]["xyz"][point_per_polyline//2, 1], final_lane_fea[neighbor_lane_info[0]]["xyz"][point_per_polyline//2, 0], final_lane_fea[neighbor_lane_info[0]]["xyz"][point_per_polyline//2, 1], marker="o", c="red")
            # # import ipdb
            # # ipdb.set_trace()
            # # plt.savefig("tmp.png")

            all_data = {}
            all_data["fname"] = fname
            all_data["agent_feature"] = all_agent_feature
            all_data["label"] = np.concatenate(all_label, axis=0)
            all_data["auxiliary_label"] = np.concatenate(all_auxiliary_label, axis=0)
            all_data["label_mask"] = np.stack(all_label_mask, axis=0)
            all_data["difficulty"] = now_difficulty

            all_data["pred_num"] = pred_num
            if len(all_other_label) != 0:
                all_other_label = np.concatenate(all_other_label, axis=0)
                all_other_label_mask = np.stack(all_other_label_mask, axis=0)
            all_data["other_label"] = all_other_label
            all_data["other_label_mask"] = all_other_label_mask
            all_data["obejct_id_lis"] = np.array(all_object_id)
            all_data["scene_id"] = now_scene_id
            all_data["agent_type"] = all_agent_type
            all_data["map_fea"] = [final_lane_fea, final_polygon_fea]

            with open(os.path.join(saving_folder, data_version + "_case"+str(case_cnt)+".pkl"), "wb") as g:
                pickle.dump(all_data, g)


            num_of_element_lis.append(all_agent_feature.shape[0]+len(final_lane_fea)+len(final_polygon_fea))
            del all_data
            gc.collect()
            if case_cnt % 10000 == 0:
                print(data_split_type, case_cnt, "done", flush=True)
            case_cnt += 1

with open(os.path.join(saving_folder, data_version+"_number_of_case.pkl"), "wb") as g:
    pickle.dump(np.array(num_of_element_lis), g)

print("Scene Index:", current_scene_index, "Data Split:", data_split_type, "Done!!!")

