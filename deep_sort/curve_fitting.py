# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from collections import deque

# Column master element Gauss elimination method
def Guess(a, b):
    m, n = a.shape
    if (m < n):
        print("Infinite Solutions")
    else:
        for i in range(n):
            max_i = np.fabs(a[i:, i]).argmax()
            if i != max_i + i:
                a[[i, max_i], :] = a[[max_i, i], :]
                b[i], b[max_i] = b[max_i], b[i]
            if (a[i][i] == 0):
                break
            for j in range(i+1, n):
                a[j, :] = a[j, :] - a[i, :] * (a[j, i] / a[i, i])
                b[j] = b[j] - b[i] * (a[j, i] / a[i, i])
        x = np.zeros(n)
        x[n-1] = b[n-1] / a[n-1, n-1]
        for k in range(n-2, -1, -1):
            for l in range(k+1, n):
                b[k] -= a[k, l] * x[l]
            x[k] = b[k] / a[k, k]
        return x

def curve_fitting(frame_idx, bbox):
    """Make a cubic polynomial curve fit of x and y, x and p, respectively
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height, confidence)`.
    """
    """surface fitting
    B = np.array([np.ones(5), np.array(x)*np.array(x)]).T
    G = np.array([np.ones(5), np.array(y), np.array(y)*np.array(y)]).T
    U = np.diag(list(p))
    B_inv = np.linalg.inv(np.dot(B.T, B))
    G_inv = np.linalg.inv(np.dot(G.T, G))
    C = np.linalg.multi_dot((B_inv, B.T, U, G, G_inv))
    """
    def __init__(self, seq_info):
        self.frame_idx = seq_info["min_frame_idx"]
        self.last_idx = seq_info["max_frame_idx"]
    while self.frame_idx <= self.last_idx:
        frame_callback(self, self.frame_idx)
        self.frame_idx += 1
        
    def id_storage(tracks, dets, track_indices, detection_indices):
        features = np.array([dets[i].feature for i in detection_indices])
        targets = np.array([tracks[i].track_id for i in track_indices])
        targets_size = np.array([tracks[i].to_tlwh() for i in track_indices])           #add
        
        masking_id = self.metric.overlap_id(features, targets, targets_size)
        
        return cost_matrix
	    
    # before the match, after the update
    def data_storage():
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]])
        for row in results:
            print(row[0], row[1], row[2], row[3], row[4], row[5], row[6],file=f)
        
        return []
        
    def overlap_id(self, features, targets, targets_size):
        cost_dictionary = {}
        size_dictionary = {}
        fitting_id = []
        cost_matrix = self.distance(features, targets)
        for i, target in enumerate(targets):
            cost_dictionary[target] = cost_matrix[i, :].argmin()
            size_dictionary[target] = targets_size[i]

        cost_order = sorted(list(cost_dictionary.values()))
        for i in range(len(targets)-1):
            if cost_order[i] == cost_order[i+1]:
                overlap_id = [list(cost_dictionary.keys())[j] for j, x in enumerate(list(cost_dictionary.values())) if x == cost_order[i]]
                iou_calculate = [size_dictionary[list(cost_dictionary.keys())[j]] for j, x in enumerate(list(cost_dictionary.values())) if x == cost_order[i]]
                max_confidence_id = max(zip(np.array(iou_calculate)[:, 4], overlap_id))[1]

                Objects_IOU = self.couple_iou(iou_calculate[0], iou_calculate[1])
                return Objects_IOU
        return 0
                """if Objects_IOU > 0.2:
                    fitting_id.append(list([overlap_id, max_confidence_id]))
                    # print(overlap_id)
                    # print(iou_calculate)
                    # print(Objects_IOU, max_confidence_id)
                    # input("---")
                    return fitting_id
        return fitting_id"""
    
    # add the data storage
    Database = []
    next_database = []
    def isIndatabase(next_data, Database):
	    for idx, data in enumerate(Database):
		    if next_data["ID"] == data[0]["ID"]:
			    return (True, idx)
	    return (False, 0)
    
    frame_idx = seq_info["min_frame_idx"]
    last_idx = seq_info["max_frame_idx"]
    
    while frame_idx <= last_idx:
        next_database.append({"frame":frame_idx, "ID":track.track_id, "box":bbox})

        if Database == []:
            for next_data in next_database:
	            deq = deque(maxlen=5)
	            deq.append(next_data)
	            Database.append(deq)
        if Database != []:
            for next_data in next_database:
	            isIn, idx = isIndatabase(next_data, Database)
	            if isIn:
		            Database[idx].append(next_data)
	            else:
		            deq = deque(maxlen=5)
		            deq.append(next_data)
		            Database.append(deq)
        frame_idx += 1
    
    # Polynomial Curve Fitting && Multiple Straight-Line Regression Algorithm
    x = np.array(bbox[:, 0])
    y = np.array(bbox[:, 1])
    w = np.array(bbox[:, 2])
    h = np.array(bbox[:, 3])
    p = np.array(bbox[:, 4])
    
    fitting_x = np.polyfit(frame_idx, x, 3)
    frame_x = np.poly1d(fitting_x)
    fitting_y = np.polyfit(frame_idx, y, 3)
    frame_y = np.poly1d(fitting_y)
    """fitting_w = np.polyfit(frame_idx, w, 3)
    frame_w = np.poly1d(fitting_w)
    fitting_h = np.polyfit(frame_idx, h, 3)
    frame_h = np.poly1d(fitting_h)
    fitting_p = np.polyfit(frame_idx, p, 3)
    frame_p = np.poly1d(fitting_p)
    """
    next_bbox = [frame_x(next_frame), frame_y(next_frame)]
    
