# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from collections import deque
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
import warnings

class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=40, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self.Database = []
        self.fit_dictionary = []
        self.fitting_object = []
        self.disappearing_box = deque(maxlen=0)

        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections, frame_idx):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections, frame_idx)

        next_database = []                                                           #add
        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
            next_database.append({"frame":frame_idx, 
                "ID":self.tracks[track_idx].track_id, "box":self.tracks[track_idx].to_tlwh()})
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
        # add the data storage within 5 frames
        def isIndatabase(next_data, Database):
	        for idx, data in enumerate(Database):
		        if next_data["ID"] == data[0]["ID"]:
			        return (True, idx)
	        return (False, 0)
            
        if self.Database == []:
            for next_data in next_database:
	            deq = deque(maxlen=5)
	            deq.append(next_data)
	            self.Database.append(deq)
        else:
            for next_data in next_database:
	            isIn, idx = isIndatabase(next_data, self.Database)
	            if isIn:
		            self.Database[idx].append(next_data)
	            else:
		            deq = deque(maxlen=5)
		            deq.append(next_data)
		            self.Database.append(deq)

        if self.fitting_object != []:
            self.disappearing_box = deque(maxlen=len(self.fitting_object))
            for fit_obj in self.fitting_object:
                fitting_data = [data for data in self.Database if data[0]["ID"] == fit_obj]
                if len(fitting_data[0]) == 5:
                    warnings.filterwarnings("ignore")
                    self.disappearing_box.append([fit_obj, self.curve_fitting(frame_idx, fitting_data)])

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []

        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []

        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)
            
    def curve_fitting(self, now_frame, fitting_data):
        for i in range(len(fitting_data[0])):
            frame_idx = np.append([], fitting_data[0][i]["frame"])
            x = np.append([], fitting_data[0][i]["box"][0])
            y = np.append([], fitting_data[0][i]["box"][1])
            w = np.append([], fitting_data[0][i]["box"][2])
            h = np.append([], fitting_data[0][i]["box"][3])
            p = np.append([], fitting_data[0][i]["box"][4])
        
        fitting_x = np.polyfit(frame_idx, x, 4)
        frame_x = np.poly1d(fitting_x)
        fitting_y = np.polyfit(frame_idx, y, 4)
        frame_y = np.poly1d(fitting_y)
        fitting_w = np.polyfit(frame_idx, w, 4)
        frame_w = np.poly1d(fitting_w)
        fitting_h = np.polyfit(frame_idx, h, 4)
        frame_h = np.poly1d(fitting_h)
        fitting_p = np.polyfit(frame_idx, p, 4)
        frame_p = np.poly1d(fitting_p)
        
        now_bbox = [frame_x(now_frame), frame_y(now_frame), 
            frame_w(now_frame), frame_h(now_frame), frame_p(now_frame)]
        return now_bbox

    def _match(self, detections, frame_idx):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            targets_size = np.array([tracks[i].to_tlwh() for i in track_indices])           #add
            
            # add judgment, if the Euclidean Metric between fitting_objects and detections over threshold, 
            # we need to remove the fitting_object in track_indices
            dets_x = np.array([dets[i].tlwh[0] for i in detection_indices])
            dets_y = np.array([dets[i].tlwh[1] for i in detection_indices])
            
            if self.disappearing_box != []:
                for dis_box in self.disappearing_box:
                    distance = ((dis_box[1][0] - dets_x)**2 + (dis_box[1][1] - dets_y)**2)**0.5
                    if min(distance) > 100 and dis_box[0] in targets:
                        temporary = list(zip(list(targets), list(track_indices)))
                        temporary.pop(list(targets).index(dis_box[0]))
                        targets, track_indices = zip(*temporary)
                        targets = np.array(targets)
            
            cost_matrix = self.metric.distance(features, targets, targets_size)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix
        
        # Test for change track_id match sequence through compare confidence          #add    
        confidence_score = []
        Sequence_tracks = []
        if(len(self.tracks) != 0):
            for i in range(len(self.tracks)):
                confidence_score.append(self.tracks[i].to_tlwh()[4])
            confidence_idxs = np.argsort(np.array(confidence_score))
            while len(confidence_idxs) > 0:
                idxs_last = len(confidence_idxs) - 1
                j = confidence_idxs[idxs_last]
                Sequence_tracks.append(self.tracks[j])
                confidence_idxs = np.delete(confidence_idxs,[idxs_last])
            self.tracks = Sequence_tracks

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)
                
        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]                                    #lost frames over twice
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        
        MT_a, MT_b, UMT_a, UMT_b, UCT = [], [], [], [], []
        for i,a in enumerate(matches_a):
            MT_a.append(self.tracks[a[0]].track_id)
        for i,b in enumerate(matches_b):
            MT_b.append(self.tracks[b[0]].track_id)
        for i,m in enumerate(unmatched_tracks_a):
            UMT_a.append(self.tracks[m].track_id)
        for i,n in enumerate(unmatched_tracks_b):
            UMT_b.append(self.tracks[n].track_id)
        for i,t in enumerate(unconfirmed_tracks):
            UCT.append(self.tracks[t].track_id) # new target
        
        for fit in self.metric.fitting_id:
            if fit[0][0] in UMT_a and fit[0][1] in UMT_a and fit[1] not in self.fitting_object:
                self.fitting_object.append(fit[1])
             
        if self.fitting_object != []:
            for fit_obj in self.fitting_object:
                if fit_obj not in UMT_a:
                    self.fitting_object.remove(fit_obj)
        
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature,detection.confidence))
        self._next_id += 1
