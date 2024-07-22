import cv2
import numpy as np


class OpticalFlow:
    def __init__(self, track_len, detect_interval, resize_scale):
        self.track_len = track_len
        self.detect_interval = detect_interval
        self.resize_scale = resize_scale

        self.tracks = []
        self.frame_idx = 0
        self.prev_gray = None

        # Parameters for ShiTomasi corner detection
        self.feature_params = dict(
            maxCorners=5,
            qualityLevel=0.5,
            minDistance=7,
            blockSize=7
        )

        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(
            winSize=(7, 7),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

    def process_image(self, image, delta_t, debug_viz_on=False):
        """
        Process the input image using optical flow algorithm.

        Args:
            image (numpy.ndarray): The input image.
            delta_t (float): The time interval between frames.
            debug_viz_on (bool, optional): Flag to enable debug visualization. Defaults to False.

        Returns:
            disp_arr (numpy.ndarray): Array of displacements.
            velocities_arr (numpy.ndarray): Array of velocities.
            locations (list): List of locations.
            vis (numpy.ndarray): The debug visualization image.
            debug_str (str): The debug string.
        """
        scaled_height, scaled_width = (int(self.resize_scale * dim) for dim in image.shape[:2])
        image_cv = cv2.resize(image, (scaled_height, scaled_width), interpolation=cv2.INTER_NEAREST)
        frame_gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

        if debug_viz_on:
            vis = image_cv.copy()

        debug_str = ""

        disp_arr = np.array([(0,0),])
        velocities_arr = np.array([(0,0),])
        locations = [(0,0), ]

        if len(self.tracks) > 0:
            img0, img1 = self.prev_gray, frame_gray
            p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
            p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
            
            # Check if the optical flow is valid by computing it backwards and comparing the results with the original frame
            p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)
            d = abs(p0 - p0r).reshape(-1, 2).max(-1)
            good = d < 1

            new_tracks = []
            speeds = []  # Unit: pixel / sec
            displacements = []  # Unit: pixel
            locations = []  # Locations of the vectors

            for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                tr.append((x, y))
                if len(tr) > self.track_len:
                    del tr[1]
                new_tracks.append(tr)

                v_est = self.single_track_speed_est(track=tr, delta_t=delta_t)
                if v_est is not None:
                    speeds.append(v_est)
                    locations.append((x, y))

                est = self.single_track_est(track=tr)
                if est is not None:
                    displacements.append(est)

                if debug_viz_on:
                    cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
            self.tracks = new_tracks
            

            if len(speeds) > 0:
                velocities_arr = np.array(speeds)
                m_vx, m_vy = np.mean(velocities_arr, axis=0)
                std_v = np.std(velocities_arr, axis=0)
                debug_str = f"vx: {m_vx:>10.4f} [px/s], vy: {m_vy:>10.4f} [px/s], stddev: {std_v} [px/s]\n"

            if len(displacements) > 0:
                disp_arr = np.array(displacements)
                m_x, m_y = np.mean(disp_arr, axis=0)
                std_xy = np.std(disp_arr, axis=0)
                debug_str += f"x: {m_x:>10.4f} [px], y: {m_y:>10.4f} [px], stddev: {std_xy} [px]\n"

            if debug_viz_on:
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))

        if self.frame_idx % self.detect_interval == 0 or len(self.tracks) == 0:
            mask = np.zeros_like(frame_gray)
            mask[:] = 255
            for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                cv2.circle(mask, (x, y), 5, 0, -1)
            p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **self.feature_params)
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    self.tracks.append([(x, y)])

        self.frame_idx += 1
        self.prev_gray = frame_gray

        return disp_arr, velocities_arr, locations, vis if debug_viz_on else None, debug_str

    @staticmethod
    def single_track_speed_est(track, delta_t):
        if len(track) > 1:
            x0, y0 = track[-2]
            x1, y1 = track[-1]
            vx = (x1 - x0) / delta_t
            vy = (y1 - y0) / delta_t
            return vx, vy
        else:
            return None

    @staticmethod
    def single_track_est(track, thres: float = 0.1):
        if len(track) > 1:
            x0, y0 = track[0]
            x1, y1 = track[-1]
            dx = x1 - x0
            dy = y1 - y0
            if abs(dx) < thres:
                dx = 0.0
            if abs(dy) < thres:
                dy = 0.0
            return dx, dy
        else:
            return None
