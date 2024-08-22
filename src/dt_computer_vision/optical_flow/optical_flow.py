from typing import List, Optional, Tuple
import cv2
import numpy as np

from dt_computer_vision.camera.homography import Homography
from dt_computer_vision.camera.types import CameraModel, Pixel


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
    def compute_motion_vectors(self, image, delta_t : float) -> Tuple[List[Pixel], List[Pixel], List[Pixel], str] :
        """
        Process the input image using optical flow algorithm.

        Args:
            image (numpy.ndarray): The input image.
            delta_t (float): The time interval between frames.

        Returns:
            disp_arr (List[Pixel]): Array of displacements in [pixels] (detected_features x 2).
            velocities_arr (List[Pixel]): Array of velocities in [pixels/s] (detected_features x 2).
            locations (list): List of locations (detected_features x 2).
            debug_str (str): The debug string.
        """

        image_cv = self.scale_image(image)
        frame_gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

        debug_str = ""
        disp_arr = np.array([(0, 0)])
        velocities_arr = np.array([(0, 0)])
        locations = [(0, 0)]

        if len(self.tracks) > 0:
            img0, img1 = self.prev_gray, frame_gray
            p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
            p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
            
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

            self.tracks = new_tracks

            if len(speeds) > 0:
                velocities_arr = np.array(speeds)
                m_vx, m_vy = np.mean(velocities_arr, axis=0)
                std_v = np.std(velocities_arr, axis=0)
                debug_str = f"vx:{m_vx:>10.4f}[px/s],vy:{m_vy:>10.4f}[px/s]\n stddev: {std_v} [px/s]\n"

            if len(displacements) > 0:
                disp_arr = np.array(displacements)
                m_x, m_y = np.mean(disp_arr, axis=0)
                std_xy = np.std(disp_arr, axis=0)
                debug_str += f"x:{m_x:>10.4f}[px],y:{m_y:>10.4f}[px]\n stddev: {std_xy} [px]\n"

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

        locations_px = [Pixel(*loc) for loc in locations]
        disp_arr_px = [Pixel(*disp) for disp in disp_arr]
        velocities_arr_px = [Pixel(*vel) for vel in velocities_arr]
    
        return disp_arr_px, velocities_arr_px, locations_px, debug_str

    def scale_image(self, image : np.ndarray) -> np.ndarray:
        scaled_height, scaled_width = (int(self.resize_scale * dim) for dim in image.shape[:2])
        image_cv = cv2.resize(image, (scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST)
        return image_cv


    def create_debug_visualization(self, image : np.ndarray, locations : List[Pixel], debug_str : Optional[str] = None, motion_vectors : Optional[List[Pixel]] = None) -> np.ndarray:
        """
        Create a debug visualization image.

        Args:
            image (numpy.ndarray): The original image.
            locations (list): List of locations (detected_features x 2) they have to match the scale of OpticalFlow.resize_scale.
            motion_vectors (list): List of motion vectors (detected_features x 2).
            debug_str (str): The debug string.
            scale (int): Scale factor for the debug image.

        Returns:
            vis (numpy.ndarray): The debug visualization image.
        """

        image_cv = self.scale_image(image)
        vis = image_cv.copy()

        if motion_vectors is not None:
            for loc, vector in zip(locations, motion_vectors):
                x, y = loc.x, loc.y
                dx, dy = vector.x, vector.y
                cv2.arrowedLine(vis, (int(x), int(y)), (int(x+dx), int(y+dy)), (0, 255, 0), 2)
        else:
            for loc in locations:
                x, y = loc.x, loc.y
                cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
                
        # Embed debug string in the debug image
        if debug_str is not None:
            cv2.putText(vis, debug_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Check that the returned image has the same aspect ratio as the input image
        assert vis.shape[0] == image.shape[0] * self.resize_scale, "The height of the debug image is not correct."
        assert vis.shape[1] == image.shape[1] * self.resize_scale, "The width of the debug image is not correct."
        
        return vis

    def project_motion_vectors(
        self,
        vectors: List[Pixel],
        locations: List[Pixel],
        camera: CameraModel,
        H: Homography,
        # projector: GroundProjector,
    ) -> Tuple[List[Pixel], List[Pixel]]:
        """
        Project the motion vector using the provided projector.
        
        Args:
            vectors (List[Pixel]): The motion vector.
            locations (List[Pixel]): The location of the motion vector.
            camera (CameraModel): The camera model.
            projector (GroundProjector): The ground projector.
            
        Returns:
            List[Pixel]: The projected motion vector        
            List[Pixel]: The projected locations of the motion vector        
        """
        
        # Since we are processing the image in a scaled form, we need to scale the vector back to the original image size
        projected_motion_vectors = []
        projected_locations = []
        for vector, loc in zip(vectors, locations):
            
            # Compute the absolute position of the head and tail of the displacement vector
            head = loc + vector
            tail = loc


            # Rectify the head and tail of the displacement vector, scaling them to the original image size
            head_px = head / self.resize_scale
            tail_px = tail / self.resize_scale

            # head = camera.rectifier.rectify_pixel(head_px)
            # tail = camera.rectifier.rectify_pixel(tail_px)

            # Project the head and tail of the displacement vector to the ground
            
            # head_ground = projector.vector2ground(camera.pixel2vector(head))
            # tail_ground = projector.vector2ground(camera.pixel2vector(tail))

            head_ground = self._project_pixel(head_px, H)
            tail_ground = self._project_pixel(tail_px, H)

            projected_locations.append(tail_ground)

            # Compute the resulting displacement vector
            projected_vector = head_ground - tail_ground

            projected_motion_vectors.append(projected_vector)
            
        return projected_motion_vectors, projected_locations

    @staticmethod
    def _project_pixel(pixel: Pixel, H : Homography) -> Pixel:
        # project the pixel to the ground plane
        p = np.array([pixel.x, pixel.y, 1]).reshape(3, 1)
        p_ground = H @ p
        p_ground /= p_ground[2]
        
        return Pixel(p_ground[0], p_ground[1])


    def compute_velocity_vector(self, motion_vectors: List[Pixel]) -> np.ndarray:
        """
        Compute a 2D velocity vector (in [m/s]) from a list of motion vectors (in [px/s]), according to the ground projector and camera models given. 
        """
        # Convert the motion_vectors to a numpy array
        motion_vectors_arr = [np.array([v.x, v.y]) for v in motion_vectors]
        
        mean_velocity_vector = np.mean(np.array(motion_vectors_arr), axis=0)
        
        return mean_velocity_vector

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
