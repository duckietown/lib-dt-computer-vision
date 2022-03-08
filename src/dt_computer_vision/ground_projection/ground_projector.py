#!/usr/bin/env python3

import numpy as np

from dt_computer_vision.ground_projection.types import \
    GroundPoint, \
    NormalizedImagePoint, \
    CameraModel


class GroundProjector:
    """
        Handles the Ground Projection operations.

        This class projects points in the image to the ground plane and in the robot's
        reference frame.
        It enables lane localization in the 2D ground plane.
        This projection uses the homography obtained from the extrinsic calibration procedure.

        Note:
            All pixel and image operations in this class assume that the pixels and images are
            *already rectified*.
            If unrectified pixels or images are supplied, the outputs of these operations will
            be incorrect.

        Args:
            camera (``CameraModel``): Object describing the camera model

        """

    def __init__(self, camera: CameraModel):
        self.camera = camera
        # store homography
        if self.camera.H is None:
            raise ValueError("You need to set a homography `H` in your CameraModel object before "
                             "you can use it to create an instance of GroundProjector.")
        self.H: np.ndarray = self.camera.H.reshape((3, 3))
        # invert homography
        self.Hinv: np.ndarray = np.linalg.inv(self.H)
        # cache/support objects
        self.debug_img_bg = None

    def vector2ground(self, point: NormalizedImagePoint) -> GroundPoint:
        """
        Projects a normalized point (``[0, 1] X [0, 1]``) to the ground
        plane using the homography matrix.

        Args:
            point (:py:class:`Point`): A :py:class:`Point` object in normalized coordinates.

        Returns:
            :py:class:`Point` : A :py:class:`Point` object on the ground plane.

        """
        uv_raw = np.array([point.x, point.y, 1.0])
        ground_point = np.dot(self.H, uv_raw)
        x = ground_point[0] / ground_point[2]
        y = ground_point[1] / ground_point[2]
        return GroundPoint(x, y)

    def ground2vector(self, point: GroundPoint) -> NormalizedImagePoint:
        """
        Projects a point on the ground plane to a normalized pixel (``[0, 1] X [0, 1]``) using the
        homography matrix.

        Args:
            point (:py:class:`Point`): A :py:class:`Point` object on the ground plane.

        Returns:
            :py:class:`Point` : A :py:class:`Point` object in normalized coordinates.

        """
        ground_point = np.array([point.x, point.y, 1.0])
        image_point = np.dot(self.Hinv, ground_point)
        image_point = image_point / image_point[2]
        return NormalizedImagePoint(image_point[0], image_point[1])

    def project_to_ground(self, point: NormalizedImagePoint) -> GroundPoint:
        """
        Creates a :py:class:`ground_projection.types.GroundPoint` object from a normalized point
        message from a distorted image. It converts it to pixel coordinates and rectifies it.
        Then projects it to the ground plane.

        Args:
            point (:obj:`ground_projection.types.NormalizedImagePoint`): Normalized point
            coordinates from a distorted image.

        Returns:
            :obj:`ground_projection.types.GroundPoint`: Point coordinates in the ground
            reference frame.

        """
        # point to pixel [distorted point -> distorted pixel]
        # pixel = self.camera.vector2pixel(point)
        # rectify [distorted pixel -> rectified pixel]
        # pixel_rect = self.camera.rectifier.rectify_point(point)

        #
        point_rect = self.camera.rectifier.rectify_point(point)

        # convert back to point [rectified pixel -> rectified point]
        # point_rect = self.camera.pixel2vector(pixel_rect)
        # project on ground [rectified point -> ground point]
        ground_pt = self.vector2ground(point_rect)
        # ---
        return ground_pt

    # def lineseglist_cb(self, seglist_msg):
    #     """
    #     Projects a list of line segments on the ground reference frame point by point by
    #     calling :py:meth:`pixel_msg_to_ground_msg`. Then publishes the projected list of segments.
    #
    #     Args:
    #         seglist_msg (:obj:`duckietown_msgs.msg.SegmentList`): Line segments in pixel space from
    #         unrectified images
    #
    #     """
    #     if self.camera_info_received:
    #         seglist_out = SegmentList()
    #         seglist_out.header = seglist_msg.header
    #         for received_segment in seglist_msg.segments:
    #             new_segment = Segment()
    #             new_segment.points[0] = self.pixel_msg_to_ground_msg(received_segment.pixels_normalized[0])
    #             new_segment.points[1] = self.pixel_msg_to_ground_msg(received_segment.pixels_normalized[1])
    #             new_segment.color = received_segment.color
    #             # TODO: what about normal and points?
    #             seglist_out.segments.append(new_segment)
    #         self.pub_lineseglist.publish(seglist_out)
    #
    #         if not self.first_processing_done:
    #             self.log("First projected segments published.")
    #             self.first_processing_done = True
    #
    #         if self.pub_debug_img.get_num_connections() > 0:
    #             debug_image_msg = self.bridge.cv2_to_compressed_imgmsg(self.debug_image(seglist_out))
    #             debug_image_msg.header = seglist_out.header
    #             self.pub_debug_img.publish(debug_image_msg)
    #     else:
    #         self.log("Waiting for a CameraInfo message", "warn")
