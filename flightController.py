import json
from pathlib import Path
import math
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleAttitude,
)

import message_filters


def px4_qos():
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )


def quat_to_yaw(q):
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


@dataclass
class ObservationFrame:
    t_sec: float
    pos_x: float
    pos_y: float
    pos_z: float
    yaw: float
    points_body: np.ndarray
    img_bins: np.ndarray


class HybridEKF2FusionAvoidance(Node):
    def __init__(self):
        super().__init__('hybrid_ekf2_fusion_avoidance')

        # Topics
        self.declare_parameter(
            'image_topic',
            '/world/baylands/model/x500_depth_0/link/camera_link/sensor/IMX214/image'
        )
        self.declare_parameter('cloud_topic', '/depth_camera/points')

        # Approximate timestamp sync
        self.declare_parameter('sync_slop_ms', 8.0)
        self.declare_parameter('sync_queue_size', 50)

        # EKF2-backed local map memory
        self.declare_parameter('map_window_sec', 1.5)
        self.declare_parameter('max_points_per_frame', 2500)

        # Point-cloud filtering
        self.declare_parameter('front_min_m', 0.8)
        self.declare_parameter('front_max_m', 8.0)
        self.declare_parameter('lateral_half_width_m', 4.0)
        self.declare_parameter('vertical_half_height_m', 1.0)

        # Image clutter
        self.declare_parameter('img_edge_threshold_low', 80)
        self.declare_parameter('img_edge_threshold_high', 150)

        # Candidate heading planner
        self.declare_parameter('num_heading_candidates', 11)
        self.declare_parameter('max_heading_offset_deg', 16.0)
        self.declare_parameter('lookahead_m', 4.0)

        # Costs
        self.declare_parameter('goal_weight', 0.70)
        self.declare_parameter('switch_penalty', 0.12)
        self.declare_parameter('img_weight', 0.08)
        self.declare_parameter('map_weight', 0.92)

        # Occupancy shaping
        self.declare_parameter('depth_sigma_m', 1.2)
        self.declare_parameter('min_points_per_candidate', 10)
        self.declare_parameter('inflation_neighbors', 1)
        self.declare_parameter('hard_block_threshold', 0.78)

        # Mission
        self.declare_parameter('position_tolerance', 6.0)
        self.declare_parameter('hold_ticks', 5)
        self.declare_parameter('max_wp_time_sec', 35.0)

        #waypoints
        self.declare_parameter('waypoint_file', '')

        self.image_topic = self.get_parameter('image_topic').value
        self.cloud_topic = self.get_parameter('cloud_topic').value

        self.sync_slop_ms = float(self.get_parameter('sync_slop_ms').value)
        self.sync_queue_size = int(self.get_parameter('sync_queue_size').value)

        self.map_window_sec = float(self.get_parameter('map_window_sec').value)
        self.max_points_per_frame = int(self.get_parameter('max_points_per_frame').value)

        self.front_min_m = float(self.get_parameter('front_min_m').value)
        self.front_max_m = float(self.get_parameter('front_max_m').value)
        self.lateral_half_width_m = float(self.get_parameter('lateral_half_width_m').value)
        self.vertical_half_height_m = float(self.get_parameter('vertical_half_height_m').value)

        self.img_edge_threshold_low = int(self.get_parameter('img_edge_threshold_low').value)
        self.img_edge_threshold_high = int(self.get_parameter('img_edge_threshold_high').value)

        self.num_heading_candidates = int(self.get_parameter('num_heading_candidates').value)
        self.max_heading_offset_deg = float(self.get_parameter('max_heading_offset_deg').value)
        self.lookahead_m = float(self.get_parameter('lookahead_m').value)

        self.goal_weight = float(self.get_parameter('goal_weight').value)
        self.switch_penalty = float(self.get_parameter('switch_penalty').value)
        self.img_weight = float(self.get_parameter('img_weight').value)
        self.map_weight = float(self.get_parameter('map_weight').value)

        self.depth_sigma_m = float(self.get_parameter('depth_sigma_m').value)
        self.min_points_per_candidate = int(self.get_parameter('min_points_per_candidate').value)
        self.inflation_neighbors = int(self.get_parameter('inflation_neighbors').value)
        self.hard_block_threshold = float(self.get_parameter('hard_block_threshold').value)

        self.position_tolerance = float(self.get_parameter('position_tolerance').value)
        self.hold_ticks = int(self.get_parameter('hold_ticks').value)
        self.max_wp_time_sec = float(self.get_parameter('max_wp_time_sec').value)

        qos = px4_qos()

        # PX4 pubs
        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.traj_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.cmd_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos)

        # PX4 subs (EKF2-backed state)
        self.local_pos_sub = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self.local_pos_cb, qos
        )
        self.att_sub = self.create_subscription(
            VehicleAttitude, '/fmu/out/vehicle_attitude', self.attitude_cb, qos
        )

        # Synced perception
        self.img_sub = message_filters.Subscriber(self, Image, self.image_topic)
        self.cloud_sub = message_filters.Subscriber(self, PointCloud2, self.cloud_topic)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.img_sub, self.cloud_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop_ms / 1000.0
        )
        self.sync.registerCallback(self.synced_cb)

        self.timer = self.create_timer(0.1, self.timer_cb)

        # EKF2 state
        self.current_x = None
        self.current_y = None
        self.current_z = None
        self.current_yaw = 0.0

        # rolling observation memory
        self.obs_buffer = deque(maxlen=60)
        self.latest_pair_time = None
        self.prev_choice = self.num_heading_candidates // 2

        # mission execution
        self.counter = 0
        self.state = "startup"
        self.wp_index = 0
        self.hold_counter = 0
        self.wp_start_time = None

        waypoint_file = self.get_parameter('waypoint_file').value

        if waypoint_file:
            p = Path(waypoint_file)
            with open(p, 'r') as f:
                self.waypoints = json.load(f)
            self.get_logger().info(f'Loaded waypoint file: {p}')
        else:
            z = -5.0
            self.waypoints = [
                (0.0,   0.0,  z),
                (14.0,  1.0,  z),
                (28.0,  3.0,  z),
                (42.0, -3.5, z),
                (58.0,  4.5, z),
                (74.0, -4.5, z),
                (90.0,  5.0, z),
                (106.0, -5.0, z),
                (122.0,  4.5, z),
                (136.0, -3.5, z),
                (128.0,  4.0, z),
                (108.0, -4.5, z),
                (84.0,   4.0, z),
                (58.0,  -3.5, z),
                (28.0,   1.5, z),
                (0.0,    0.0, z),
            ]

        # 20-waypoint but shorter mission
        # self.waypoints = [
        #     (0.0,   0.0,  z),
        #     (8.0,   0.0,  z),
        #     (16.0,  3.0,  z),
        #     (24.0, -3.0,  z),
        #     (32.0,  3.0,  z),
        #     (40.0, -3.0,  z),
        #     (48.0,  4.0,  z),
        #     (56.0, -4.0,  z),
        #     (64.0,  4.0,  z),
        #     (72.0, -4.0,  z),
        #     (80.0, -1.5, z),
        #     (88.0,  1.5, z),
        #     (80.0,  3.5, z),
        #     (72.0, -3.5, z),
        #     (64.0,  3.5, z),
        #     (56.0, -3.5, z),
        #     (48.0,  2.5, z),
        #     (36.0,  0.0, z),
        #     (18.0,  0.0, z),
        #     (0.0,   0.0, z),
        # ]

        # self.waypoints = [
        #     (0.0,    0.0,  z),   # 1
        #     (8.0,    0.0,  z),   # 2
        #     (20.0,   3.0,  z),   # 3
        #     (32.0,  -3.0,  z),   # 4
        #     (44.0,   3.5,  z),   # 5
        #     (56.0,  -3.5,  z),   # 6
        #     (72.0,   5.5,  z),   # 7
        #     (92.0,  -5.5,  z),   # 8
        #     (112.0,  5.5,  z),   # 9
        #     (128.0, -5.5,  z),   # 10
        #     (140.0, -3.0,  z),   # 11
        #     (132.0,  3.0,  z),   # 12
        #     (118.0,  5.0,  z),   # 13
        #     (98.0,  -5.0,  z),   # 14
        #     (78.0,   5.0,  z),   # 15
        #     (60.0,  -5.0,  z),   # 16
        #     (44.0,   2.5,  z),   # 17
        #     (28.0,   0.0,  z),   # 18
        #     (14.0,   0.0,  z),   # 19
        #     (0.0,    0.0,  z),   # 20
        # ]

        self.get_logger().info(
            f"Hybrid EKF2 fusion avoidance started. sync_slop={self.sync_slop_ms} ms"
        )

    # -------------------------
    # PX4 / EKF2 state
    # -------------------------
    def local_pos_cb(self, msg):
        self.current_x = float(msg.x)
        self.current_y = float(msg.y)
        self.current_z = float(msg.z)

    def attitude_cb(self, msg):
        self.current_yaw = quat_to_yaw([float(v) for v in msg.q])

    def now_sec(self):
        return self.get_clock().now().nanoseconds / 1e9

    def wrap_angle(self, a):
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    # -------------------------
    # PX4 commands
    # -------------------------
    def publish_offboard_heartbeat(self):
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = True
        self.offboard_pub.publish(msg)

    def publish_setpoint(self, x, y, z, yaw=0.0):
        msg = TrajectorySetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = [float(x), float(y), float(z)]
        msg.velocity = [math.nan, math.nan, math.nan]
        msg.acceleration = [math.nan, math.nan, math.nan]
        msg.yaw = float(yaw)
        msg.yawspeed = math.nan
        self.traj_pub.publish(msg)

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.command = int(command)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.cmd_pub.publish(msg)

    def arm(self):
        self.publish_vehicle_command(400, 1.0, 0.0)

    def set_offboard_mode(self):
        self.publish_vehicle_command(176, 1.0, 6.0)

    def land(self):
        self.publish_vehicle_command(21, 0.0, 0.0)

    # -------------------------
    # Mission geometry
    # -------------------------
    def point_segment_metrics(self, p, a, b):
        ab = b - a
        ab2 = np.dot(ab, ab)
        if ab2 < 1e-9:
            return float(np.linalg.norm(p - a)), 0.0
        t = float(np.dot(p - a, ab) / ab2)
        t_clamped = max(0.0, min(1.0, t))
        proj = a + t_clamped * ab
        dist = float(np.linalg.norm(p - proj))
        return dist, t

    def current_nominal_target(self):
        return self.waypoints[self.wp_index]

    def distance_to_nominal_wp(self):
        if self.current_x is None:
            return None
        tx, ty, tz = self.current_nominal_target()
        dx = tx - self.current_x
        dy = ty - self.current_y
        dz = tz - self.current_z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def passed_waypoint(self):
        if self.current_x is None or self.wp_index == 0:
            return False
        prev_wp = np.array(self.waypoints[self.wp_index - 1], dtype=float)
        curr_wp = np.array(self.waypoints[self.wp_index], dtype=float)
        pos = np.array([self.current_x, self.current_y, self.current_z], dtype=float)
        _, t = self.point_segment_metrics(pos, prev_wp, curr_wp)
        return t > 1.05

    def should_advance_waypoint(self):
        dist = self.distance_to_nominal_wp()
        if dist is not None and dist < self.position_tolerance:
            self.hold_counter += 1
            if self.hold_counter >= self.hold_ticks:
                return True
        else:
            self.hold_counter = 0

        if self.passed_waypoint():
            return True

        if self.wp_start_time is not None and (self.now_sec() - self.wp_start_time) > self.max_wp_time_sec:
            self.get_logger().warn(f"Forcing advance of waypoint {self.wp_index + 1}")
            return True

        return False

    # -------------------------
    # Perception feature extraction
    # -------------------------
    def image_msg_to_bgr(self, msg):
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        enc = msg.encoding.lower()
        if enc == 'rgb8':
            img = arr.reshape((msg.height, msg.width, 3))
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif enc == 'bgr8':
            return arr.reshape((msg.height, msg.width, 3))
        elif enc == 'mono8':
            gray = arr.reshape((msg.height, msg.width))
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        raise ValueError(f"Unsupported encoding: {msg.encoding}")

    def extract_img_bins(self, img_msg):
        img = self.image_msg_to_bgr(img_msg)
        h, w, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cy0 = int(0.30 * h)
        cy1 = int(0.80 * h)
        band = gray[cy0:cy1, :]

        edges = cv2.Canny(band, self.img_edge_threshold_low, self.img_edge_threshold_high)

        bins = np.zeros(3, dtype=np.float32)
        thirds = [0, w // 3, 2 * w // 3, w]
        for i in range(3):
            roi = edges[:, thirds[i]:thirds[i + 1]]
            bins[i] = float(np.mean(roi > 0)) if roi.size > 0 else 0.0

        mx = float(np.max(bins))
        if mx > 1e-6:
            bins = bins / mx
        return bins

    def extract_cloud_points_body(self, cloud_msg):
        pts = []
        for p in point_cloud2.read_points(cloud_msg, field_names=['x', 'y', 'z'], skip_nans=True):
            try:
                x, y, z = float(p[0]), float(p[1]), float(p[2])
                if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                    if (z > self.front_min_m and z < self.front_max_m and
                        abs(x) < self.lateral_half_width_m and
                        abs(y) < self.vertical_half_height_m):
                        pts.append((x, y, z))
            except Exception:
                continue

        if not pts:
            return np.zeros((0, 3), dtype=np.float32)

        pts = np.asarray(pts, dtype=np.float32)

        if len(pts) > self.max_points_per_frame:
            idx = np.random.choice(len(pts), self.max_points_per_frame, replace=False)
            pts = pts[idx]

        return pts

    def synced_cb(self, img_msg, cloud_msg):
        if self.current_x is None:
            return

        img_bins = self.extract_img_bins(img_msg)
        pts_body = self.extract_cloud_points_body(cloud_msg)

        obs = ObservationFrame(
            t_sec=self.now_sec(),
            pos_x=self.current_x,
            pos_y=self.current_y,
            pos_z=self.current_z,
            yaw=self.current_yaw,
            points_body=pts_body,
            img_bins=img_bins,
        )
        self.obs_buffer.append(obs)
        self.latest_pair_time = obs.t_sec

    # -------------------------
    # EKF2-backed local map
    # -------------------------
    def body_to_world_xy(self, x_right, z_forward, yaw, pos_x, pos_y):
        wx = pos_x + z_forward * math.cos(yaw) - x_right * math.sin(yaw)
        wy = pos_y + z_forward * math.sin(yaw) + x_right * math.cos(yaw)
        return wx, wy

    def world_to_current_body(self, wx, wy):
        dx = wx - self.current_x
        dy = wy - self.current_y

        forward = dx * math.cos(self.current_yaw) + dy * math.sin(self.current_yaw)
        right = -dx * math.sin(self.current_yaw) + dy * math.cos(self.current_yaw)
        return right, forward

    def aggregate_candidate_occupancy(self):
        n = self.num_heading_candidates
        occ = np.zeros(n, dtype=np.float32)

        if self.current_x is None:
            return occ

        now = self.now_sec()
        max_off = math.radians(self.max_heading_offset_deg)
        cand_angles = np.linspace(-max_off, max_off, n)

        for obs in list(self.obs_buffer):
            if (now - obs.t_sec) > self.map_window_sec:
                continue
            if obs.points_body.shape[0] == 0:
                continue

            local_pts = []
            for p in obs.points_body:
                x_r, _, z_f = float(p[0]), float(p[1]), float(p[2])

                wx, wy = self.body_to_world_xy(x_r, z_f, obs.yaw, obs.pos_x, obs.pos_y)
                curr_r, curr_f = self.world_to_current_body(wx, wy)

                if (curr_f > self.front_min_m and curr_f < self.front_max_m and
                    abs(curr_r) < self.lateral_half_width_m):
                    local_pts.append((curr_r, curr_f))

            if not local_pts:
                continue

            local_pts = np.asarray(local_pts, dtype=np.float32)
            pt_ang = np.arctan2(local_pts[:, 0], local_pts[:, 1])  # angle in current body frame

            for i, ang in enumerate(cand_angles):
                delta = np.abs(pt_ang - ang)
                delta = np.minimum(delta, 2 * np.pi - delta)

                mask = delta < math.radians(7.0)
                region = local_pts[mask]
                if len(region) < self.min_points_per_candidate:
                    continue

                dmin = float(np.min(np.sqrt(region[:, 0]**2 + region[:, 1]**2)))
                cost = float(np.exp(-(dmin - self.front_min_m) / max(self.depth_sigma_m, 1e-3)))
                occ[i] = max(occ[i], max(0.0, min(1.0, cost)))

        # inflate neighbors
        inflated = occ.copy()
        for _ in range(self.inflation_neighbors):
            tmp = inflated.copy()
            for i in range(n):
                vals = [inflated[i]]
                if i > 0:
                    vals.append(inflated[i - 1])
                if i < n - 1:
                    vals.append(inflated[i + 1])
                tmp[i] = max(vals)
            inflated = tmp

        return inflated

    # -------------------------
    # Hybrid planning
    # -------------------------
    def choose_heading(self):
        tx, ty, _ = self.current_nominal_target()

        if self.current_x is None:
            return self.current_yaw

        goal_heading = math.atan2(ty - self.current_y, tx - self.current_x)
        rel_goal = self.wrap_angle(goal_heading - self.current_yaw)

        n = self.num_heading_candidates
        max_off = math.radians(self.max_heading_offset_deg)
        cand_angles = np.linspace(-max_off, max_off, n)

        occ = self.aggregate_candidate_occupancy()

        if self.obs_buffer:
            img3 = self.obs_buffer[-1].img_bins
        else:
            img3 = np.zeros(3, dtype=np.float32)

        imgN = np.interp(
            np.linspace(0, 2, n),
            np.array([0, 1, 2], dtype=np.float32),
            img3
        )

        rel_goal_clipped = max(-max_off, min(max_off, rel_goal))

        costs = np.zeros(n, dtype=np.float32)
        for i in range(n):
            goal_cost = self.goal_weight * abs(float(cand_angles[i] - rel_goal_clipped))
            switch_cost = self.switch_penalty if i != self.prev_choice else 0.0
            hazard_cost = self.map_weight * occ[i] + self.img_weight * imgN[i]
            costs[i] = hazard_cost + goal_cost + switch_cost

        choice = int(np.argmin(costs))

        if occ[choice] > self.hard_block_threshold:
            choice = int(np.argmin(occ))

        self.prev_choice = choice
        desired_heading = self.current_yaw + cand_angles[choice]
        return self.wrap_angle(desired_heading)

    def compute_local_target(self):
        if self.current_x is None:
            return 0.0, 0.0, -5.0, 0.0

        desired_heading = self.choose_heading()
        _, _, tz = self.current_nominal_target()

        tx = self.current_x + self.lookahead_m * math.cos(desired_heading)
        ty = self.current_y + self.lookahead_m * math.sin(desired_heading)

        return tx, ty, tz, desired_heading

    # -------------------------
    # Main execution loop
    # -------------------------
    def timer_cb(self):
        self.publish_offboard_heartbeat()

        if self.counter < 15:
            self.publish_setpoint(0.0, 0.0, -5.0, 0.0)
            self.counter += 1
            return

        if self.counter == 15:
            self.set_offboard_mode()
            self.arm()
            self.counter += 1
            self.state = "mission"
            self.wp_start_time = self.now_sec()
            return

        if self.state == "mission":
            if self.wp_index >= len(self.waypoints):
                self.get_logger().info("Mission complete, landing")
                self.state = "land"
                return

            tx, ty, tz, yaw = self.compute_local_target()
            self.publish_setpoint(tx, ty, tz, yaw)

            if self.should_advance_waypoint():
                self.get_logger().info(f"Reached waypoint {self.wp_index + 1}/{len(self.waypoints)}")
                self.wp_index += 1
                self.hold_counter = 0
                self.wp_start_time = self.now_sec()

        elif self.state == "land":
            self.land()


def main(args=None):
    rclpy.init(args=args)
    node = HybridEKF2FusionAvoidance()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

