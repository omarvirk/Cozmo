#!/usr/bin/env python3

''' Get a raw frame from camera and display in OpenCV
By press space, save the image from 001.bmp to ...
'''

import cv2
import cozmo
import numpy as np
from numpy.linalg import inv
import threading
import time

from ar_markers.hamming.detect import detect_markers

from grid import CozGrid
from gui import GUIWindow
from particle import Particle, Robot
from setting import *
from particle_filter import *
from utils import *
from cozmo.util import *
from cozmo import anim

# camera params
camK = np.matrix([[295, 0, 160], [0, 295, 120], [0, 0, 1]], dtype='float32')

# marker size in inches
marker_size = 3.5

# tmp cache
last_pose = cozmo.util.Pose(0, 0, 0, angle_z=cozmo.util.Angle(degrees=0))
flag_odom_init = False

# goal location for the robot to drive to, (x, y, theta)
goal = (6, 10, 0)

# map
Map_filename = "map_arena.json"
grid = CozGrid(Map_filename)
gui = GUIWindow(grid)


async def image_processing(robot):
    global camK, marker_size

    event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)

    # convert camera image to opencv format
    opencv_image = np.asarray(event.image)

    # detect markers
    markers = detect_markers(opencv_image, marker_size, camK)

    # show markers
    for marker in markers:
        marker.highlite_marker(opencv_image, draw_frame=True, camK=camK)
        # print("ID =", marker.id);
        # print(marker.contours);
    cv2.imshow("Markers", opencv_image)

    return markers


# calculate marker pose
def cvt_2Dmarker_measurements(ar_markers):
    marker2d_list = [];

    for m in ar_markers:
        R_1_2, J = cv2.Rodrigues(m.rvec)
        R_1_1p = np.matrix([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
        R_2_2p = np.matrix([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        R_2p_1p = np.matmul(np.matmul(inv(R_2_2p), inv(R_1_2)), R_1_1p)
        # print('\n', R_2p_1p)
        yaw = -math.atan2(R_2p_1p[2, 0], R_2p_1p[0, 0])

        x, y = m.tvec[2][0] + 0.5, -m.tvec[0][0]
        # print('x =', x, 'y =', y,'theta =', yaw)

        # remove any duplate markers
        dup_thresh = 2.0
        find_dup = False
        for m2d in marker2d_list:
            if grid_distance(m2d[0], m2d[1], x, y) < dup_thresh:
                find_dup = True
                break
        if not find_dup:
            marker2d_list.append((x, y, math.degrees(yaw)))

    return marker2d_list


# compute robot odometry based on past and current pose
def compute_odometry(curr_pose, cvt_inch=True):
    global last_pose, flag_odom_init
    last_x, last_y, last_h = last_pose.position.x, last_pose.position.y, \
                             last_pose.rotation.angle_z.degrees
    curr_x, curr_y, curr_h = curr_pose.position.x, curr_pose.position.y, \
                             curr_pose.rotation.angle_z.degrees

    dx, dy = rotate_point(curr_x - last_x, curr_y - last_y, -last_h)
    if cvt_inch:
        dx, dy = dx / 25.6, dy / 25.6

    return (dx, dy, diff_heading_deg(curr_h, last_h))


# particle filter functionality
class ParticleFilter:
    def __init__(self, grid):
        self.particles = Particle.create_random(PARTICLE_COUNT, grid)
        self.grid = grid

    def update(self, odom, r_marker_list):
        # ---------- Motion model update ----------
        self.particles = motion_update(self.particles, odom)

        # ---------- Sensor (markers) model update ----------
        self.particles = measurement_update(self.particles, r_marker_list, self.grid)

        # ---------- Show current state ----------
        # Try to find current best estimate for display
        m_x, m_y, m_h, m_confident = compute_mean_pose(self.particles)
        return (m_x, m_y, m_h, m_confident)


async def run(robot: cozmo.robot.Robot):
    global flag_odom_init, last_pose
    global grid, gui

    # start streaming
    robot.camera.image_stream_enabled = True

    # start particle filter
    pf = ParticleFilter(grid)

    end = 0

    ###################

    await robot.set_head_angle(cozmo.util.degrees(10)).wait_for_completed()
    goalReached = False
    while True:
        await robot.drive_straight(distance_mm(0), speed_mmps(50)).wait_for_completed()
        print(robot.is_picked_up)
        if goalReached and not robot.is_picked_up:
            continue

        if robot.is_picked_up:
            robot.stop_all_motors()
            pf = ParticleFilter(grid)
            await robot.play_anim_trigger(anim.Triggers.ReactToPickup).wait_for_completed()
            await robot.set_head_angle(degrees(10)).wait_for_completed()
            await robot.drive_straight(distance_mm(0), speed_mmps(50)).wait_for_completed()
            await robot.say_text("STOP, PUT ME DOWN").wait_for_completed()
            goalReached = False
            continue

        currPose = robot.pose
        odomDiff = compute_odometry(currPose)

        markers = await image_processing(robot)
        markers = cvt_2Dmarker_measurements(markers)

        meanEstimate = pf.update(odomDiff, markers)
        meanEstimateX = meanEstimate[0]
        meanEstimateY = meanEstimate[1]
        meanEstimateTh = meanEstimate[2]
        meanEstimateConfidence = meanEstimate[3]

        gui.show_particles(pf.particles)
        gui.show_mean(meanEstimateX, meanEstimateY, meanEstimateTh, meanEstimateConfidence)
        gui.updated.set()

        if not meanEstimateConfidence:
            # Haven't yet converged, turn in place
            await robot.drive_wheels(-10, 10, duration=0)
        else:
            robot.stop_all_motors()

            goalX = goal[0]
            goalY = goal[1]

            # await turnToFace(meanEstimateTh, math.degrees(90), robot)
            print(meanEstimateTh)
            print(- meanEstimateTh + 85)
            await robot.turn_in_place(degrees(-meanEstimateTh + 85)).wait_for_completed()
            robot.stop_all_motors()
            gui.show_particles(pf.particles)
            gui.show_mean(meanEstimateX, meanEstimateY, 90, meanEstimateConfidence)
            gui.updated.set()

            angleToGoal = math.atan2(math.fabs(meanEstimateX - goalX), math.fabs(meanEstimateY - goalY))

            if goalX - meanEstimateX > 0:
                if goalY - meanEstimateY > 0:
                    faceGoal = 0 - angleToGoal
                    finishAngle = -(math.pi / 2) + angleToGoal
                else:
                    faceGoal = -math.pi + angleToGoal
                    finishAngle = (math.pi / 2) - angleToGoal
            else:
                if goalY - meanEstimateY > 0:
                    faceGoal = angleToGoal
                    finishAngle = -(math.pi / 2) - angleToGoal
                else:
                    faceGoal = math.pi - angleToGoal
                    finishAngle = (math.pi / 2) + angleToGoal
            await robot.turn_in_place(radians(faceGoal)).wait_for_completed()

            robot.stop_all_motors()

            distanceToGoal = grid_distance(goalX, goalY, meanEstimateX, meanEstimateY)
            print("MATH:", math.hypot(math.fabs(goalX - meanEstimateX), math.fabs(goalY - meanEstimateY)))
            print(meanEstimateX, ",", meanEstimateY)
            print(goalX, ",", goalY)
            print("GRID DIST:", distanceToGoal)
            start = time.time()

            while end - start < distanceToGoal:
                await robot.drive_straight(distance_mm(0), speed=speed_mmps(50)).wait_for_completed()
                if robot.is_picked_up:
                    print("Picked Up")
                    robot.stop_all_motors()
                    pf = ParticleFilter(grid)
                    await robot.play_anim_trigger(anim.Triggers.ReactToPickup).wait_for_completed()
                    await robot.set_head_angle(cozmo.util.degrees(10)).wait_for_completed()
                    await robot.drive_straight(distance_mm(0), speed_mmps(50)).wait_for_completed()
                    await robot.say_text("STOP, PUT ME DOWN").wait_for_completed()
                    break
                print("Driving")
                await robot.drive_wheels(30, 30, duration=0)
                await robot.drive_straight(distance_mm(0), speed=speed_mmps(50)).wait_for_completed()
                end = time.time()

            robot.stop_all_motors()
            print("Here")
            print(robot.is_picked_up)
            if robot.is_picked_up:
                print("PICKED UP")
                pf = ParticleFilter(grid)
                await robot.play_anim_trigger(anim.Triggers.ReactToPickup).wait_for_completed()
                await robot.set_head_angle(degrees(10)).wait_for_completed()
                await robot.drive_straight(distance_mm(0), speed_mmps(50)).wait_for_completed()
                continue

            # await turnToFace(degrees(finishAngle), 0, robot)
            print("Turning: ", finishAngle)
            await robot.turn_in_place(radians(finishAngle)).wait_for_completed()
            await robot.play_anim_trigger(anim.Triggers.BuildPyramidSuccess).wait_for_completed()
            await robot.say_text("Yay homie").wait_for_completed()
            goalReached = True

        last_pose = currPose


class CozmoThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self, daemon=False)

    def run(self):
        cozmo.robot.Robot.drive_off_charger_on_connect = False  # Cozmo can stay on his charger
        cozmo.run_program(run, use_viewer=False)


if __name__ == '__main__':
    # cozmo thread
    cozmo_thread = CozmoThread()
    cozmo_thread.start()

    # init
    grid = CozGrid(Map_filename)
    gui = GUIWindow(grid)
    gui.start()
