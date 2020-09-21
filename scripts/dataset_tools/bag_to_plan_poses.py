#!/usr/bin/env python2
import os
import argparse

import rosbag
from pyquaternion import Quaternion
from tf.transformations import *
import numpy as np
import math


def extract(bagfile, pose_topic, msg_type, out_filename):
    n = 0
    prev_yaw = 0
    yaw_align = 0
    f = open(out_filename, 'w')
    f.write('timestamp x y z vx vy vz yaw yaw_dot\n')
    with rosbag.Bag(bagfile, 'r') as bag:
        for (topic, msg, ts) in bag.read_messages(topics=str(pose_topic)):
            if msg_type == "Odometry":
                # quat = Quaternion(msg.pose.pose.orientation.w, msg.pose.pose.orientation.x,
                #                       msg.pose.pose.orientation.y, msg.pose.pose.orientation.z)
                rpy_angles = list(euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                                                    msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]))
                yaw = math.degrees(rpy_angles[2])
                yaw_dot = math.degrees(msg.twist.twist.angular.z)
                if n != 0 and yaw - prev_yaw >= 180:
                    yaw_align = -360
                elif yaw - prev_yaw <= -180:
                    yaw_align = 360
                f.write('%.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f\n' %
                            (msg.header.stamp.to_sec(),
                             msg.pose.pose.position.x,
                             msg.pose.pose.position.y,
                             msg.pose.pose.position.z,
                             msg.twist.twist.linear.x,
                             msg.twist.twist.linear.y,
                             msg.twist.twist.linear.z,
                             yaw + yaw_align,
                             yaw_dot))
            elif msg_type == "PositionCommand":
                yaw = math.degrees(msg.yaw)
                yaw_dot = math.degrees(msg.yaw_dot)
                if n != 0 and yaw - prev_yaw >= 180:
                    yaw_align = -360
                elif yaw - prev_yaw <= -180:
                    yaw_align = 360
                f.write('%.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f\n' %
                        (msg.header.stamp.to_sec(),
                         msg.position.x,
                         msg.position.y,
                         msg.position.z,
                         msg.velocity.x,
                         msg.velocity.y,
                         msg.velocity.z,
                         yaw + yaw_align,
                         yaw_dot))  # orientation W

            else:
                assert False, "Unknown message type"
            prev_yaw = yaw
            n += 1
    print('wrote ' + str(n) + ' imu messages to the file: ' + out_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    Extracts IMU messages from bagfile.
    ''')
    parser.add_argument('bag', help='Bagfile')
    parser.add_argument('topic', help='Topic')
    parser.add_argument('--msg_type', default='PoseStamped',
                        help='message type')
    parser.add_argument('--output', default='stamped_poses.txt',
                        help='output filename')
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.bag))
    out_fn = os.path.join(out_dir, args.output)

    print('Extract pose from bag '+args.bag+' in topic ' + args.topic)
    print('Saving to file '+out_fn)
    extract(args.bag, args.topic, args.msg_type, out_fn)
