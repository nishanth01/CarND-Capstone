#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
import math
import numpy as np
from scipy.spatial import KDTree

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.
As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.
Once you have created dbw_node, you will update this node to use the status of traffic lights too.
Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.
TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 100 
FREQUENCY  = 10
MAX_DECEL_LIMIT = 0.5


class WaypointUpdater(object):

    def __init__(self):

        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.waypoints_publisher = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoints_tree = None
        self.stop_line_idx = -1

        self.loop()


    # This is our main loop which is run at a set interval
    def loop(self):
        rate = rospy.Rate(FREQUENCY)
        while not rospy.is_shutdown():
            if self.pose and self.waypoints_tree: 
                lane = self.generate_lane()
                self.waypoints_publisher.publish(lane) 
            rate.sleep()


    def generate_lane(self):
    
        lane = Lane()
        start_index = self.closest_waypoint()
        end_index = start_index + LOOKAHEAD_WPS
        lane_waypoints = self.base_waypoints[start_index:end_index]

        if (self.stop_line_idx == -1) or (self.stop_line_idx > end_index):
            #rospy.logwarn("DRIVE")
            lane.waypoints =  lane_waypoints              
        else:
            #rospy.logwarn("STOP")
            lane.waypoints = self.decelerate(lane_waypoints, start_index)

        return lane


    def decelerate(self, waypoints, start_index):

        processed_waypoints = []
        deceleration_rate  = MAX_DECEL_LIMIT

        for i, waypoint in enumerate(waypoints):
            p = Waypoint()
            p.pose = waypoint.pose

            stop_index = max(self.stop_line_idx - start_index - 2,0)           
            stop_distance = self.distance(waypoints, i, stop_index)

            if i >= stop_index:
                target_speed = 0
            elif stop_distance < 20:
                target_speed = math.sqrt(2 * deceleration_rate * stop_distance)
            else:
                target_speed = self.get_waypoint_velocity(waypoint)

            if(target_speed < 1.):
                target_speed = 0.    

            p.twist.twist.linear.x = min(target_speed, self.get_waypoint_velocity(waypoint))
            processed_waypoints.append(p)

        return processed_waypoints


    def closest_waypoint(self):

        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoints_tree.query([x, y], 1)[1]

        # Check if closest is ahead or behind vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]

        # Equation for hyperplane through closest_coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)

        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)

        return closest_idx


    def pose_cb(self, msg):
        self.pose = msg


    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints.waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x,waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoints_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.stop_line_idx = msg.data

    def velocity_cb(self, velocity):
        self.current_velocity = velocity.twist.linear.x

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')