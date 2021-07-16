#!/usr/bin/env python

import rospy
from copy import copy
from opensot_visual_servoing.msg import VisualFeatures

def callback(data):
    refs = copy(data)

    for point in refs.features:
        point.x = point.x - 0.6

    pub = rospy.Publisher("/cartesian/visual_servoing_D435i_camera_color_optical_frame/desired_features", VisualFeatures, queue_size=10)
    rate = rospy.Rate(10)  # 10hz
    for i in range(5):
        pub.publish(refs)
        rate.sleep()

    rospy.signal_shutdown("new reference were published")

def publisher():
    rospy.init_node('reference_points_pub', anonymous=True)

    rospy.Subscriber("/cartesian/visual_servoing_D435i_camera_color_optical_frame/reference_features", VisualFeatures, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    publisher()