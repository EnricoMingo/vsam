#!/usr/bin/env python
import rospy
from sensor_msgs.msg import JointState

def postural_pub():
    pub = rospy.Publisher('cartesian/Postural/reference', JointState, queue_size=10)
    rospy.init_node('vsam_postural_publisher', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    print "Start publishing postural..."
    while not rospy.is_shutdown():
        posture = JointState()
	posture.name = ['VIRTUALJOINT_1', 'VIRTUALJOINT_2', 'VIRTUALJOINT_3', 'VIRTUALJOINT_4', 'VIRTUALJOINT_5', 'VIRTUALJOINT_6', 
			'LHipSag', 'LHipLat', 'LHipYaw', 'LKneeSag', 'LAnkLat', 'LAnkSag', 
			'RHipSag', 'RHipLat', 'RHipYaw', 'RKneeSag', 'RAnkLat', 'RAnkSag', 
			'WaistLat', 'WaistSag', 'WaistYaw', 
			'LShSag', 'LShLat', 'LShYaw', 'LElbj', 'LForearmPlate', 'LWrj1', 'LWrj2', 
			'RShSag', 'RShLat', 'RShYaw', 'RElbj', 'RForearmPlate', 'RWrj1', 'RWrj2']
	posture.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
			   -0.45, 0.0, 0.0, 0.9, 0.0, -0.45, 
			   -0.45, 0.0, 0.0, 0.9, 0.0, -0.45, 
			    0.0, 0.0, 0.0, 
			    0.5, 0.5, -0.3, -1.57, 0.0, 0.0, 0.0, 
			    0.5, -0.5, 0.3, -1.57, 0.0, 0.0, 0.0]

	pub.publish(posture)
        rate.sleep()

if __name__ == '__main__':
    try:
        postural_pub()
    except rospy.ROSInterruptException:
        pass










