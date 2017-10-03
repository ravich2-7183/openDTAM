#!/usr/bin/env python
import rospy, rosbag
import cv2
import sys
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

usage = """Listens for Bool messages on the /img_processed topic, and in response publishes an Image message on the /camera/image_raw topic, reading from the video file supplied as the first argument. 
E.g.: $ ./video_play_on_msg.py video.mp4"""

def callback(data):
    global image_pub
    global bridge
    global video_file
    
    ret, frame = video_file.read()
    frame = cv2.resize(frame, (640, 480))
    
    img_msg              = bridge.cv2_to_imgmsg(frame, "bgr8")
    img_msg.header.stamp = rospy.get_rostime()
    
    image_pub.publish(img_msg)


def listener():
    rospy.Subscriber("img_processed", Bool, callback)
    rospy.spin()


if __name__ == '__main__':
    global image_pub
    global bridge
    global video_file
    
    rospy.init_node('video_pub', anonymous=True)
    
    image_pub  = rospy.Publisher('/camera/image_raw', Image, queue_size=10)
    bridge     = CvBridge()
    video_file = cv2.VideoCapture(sys.argv[1])
    
    if video_file.isOpened():
        print "Video file successfully opened. Ready to publish frames, on receipt of img_processed messages..."
    
    listener()

