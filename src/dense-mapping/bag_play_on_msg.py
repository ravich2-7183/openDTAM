#!/usr/bin/env python
import rospy, rosbag
import sys
from std_msgs.msg import Bool
from sensor_msgs.msg import Image

usage = """Listens for Bool messages on the /img_processed topic, and in response publishes an Image message on the /camera/image_raw topic, reading from the bag file supplied as the first argument.
E.g.: $ ./bag_play_on_msg.py video_stream.bag"""

def callback(data):
    global bag_generator
    global image_pub
    
    while True:
        topic, msg, time = bag_generator.next()
        if topic == "/camera/image_raw":
            break
    
    msg.header.stamp = rospy.get_rostime()
    image_pub.publish(msg)

def listener():
    rospy.Subscriber("img_processed", Bool, callback)
    rospy.spin()

if __name__ == '__main__':
    global bag_generator
    global image_pub
    
    image_pub = rospy.Publisher('/camera/image_raw', Image, queue_size=10)
    
    rospy.init_node('bag_pub', anonymous=True)
    
    bag_file_name = sys.argv[1]
    bag = rosbag.Bag(bag_file_name)
    bag_generator = bag.read_messages()
    print "Finished reading bag messages. Ready to publish on receipt of img_processed messages..."
    
    listener()

