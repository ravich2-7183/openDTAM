#!/usr/bin/env python
import os, sys
import csv
import rospy
import tf
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError 


usage = """
$ ./blender_play_on_msg.py /path/to/blender_images/ /path/to/blender_camera_poses.csv

Starts a ros node that listens for Bool messages on the /img_processed topic, and in response publishes an Image message on the /camera/image_raw topic and the corresponding pose on /tf/blender_world; /tf/blender_camera, reading images from the folder supplied as the first argument, and poses from the csv file supplied as the 2nd argument. 

csv file format (camera poses)
frame#, x, y, z, phi, theta, psi
"""


class BlenderROSPlay(object):
    def __init__(self, blender_images_dir, blender_camera_poses_csv):
        rospy.init_node('blender_play', anonymous=True)
        
        self.image_pub = rospy.Publisher('/camera/image_raw', Image, queue_size=10)
        self.tf_broadcaster = tf.TransformBroadcaster()
        
        self.bridge = CvBridge()
        
        self.img_filenames = self.read_image_filenames(blender_images_dir)
        self.blender_images_dirname = blender_images_dir
        self.poses = self.read_poses(blender_camera_poses_csv)
        assert(len(self.img_filenames) == len(self.poses))
        self.data_length = len(self.poses)
        
        self.counter =  1
        self.incr    = +1
        
        self.listener()
    
    def read_image_filenames(self, images_dirname):
        img_filenames_list = os.listdir(images_dirname)
        
        img_filenames = {}
        for i in range(len(img_filenames_list)):
            img_filenames[int(os.path.splitext(img_filenames_list[i])[0])] = img_filenames_list[i]
        
        return img_filenames
    
    def read_poses(self, poses_filename):
        poses = {}
        with open(poses_filename, 'rb') as f:
            reader = csv.reader(f)
            for row in reader:
                poses[int(row[0])] = [float(x) for x in row[1:]]
        return poses
    
    def blender_play_callback(self, data):
        i = self.counter
        
        im = cv2.imread(os.path.join(self.blender_images_dirname, self.img_filenames[i]))
        img_msg = self.bridge.cv2_to_imgmsg(im, "bgr8")
        img_msg.header.stamp = rospy.Time.now()
        self.image_pub.publish(img_msg)
        
        pose = self.poses[i]
        x, y, z, phi, theta, psi = self.poses[i]
        self.tf_broadcaster.sendTransform((x, y, z),
                                          tf.transformations.quaternion_from_euler(phi, theta, psi), # check that axes sequence is correct
                                          rospy.Time.now(),
                                          "/blender/camera",
                                          "/blender/world")
        
        self.incr = +1 if(self.incr == +1 and self.counter < self.data_length) else -1
        self.incr = -1 if(self.incr == -1 and self.counter > 1) else +1
        self.counter += self.incr
    
    def listener(self):
        rospy.Subscriber("img_processed", Bool, self.blender_play_callback)
        rospy.spin()

if __name__ == '__main__':
    blender_images_dir = sys.argv[1]
    blender_camera_poses_csv = sys.argv[2]
    
    BlenderROSPlay(blender_images_dir, blender_camera_poses_csv)

