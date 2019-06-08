from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import datetime
import rospy


class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        PATH_TO_MODEL = r'light_classification/model/ssd_incp_v2.pb'

        self.graph = tf.Graph()
        self.threshold = .5

        with self.graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                graph_def.ParseFromString(fid.read())
                tf.import_graph_def(graph_def, name='')

            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.graph.get_tensor_by_name('num_detections:0')

        self.sess = tf.Session(graph=self.graph)        


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        img_expand = np.expand_dims(image, axis=0)        
        with self.graph.as_default():
            (boxes, scores, classes, num_detections) = self.sess.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict={self.image_tensor: img_expand})

        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)


        if scores[0] > self.threshold:
            if classes[0] == 1:
                #rospy.logwarn("RETURN -- GREEN")
                return TrafficLight.GREEN
            elif classes[0] == 2:
                #rospy.logwarn("RETURN -- RED")                
                return TrafficLight.RED
            elif classes[0] == 3:
                #rospy.logwarn("RETURN -- YELLOW")
                return TrafficLight.YELLOW        
        #rospy.logwarn("RETURN -- UNK")                
        return TrafficLight.UNKNOWN
