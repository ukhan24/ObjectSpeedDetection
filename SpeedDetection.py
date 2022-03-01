# References
# https://imageai.readthedocs.io/en/latest/video/
# https://github.com/OlafenwaMoses/ImageAI/blob/master/imageai/Detection/VIDEO.md
# https://readthedocs.org/projects/imageai/downloads/pdf/latest/
# https://answers.opencv.org/question/194295/compare-video-frames/
# https://medium.com/@igorirailean/dense-optical-flow-with-python-using-opencv-cb6d9b6abcaf

import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from imageai.Detection import VideoObjectDetection

global height
global weight
global old_frame
global difference_frame

# *** Functions ***
#
def forFrame(frame_number, output_array, output_count, returned_frame):
    ''''''
    returned_frame = np.asarray(returned_frame, dtype=np.uint8)
    gray_frame = cv2.cvtColor(returned_frame, cv2.COLOR_BGR2GRAY)
    #gray_old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    if old_frame is not None:
        
        # Calculate optical flow (motion estimation) between frames
        flow = cv2.calcOpticalFlowFarneback(old_frame, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # disp_frame = np.uint8(255.0*flow/float(flow.max()))
        # disp = Image.fromarray(disp_frame)
        # display(disp)
        
        # Computes the magnitude and angle of the 2D vectors 
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1]) 
        
        # Sets image hue according to the optical flow direction 
        mask[..., 0] = angle * 180 / np.pi / 2
        
        # Sets image value according to the optical flow 
        # magnitude (normalized) 
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX) 
        
        # Converts HSV to RGB (BGR) color representation
        '''
        mask_zero = np.uint8(255.0*mask[..., 0]/float(mask[..., 0].max()))
        mask_one = np.uint8(255.0*mask[..., 1]/float(mask[..., 1].max()))
        mask_two = np.uint8(255.0*mask[..., 2]/float(mask[..., 2].max()))
        '''
    
        mask_image = Image.fromarray(mask, 'RGB')
        # display(mask_image)
        
        '''
        rgb = cv2.cvtColor(mask_image, cv2.COLOR_HSV2BGR) 
        
        disp_frame = np.uint8(255.0*mask_image/float(mask_image.max()))
        disp = Image.fromarray(disp_frame)
        display(disp)
        
        # difference_frame = gray_frame - old_frame
        # difference_frame -= difference_frame.min()
        
        # disp_frame = np.uint8(255.0*difference_frame/float(difference_frame.max()))
        # disp = Image.fromarray(disp_frame)
        # display(disp)
                  
    # print(old_frame)
    # old_frame = cv2.cvtColor(old_frame_converted, cv2.COLOR_RGB2GRAY)
    # print(type(old_frame_converted))
    # print(old_frame_converted.shape)    
    # old_frame = gray_frame
    '''

    # Set previous frame to current (returned) frame
    old_frame_converted = Image.fromarray(old_frame, 'RGB')

    '''
    if old_frame is None: # & frame_number == 1:
        # old_frame = np.zeros([1080, 1920])
        print(old_frame)
       
    elif old_frame is not None & frame_number > 1:
        difference_frame = gray_frame - old_frame
        difference_frame -= difference_frame.min()
        # disp_frame = np.uint8(255.0*difference_frame/float(difference_frame.max()))
        # disp = Image.fromarray(disp_frame)
        # display(disp)
        # cv2.imshow('diff_frame',disp_frame)
          
    old_frame = gray_frame
    print("\n")
    print(old_frame)
    '''
    
    # for i in output_array:
    #     print(i.get('name'))
    #     print(i.get('box_points'))
    
    print("\n")
    ''''''


# ******* Main *******
#
# Get current working directory
execution_path = os.getcwd()

# Set video capture stream to play mp4 video(s)
camera = cv2.VideoCapture("carsTest2.mp4")

# Get video resolution (frame height and width)
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))

# Initially set previous and difference frames to null (zeroes)
old_frame = np.zeros([height, width], dtype = int)
difference_frame = np.zeros([height, width], dtype = int)

# Create mask array to get hue and values from HSV image frames
mask = np.zeros([height, width, 3], dtype = int)  
mask[..., 1] = 255


# ***** RetinaNet Detection *****
print("Starting RetinaNet Video Detection\n")

# Create video object detector. Furthermore, set detector's detection model (using RetinaNet) and path, and then load the detection model (detector)
detector = VideoObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

# Detect objects (per frame) while video is playing. For each frame, function forFrame is being called to detect detected objects' speed 
video_path = detector.detectObjectsFromVideo(camera_input=camera, 
                                             output_file_path=os.path.join(execution_path, "RetinaNet-Detection"), 
                                             log_progress=True, 
                                             detection_timeout=10, 
                                             return_detected_frame=True,
                                             per_frame_function=forFrame)

# Stop the video/detection after video is complete
camera.release()
print(video_path)
print("\nRetinaNet Detection Completed\n")

'''
# ***** YOLOv3 Detection *****
print("\nStarting YOLOv3 Video Detection\n")

# Set video capture stream to play mp4 video(s)
camera = cv2.VideoCapture("carsTest2.mp4")

# Create video object detector. Furthermore, set detector's detection model (using RetinaNet) and path, and then load the detection model (detector)
detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path , "yolo.h5"))
detector.loadModel()

# Detect objects (per frame) while video is playing. For each frame, function forFrame is being called to detect detected objects' speed 
video_path = detector.detectObjectsFromVideo(camera_input=camera, 
                                             output_file_path=os.path.join(execution_path, "YOLOv3-Detection"), 
                                             log_progress=True, 
                                             detection_timeout=10, 
                                             return_detected_frame=True,
                                             per_frame_function=forFrame)

# Stop the video/detection after video is complete
camera.release()
print(video_path)
print("\nYOLOv3 Detection Completed\n")


# ***** Tiny-YOLOv3 Detection *****
print("\nStarting Tiny-YOLOv3 Video Detection\n")

# Set video capture stream to play mp4 video(s)
camera = cv2.VideoCapture("carsTest2.mp4")

# Create video object detector. Furthermore, set detector's detection model (using RetinaNet) and path, and then load the detection model (detector)
detector = VideoObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(os.path.join(execution_path , "yolo-tiny.h5"))
detector.loadModel()

# Detect objects (per frame) while video is playing. For each frame, function forFrame is being called to detect detected objects' speed 
video_path = detector.detectObjectsFromVideo(camera_input=camera, 
                                             output_file_path=os.path.join(execution_path, "Tiny-YOLOv3-Detection"), 
                                             log_progress=True, 
                                             detection_timeout=10, 
                                             return_detected_frame=True,
                                             per_frame_function=forFrame)

# Stop the video/detection after video is complete
camera.release()
print(video_path)
print("\nTiny-YOLOv3 Detection Completed\n")
'''




#******************************* ONLINE EXAMPLE - NOT PART OF CODE *******************************
#
'''
# The video feed is read in as 
# a VideoCapture object 
cap = cv2.VideoCapture("carsTest2.mp4") 
  
# ret = a boolean return value from 
# getting the frame, first_frame = the 
# first frame in the entire video sequence 
ret, first_frame = cap.read() 

# Converts frame to grayscale because we 
# only need the luminance channel for 
# detecting edges - less computationally  
# expensive 
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY) 
 
# Creates an image filled with zero 
# intensities with the same dimensions  
# as the frame 
mask = np.zeros_like(first_frame) 
  
# Sets image saturation to maximum 
mask[..., 1] = 255
  
while(cap.isOpened()): 
      
    # ret = a boolean return value from getting 
    # the frame, frame = the current frame being 
    # projected in the video 
    ret, frame = cap.read() 
      
    # Opens a new window and displays the input 
    # frame 
    # cv2.imshow("input", frame) 
      
    # Converts each frame to grayscale - we previously  
    # only converted the first frame to grayscale 
    # gray = gray.astype('uint8')
    frame = np.asarray(frame, dtype=np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
      
    # Calculates dense optical flow by Farneback method 
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,  
                                       None, 
                                       0.5, 3, 15, 3, 5, 1.2, 0) 
      
    # Computes the magnitude and angle of the 2D vectors 
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1]) 
      
    # Sets image hue according to the optical flow  
    # direction 
    mask[..., 0] = angle * 180 / np.pi / 2
      
    # Sets image value according to the optical flow 
    # magnitude (normalized) 
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX) 
      
    # Converts HSV to RGB (BGR) color representation 
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR) 
      
    # Opens a new window and displays the output frame 
    # cv2.imshow("dense optical flow", rgb) 
      
    # Updates previous frame 
    prev_gray = gray 
      
    # Frames are read by intervals of 1 millisecond. The 
    # programs breaks out of the while loop when the 
    # user presses the 'q' key 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# The following frees up resources and 
# closes all windows 
cap.release() 
cv2.destroyAllWindows() 
'''