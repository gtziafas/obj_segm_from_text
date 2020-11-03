# object_segmentation_from_text
Code for the final project of the RUG's course "Cognitive Robotics" (November 2019), based on: ["Zero-Shot Grounding of Objects from Natural Language Queries
"](https://arxiv.org/abs/1908.07129)

# demo
https://www.youtube.com/watch?v=kgQgaghf71o

# usage
- Be sure to edit paths inside files to your own static images, rosbags etc.
- The model weights are missing, too big of a file, follow instructions on the paper's [git page](https://github.com/TheShadow29/zsgnet-pytorch), our module will take care on how to implement inference
- Open 8 terminals: (steps 3-5 are equivalent, depending from where you get your visual data)
```
  # startup master
  $roscore
  
  # startup model_inference_pipeline for loading the neural net and utils
  $rosrun object_segmentation_from_text modeL_inference_pipeline.py
  
  # ONLY ONE of the following: define your visual input src (static images, rosbag in a loop, real-time data from
  # depth sensors e.g. Kinect)
  $rosrun oject_segmentation_from_text image_buffer_from_path.py file_path  
  $rosbag play -l bag_file 
  $roslaunch openni_launch openni_launch
  
  # setup visualization
  $rosrun rviz rviz 
  $rosrun tf static_transform_publisher 0 0 0 0 0 0 map camera_rgb_optical_frame 100
  
   # run this ONLY if you have visual input from RGB-D data
   # setup 3d extraction modules
  $rosrun object_segmentation_from_text 3d_image_buffer.py 
  $roslaunch object_segmentation_from_text 3d_extraction.launch
  
  # startup caption buffering
  $rosrun object_segmentation_from_text caption_buffer_from_console.py
 ```

# todo
- Wrap usage in a launch file
- Fix 3d_image_buffer node to correctly time-stamp buffed data
- Re-implement model_inference_pipeline as an action server instead of node for appripriately handling GPU memory
- Implement a speech2text module for online speech captioning instead of console input
