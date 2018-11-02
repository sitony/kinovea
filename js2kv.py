import json
import numpy as np
import cv2
import math
import argparse
from collections import OrderedDict
import glob
import sys
from lxml import etree
import copy

"""
OpenPose BODY_25
{0,  "Nose"},     {1,  "Neck"},     
{2,  "RShoulder"},     {3,  "RElbow"},     {4,  "RWrist"},
{5,  "LShoulder"},     {6,  "LElbow"},     {7,  "LWrist"},     
{8,  "MidHip"},     
{9,  "RHip"},     {10, "RKnee"},     {11, "RAnkle"},
{12, "LHip"},     {13, "LKnee"},     {14, "LAnkle"},
{15, "REye"},     {16, "LEye"},     {17, "REar"},     {18, "LEar"},
{19, "LBigToe"},     {20, "LSmallToe"},     {21, "LHeel"},
{22, "RBigToe"},     {23, "RSmallToe"},     {24, "RHeel"},
{25, "Background"}
"""

# Name of joints
body25_names = ['Nose','Neck','RShoulder','RElbow','RWrist','LShoulder','LElbow','LWrist',
'MidHip','RHip','RKnee','RAnkle','LHip','LKnee','LAnkle','REye','LEye',
'REar','LEar','LBigToe','LSmallToe','LHeel','RBigToe','RSmallToe','RHeel','Background']

# Pairs of bones
body25_pairs = [(1,8),(1,2),(1,5),(2,3),(3,4),(5,6),(6,7),(8,9),(9,10),(10,11),(8,12),(12,13), 
(13,14),(1,0), (0,15), (15,17), (0,16), (16,18), (14,19),(19,20),(14,21),(11,22),(22,23),(11,24)]

# Colors of bones
body25_colors = [(255.,0.,85.), (255.,0.,0.), (255.,85.,0.), (255.,170.,0.), (255.,255.,0.), 
(170.,255.,0.), ( 85.,255.,0.), (  0.,255.,0.), (255.,0.,0.), (  0.,255.,85.), (  0.,255.,170.), 
(  0.,255.,255.), (  0.,170.,255.), (  0.,85.,255.), (  0.,0.,255.), (255.,0.,170.), 
(170.,0.,255.), (255.,0.,255.), ( 85.,0.,255.), (  0.,0.,255.), (  0.,0.,255.), (  0.,0.,255.), 
(  0.,255.,255.), (  0.,255.,255.), (  0.,255.,255.)]

body25_colors_mpl = [(1.,0.,0.33),(1.,0.,0.),(1.,0.33,0.),(1.,0.67,0.),(1.,1.,0.),(0.67,1.,0.),
(0.33,1.,0.),(0.,1.,0.),(1.,0.,0.),(0.,1.,0.33),(0.,1.,0.67),(0.,1.,1.),(0.,0.67,1.),(0.,0.33,1.),
(0.,0.,1.),(1.,0.,0.67),(0.67,0.,1.),(1.,0.,1.),(0.33,0.,1.),(0.,0.,1.),(0.,0.,1.),(0.,0.,1.),
(0.,1.,1.),(0.,1.,1.),(0.,1.,1.)]

# -----------------------------------------------------------------------------



'''
Using OpenPose

Background information: 
 "OpenPose" is Real-time multi-person keypoint detection library for body, face, hands, and foot estimation
 OpenPose is open source for research purposes.
 https://github.com/CMU-Perceptual-Computing-Lab/openpose
 Keypoint JSON file, pose model with 25 joints, can be output in frame-by-frame. 

Although there is a problem of accuracy, it is still useful if you can read the OpenPose output files with Kinovea.
I made a prototype script to convert from OpenPose JSON files to Kinovea XML file, so I will share it.

Usage:

1. Making OpenPose Keypoint JSON files
Download OpenPose from https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases. I used Windows CPU version binary.
Run an example command with "-write_json [Output directory]".
> bin\OpenPoseDemo.exe --video examples\media\video.avi -write_json .\output
CPU version is very slow. "GPU" version is faster ten or hundred times.
"examples\media\video.avi" has 205 frames. 205 JSON files will be there.

2. Converst from OpenPose JSON to Kinovea XML
Download conversion program "js2kv.py" from https://github.com/sitony/kinovea.

Run js2kv.py convert program.
> python js2kv.py --json media\output\*.json --video media\video.avi --target 640 360 --output output.kva


'''






# js2kv.py
# python jk5.py --json media\output\*.json --video media\video.avi --target 640 360 --output output.kva


parser = argparse.ArgumentParser(description='Convert from OpenPose JSON to Kinovea KVA   (v20181103)')
parser.add_argument('--json', type=str, required=True, metavar='files', help='JSON files (e.g. ./output/*.json)')
parser.add_argument('--video', type=str, required=True, metavar='file', help='Video file (e.g. ./running.mp4')
parser.add_argument('--target', type=int, required=True, nargs=2, metavar='axis', help='X Y coordinate of target head position (e.g. 320 160)')
parser.add_argument('--output', type=str, metavar='file', help='output KVA file (e.g. output.kva)')
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Loading JSON files to json_dict_list

json_file_list = sorted(glob.glob(args.json))
json_dict_list = []

# Load all json files in a folder
for fn in json_file_list:
    f = open(fn,'r')
    json_dict = json.load(f, object_pairs_hook=OrderedDict)
    json_dict_list.append(json_dict.copy())

if len(json_dict_list) < 1:
    print("Error: There are no JSON files. ", args.json)
    sys.exit()

print("JSON frame count: ", len(json_dict_list))

# -----------------------------------------------------------------------------
# Loading Video file

video_cap = cv2.VideoCapture(args.video)
if not video_cap.isOpened():
    print("Error: Unable to open video source. ", args.video)
    sys.exit()

video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_center = [video_width / 2, video_height / 2]
video_fps = video_cap.get(cv2.CAP_PROP_FPS)
video_framecount = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
video_cap.release()

print("Video:", args.video, ", Width:",video_width,", Height:", video_height, ", FPS:", video_fps, ", frames:", video_framecount)

if video_width < args.target[0] or video_height < args.target[1]:
    print("Error: The coordinates are over than video size. ", args.target)
    sys.exit()

# -----------------------------------------------------------------------------
# Target tracking

# The coordinates, closest to the "target", are selected from the "points" coordinate list.
def serch_neighbourhood(target, points):
    pt = np.array(target)
    pls = np.array(points)
    lst = np.array([])
    for i in range(pls.shape[0]):
        lst = np.append(lst, np.linalg.norm(pls[i]-pt))
    return pls[np.argmin(lst)].tolist(), np.argmin(lst)

first_coord = args.target # Set target position

# Tracking the target in all frames
person_tracked = []
person_target_coord_tracked = []
for frame in range(len(json_dict_list)):
    peoples_neck = []
    for p in range(len(json_dict_list[frame]['people'])):
        peoples_neck.append(json_dict_list[frame]['people'][p]['pose_keypoints_2d'][3:5])
    if frame == 0:
        pcc, pc  = serch_neighbourhood(first_coord, peoples_neck) # first frame
    else:
        pcc, pc  = serch_neighbourhood(person_target_coord_tracked[frame-1], peoples_neck)
    person_tracked.append(pc)
    person_target_coord_tracked.append(pcc)

#print(person_tracked)
#print(len(person_target_coord_tracked))

# -----------------------------------------------------------------------------
# Making XML for Kinovea

minimal_kva = etree.XML('''<KinoveaVideoAnalysis>
  <FormatVersion>2.0</FormatVersion>
  <Producer>OpenPoseJSON2Kinovea.0.1</Producer>
  <OriginalFilename>MovieFile.avi</OriginalFilename>
  <ImageSize>640;480</ImageSize>
  <AverageTimeStampsPerFrame>1</AverageTimeStampsPerFrame>
  <CaptureFramerate>29.97</CaptureFramerate>
  <UserFramerate>29.97</UserFramerate>
  <FirstTimeStamp>0</FirstTimeStamp>
  <SelectionStart>0</SelectionStart>
  <Tracks>

  </Tracks>
  <Trackability />
</KinoveaVideoAnalysis>''')

track_kva = etree.XML('''<Track name="Trajectory">
        <TimePosition>0</TimePosition>
        <Mode>Label</Mode>
        <ExtraData>Name</ExtraData>
        <Marker>Cross</Marker>
        <DisplayBestFitCircle>false</DisplayBestFitCircle>
    <TrackPointList Count="1" UserUnitLength="px">
    </TrackPointList>
    <MainLabel Text="Trajectory">
        <SpacePosition>100;100</SpacePosition>
        <TimePosition>0</TimePosition>
    </MainLabel>
    <DrawingStyle>
        <Color Key="color">
            <Value>255;255;140;0</Value>
        </Color>
        <LineSize Key="line size">
            <Value>2</Value>
        </LineSize>
        <TrackShape Key="track shape">
            <Value>Solid;false</Value>
        </TrackShape>
    </DrawingStyle>
</Track>''')

# Video file data
next(minimal_kva.iter("OriginalFilename")).text = args.video
next(minimal_kva.iter("ImageSize")).text = str(video_width)+";"+str(video_height)
next(minimal_kva.iter("CaptureFramerate")).text = str(video_fps)
next(minimal_kva.iter("UserFramerate")).text = str(video_fps)

# Convert joint coordinates to KVA-XML format and embed it in basexml
def make_tracks(basexml, person_tracked, first_frame):
    trackpoints=[]
    for joint in range(25): # BODY25
        trackpoints.append(copy.deepcopy(basexml))
        next(trackpoints[joint].iter("Track")).set("name", body25_names[joint])
        next(trackpoints[joint].iter("TimePosition")).text = str(first_frame)
        next(trackpoints[joint].iter("MainLabel")).set("Text", body25_names[joint])
        next(trackpoints[joint].iter("TrackPointList")).set("Count", str(len(person_tracked)))
        trackpoints[joint].find("*/Color/Value").text = "250" +";"+ str(int(body25_colors[joint][0])) +";"+ str(int(body25_colors[joint][1])) +";"+ str(int(body25_colors[joint][2])) # Opacity,R,G,B
        for frame in range(len(person_tracked)):
            if frame >= first_frame:
                person_pose = np.array(json_dict_list[frame]['people'][person_tracked[frame]]['pose_keypoints_2d']).reshape((25, 3))  # BODY_25[x,y,score] * frames
                joint_coord = tuple(list(map(int, person_pose[joint, 0:2])))
                confidence_score = person_pose[joint, 2]
                #print(joint, ":", joint_coord, confidence_score)
                if confidence_score == 0 and frame != 0:
                    # If the confidence score is zero, fill it with the previous frame coordinates.
                    last_coord = (trackpoints[joint].findall("*/TrackPoint")[-1]).text
                    etree.SubElement(next(trackpoints[joint].iter("TrackPointList")), "TrackPoint").text = last_coord
                else:
                    etree.SubElement(next(trackpoints[joint].iter("TrackPointList")), "TrackPoint").text = str(joint_coord[0]) +";"+ str(joint_coord[1]) +";"+ str(frame)
                if frame == 0:
                    if joint_coord[0] > 50 and joint_coord[1] > 50:
                        next(trackpoints[joint].iter("SpacePosition")).text = str(joint_coord[0]-20) +";"+ str(joint_coord[1]-30)
    return trackpoints

tracklist = make_tracks(track_kva, person_tracked, 0)

# Merge minimal_kva and track points
for tp in tracklist:
    next(minimal_kva.iter("Tracks")).append(tp)

# Output kva text
kva_str = etree.tostring(minimal_kva, method="xml", encoding="unicode", pretty_print=True)


# Save kva file
with open(args.output, 'w') as f:
    f.write(kva_str)
    print("Output:", args.output)

sys.exit()

