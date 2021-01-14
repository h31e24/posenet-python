import tensorflow as tf
import cv2
import time
import argparse
import numpy as np

import json
import bottle
import gevent
from bottle.ext.websocket import GeventWebSocketServer
from bottle.ext.websocket import websocket
from multiprocessing import Pool
from PIL import Image
from io import BytesIO
import base64

import posenet

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()

socket = bottle.Bottle()
@socket.route('/', apply=[websocket])



def wesbin(ws):
  while True:
    with tf.Session() as sess:
        try:
            #decode to image
            img_str = ws.receive()
            decimg = base64.b64decode(img_str.split(',')[1], validate=True)
            decimg = Image.open(BytesIO(decimg))
            decimg = np.array(decimg, dtype=np.uint8); 
            decimg = cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)

            #############your process###############

            cap = cv2.Canny(decimg,100,200)
            #out_img = decimg
            cap.set(3, args.cam_width)
            cap.set(4, args.cam_height)



            model_cfg, model_outputs = posenet.load_model(args.model, sess)
            output_stride = model_cfg['output_stride']
            
            start = time.time()
            frame_count = 0


            while True:
              input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)

              heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                  model_outputs,
                  feed_dict={'image:0': input_image}
              )

              pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                  heatmaps_result.squeeze(axis=0),
                  offsets_result.squeeze(axis=0),
                  displacement_fwd_result.squeeze(axis=0),
                  displacement_bwd_result.squeeze(axis=0),
                  output_stride=output_stride,
                  max_pose_detections=10,
                  min_pose_score=0.15)

              keypoint_coords *= output_scale

              # TODO this isn't particularly fast, use GL for drawing and display someday...
              overlay_image = posenet.draw_skel_and_kp(
                  display_image, pose_scores, keypoint_scores, keypoint_coords,
                  min_pose_score=0.15, min_part_score=0.1)

              cv2.imshow('posenet', overlay_image)
              frame_count += 1
              if cv2.waitKey(1) & 0xFF == ord('q'):
                  break


            #############your process###############

            #encode to string
            encimg = cv2.imencode(".jpg", cap, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            img_str = encimg.tostring()
            img_str = "data:image/jpeg;base64," + base64.b64encode(img_str).decode('utf-8')
            ws.send(img_str)
        except:
            pass
            #print("error")


if __name__ == "__main__":
    # get ngrok url
    f = open("url.txt", "r")
    url = f.read()
    f.close()
    url = "wss" + url[5:]
    # prepare multiprocess
    _pool = Pool(processes=2)
    _pool.apply_async(use_cam, (url, 0.8))
    socket.run(host='0.0.0.0', port=6006, server=GeventWebSocketServer)

