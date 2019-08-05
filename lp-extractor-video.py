import sys, os
import cv2
import keras
import numpy as np
import traceback
import time

import darknet.python.darknet as dn

from os 					import makedirs
from os.path 				import splitext, basename, isdir
from darknet.python.darknet import detect
from glob 					import glob
from src.keras_utils 		import load_model, detect_lp
from src.label 				import Label, lwrite, Shape, writeShapes, dknet_label_conversion
from src.utils 				import crop_region, image_files_from_folder, im2single, nms


if __name__ == '__main__':
	try:
		input_video  = sys.argv[1]
		output_dir = sys.argv[2]
		seconds = 2 # change this to change the how often to grab image

		vehicle_threshold = .5

		vehicle_weights = 'data/vehicle-detector/yolo-voc.weights'
		vehicle_netcfg  = 'data/vehicle-detector/yolo-voc.cfg'
		vehicle_dataset = 'data/vehicle-detector/voc.data'

		vehicle_net  = dn.load_net(vehicle_netcfg, vehicle_weights, 0)
		vehicle_meta = dn.load_meta(vehicle_dataset)

		##########################################
		# Originally from lp-detector
		wpod_net = load_model('data/lp-detector/wpod-net_update1.h5')
		##########################################

		##########################################
		# Originally from lp-ocr
		ocr_weights = 'data/ocr/ocr-net.weights'
		ocr_netcfg  = 'data/ocr/ocr-net.cfg'
		ocr_dataset = 'data/ocr/ocr-net.data'

		ocr_net  = dn.load_net(ocr_netcfg, ocr_weights, 0)
		ocr_meta = dn.load_meta(ocr_dataset)
		##########################################

		if not isdir(output_dir):
			makedirs(output_dir)

		cap = cv2.VideoCapture(input_video)
		# cap = cv2.VideoCapture('samples2/input/IMG_2246.mov')
		fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
		multiplier = fps * seconds
		# print 'Video has', fps, 'fps'

		while(cap.isOpened()):	
			frameId = int(round(cap.get(1))) #current frame number
			ret, frame = cap.read()

			# print frameId, 'frame'

			if frameId % multiplier == 0:
				start = time.time()
				frame_path = '%s/video_frame%d.png' % (output_dir,frameId)
				cv2.imwrite(frame_path,frame)

				# detect_cars_start = time.time()
				R,_ = detect(vehicle_net, vehicle_meta, frame_path ,thresh=vehicle_threshold)
				# print time.time() - detect_cars_start, 'seconds to find cars'

				R = [r for r in R if r[0] in ['car','bus']]

				if len(R):
					Iorig = cv2.imread(frame_path)
					WH = np.array(Iorig.shape[1::-1],dtype=float)
					Lcars = []

					for i,r in enumerate(R):
						cx,cy,w,h = (np.array(r[2])/np.concatenate( (WH,WH) )).tolist()
						tl = np.array([cx - w/2., cy - h/2.])
						br = np.array([cx + w/2., cy + h/2.])
						label = Label(0,tl,br)
						Icar = crop_region(Iorig,label)
						Icar = Icar.astype('uint8')

						################################################
						# CODE FROM license-plate-detection

						lp_threshold = .5

						ratio = float(max(Icar.shape[:2]))/min(Icar.shape[:2])
						side  = int(ratio*288.)
						bound_dim = min(side + (side%(2**4)),608)

						Llp,LlpImgs,_ = detect_lp(wpod_net,im2single(Icar),bound_dim,2**4,(240,80),lp_threshold)

						if len(LlpImgs):
							Ilp = LlpImgs[0]
							Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
							Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

							s = Shape(Llp[0].pts)

							img_path = '%s/video_frame_lp%d_%d.png' % (output_dir, frameId, i)
							cv2.imwrite(img_path,Ilp*255.)

							################################################
							# CODE FROM license-plate-ocr

							ocr_threshold = .4

							R,(width,height) = detect(ocr_net, ocr_meta, img_path ,thresh=ocr_threshold, nms=None)

							if len(R):
								L = dknet_label_conversion(R,width,height)
								L = nms(L,.45)

								L.sort(key=lambda x: x.tl()[0])
								lp_str = ''.join([chr(l.cl()) for l in L])

								print 'Found License Plate:', lp_str, 'on frame', frameId

							# CODE FROM license-plate-ocr
							################################################

						# CODE FROM license-plate-detection
						################################################
				print time.time() - start, 'seconds to process frame'
		cap.release()
	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)
	