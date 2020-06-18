import numpy as np
import cv2
from scipy.spatial import distance as dist
import imutils
import os

def detect_humans(frame,model,layer,min_confidence,nms_threshold,person_idx=0):
	(h,w)=frame.shape[:2]
	results=[]

	blob=cv2.dnn.blobFromImage(frame,1/255.0,(416,416),swapRB=True,crop=False)
	model.setInput(blob)
	outputs=model.forward(layer)

	bboxes=[]
	centroids=[]
	confidence_scores=[]

	for output in outputs:
		for detection in output:
			scores=detection[5:]
			class_id=np.argmax(scores)
			confidence=scores[class_id]

			if class_id==person_idx and confidence>min_confidence:
				bbox=detection[0:4]*np.array([w,h,w,h])
				(centre_x,centre_y,width,height)=bbox.astype('int')
				x=int(centre_x-(width/2))
				y=int(centre_y-(height/2))
				bboxes.append([x,y,int(width),int(height)])
				centroids.append((centre_x,centre_y))
				confidence_scores.append(float(confidence))
	
	idxs=cv2.dnn.NMSBoxes(bboxes,confidence_scores,min_confidence,nms_threshold)
	if len(idxs)>0:
		for i in idxs.flatten():
			(x,y)=(bboxes[i][0],bboxes[i][1])
			(w,h)=(bboxes[i][2],bboxes[i][3])
			r=(confidence_scores[i],(x,y,x+w,y+h),centroids[i])
			results.append(r)
	return results

model_path='./yolo'
min_conf=0.3
nms_threshold=0.3
min_dist=50
input_video='pedestrians1.mp4'
output_file=os.getcwd()+'/output.avi'

labels_path=os.path.sep.join([model_path,'coco.names'])
labels=open(labels_path).read().strip().split('\n')
weights_path=os.path.sep.join([model_path,'yolov3.weights'])
config_path=os.path.sep.join([model_path,'yolov3.cfg'])

model=cv2.dnn.readNetFromDarknet(config_path,weights_path)
# model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layers=model.getLayerNames()
layer=[layers[i[0]-1] for i in model.getUnconnectedOutLayers()]
vs=cv2.VideoCapture(input_video)
writer=None
while (vs.isOpened()):
	(grabbed,frame)=vs.read()
	if not grabbed:
		break
	
	frame=imutils.resize(frame,width=700)
	results=detect_humans(frame,model,layer,min_conf,nms_threshold,person_idx=labels.index('person'))

	sd_violate=set()
	lines_dict={}
	if len(results)>=2:
		centroids=np.array([r[2] for r in results])
		d=dist.cdist(centroids,centroids,metric="euclidean")
		for i in range(0,d.shape[1]):
			for j in range(i+1,d.shape[1]):
				if d[i,j]<min_dist:
					sd_violate.add(i)
					sd_violate.add(j)
					try:
						lines_dict[i].append(j)
					except:
						lines_dict[i]=[]
						lines_dict[i].append(j)
	
	for(i,(prob,bbox,centroid)) in enumerate(results):
		(start_x,start_y,end_x,end_y)=bbox
		(c_x,c_y)=centroid
		colour=(0,255,0)
		if i in sd_violate:
			colour=(0,0,255)
		centre_ellipse=((start_x+end_x)//2,end_y)
		axes_length=(min_dist,min_dist//2)
		angle=0
		start_angle=0
		end_angle=360
		thickness=2

		cv2.circle(frame,(c_x,c_y-10),4,colour,-1)
		cv2.line(frame,(c_x,c_y-10),centre_ellipse,colour,thickness)
		cv2.ellipse(frame,centre_ellipse,axes_length,angle,start_angle,end_angle,colour,thickness)
		cv2.ellipse(frame,centre_ellipse,(8,4),angle,start_angle,end_angle,colour,-1)

	for key in lines_dict.keys():
		violaters=lines_dict[key]
		source_x,source_y=results[key][2]
		for i in range(len(violaters)):
			dest_x,dest_y=results[violaters[i]][2]
			cv2.line(frame,(source_x,source_y-10),(dest_x,dest_y-10),(0,0,255),2)
	
	if writer is None:
		writer=cv2.VideoWriter(output_file,cv2.VideoWriter_fourcc('M','J','P','G'),25,(frame.shape[1],frame.shape[0]),True)
		
	writer.write(frame)
	# cv2.imshow('frame',frame)
	# key=cv2.waitKey(1) & 0xFF
	# if key==ord('q'):
	#     break

	# if writer is not None:
	#     writer.write(frame)
vs.release()
writer.release()
cv2.destroyAllWindows()




