import cv2
import numpy as np
import random

def projection_transform(img,pts1,pts2):

	height, width, ch = img.shape
	matrix = cv2.getPerspectiveTransform(pts1, pts2)

	imgout = cv2.warpPerspective(img, matrix, (width, height))

	return imgout, matrix

def affine_transform(img,pts1,pts2):

	height, width, ch = img.shape
	matrix = cv2.getAffineTransform(pts1, pts2)

	imgout = cv2.warpAffine(img, matrix, (width, height))

	return imgout, matrix

def random_transform(img,shiftxy=[.1,.1],transform='perspective'):
	'''
	inputs:
	transform:'perspective','affine'
	img      : 2D np array
	shiftxy  : random shift in x and y in percentage of width and height
	outputs:
	imgout   : transformed image
	matrix   : transform matrix
	'''
	height, width, ch = img.shape

	dim = (width, height)
	d1=[]
	shiftx, shifty =width*shiftxy[0],height*shiftxy[1]

	# resize image
	#img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 

	#source pts
	pts1 = np.float32([[0, 0], [width, 0],[width, height],[0,height]])

	for pt in pts1:
		cv2.circle(img, tuple(pt), 5, (0, 0, 255), -1)

	#random shifts
	for i in range(4):
		d1.append([random.uniform(-shiftx, shiftx), random.uniform(-shifty, shifty)])
	d1=np.float32(d1)

	#import pdb;pdb.set_trace()
	#applying shifts
	pts2 = pts1+d1

	if transform =='perspective':
		imgout, matrix = projection_transform(img,pts1,pts2)
	elif transform =='affine':
		pts1,pts2 = pts1[0:3],pts2[0:3]
		imgout, matrix = affine_transform(img,pts1,pts2)

	return imgout, matrix



if __name__ =='__main__':
	cv2.namedWindow("input",cv2.WINDOW_NORMAL)
	cv2.namedWindow("perspective",cv2.WINDOW_NORMAL)
	cv2.namedWindow("affine",cv2.WINDOW_NORMAL)
	img = cv2.imread("an1.jpg")
	cv2.imshow("input", img)
	while True:
		result1,matrix1 = random_transform(img,shiftxy=[.1,.1],transform='perspective')
		result2,matrix2 = random_transform(img,shiftxy=[.1,.1],transform='affine')
		
		cv2.imshow("perspective", result1)
		cv2.imshow("affine", result2)
		print('\nT1',matrix1)
		print('\nT2',matrix2)
		k=cv2.waitKey(0)
		if k==27:
			break
	cv2.destroyAllWindows()
