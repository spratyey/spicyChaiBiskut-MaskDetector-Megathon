from setuptools import setup, find_packages

setup(
	name='detect_mask_video',
	install_requires=[
		'tensorflow',
		'numpy',
		'imutils',
		'opencv-python',
		'mtcnn',
		'matplotlib'
	]
)