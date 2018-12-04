#include "pch.h"
#include "FaceDetector.h"
#include <dlib/image_io.h>



FaceDetector::FaceDetector() : _detector(dlib::get_frontal_face_detector())
{
}


FaceDetector::~FaceDetector()
{
}

void FaceDetector::loadPredictor(const std::string & path)
{
	std::cout << "Loading predictor " << path << std::endl;
	dlib::deserialize(path) >> _shapePredictor;
}

std::vector<cv::Point2f> FaceDetector::getPoints(const std::string &imagePath)
{
	dlib::array2d<dlib::rgb_pixel> image;
	std::vector<cv::Point2f> points;
	std::vector<dlib::rectangle> dets;
	dlib::full_object_detection shape;

	std::cout << "Assessing input image " << imagePath << std::endl;
	dlib::load_image(image, imagePath);
	dets = _detector(image);
	if (dets.size() == 0) {
		std::cout << "No face found" << std::endl;
		return points;
	}
	if (dets.size() > 1)
		std::cout << dets.size() << " faces detected, processing first found" << std::endl;

	shape = _shapePredictor(image, dets[0]);
	std:: cout << "number of parts found: " << shape.num_parts() << std::endl;
	for (unsigned int idx = 0; idx < shape.num_parts(); ++idx) {
		dlib::point point = shape.part(idx);
		points.push_back(cv::Point2f(point.x(), point.y()));
	}
	return points;
}








