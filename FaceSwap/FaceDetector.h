#pragma once

#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>


class FaceDetector
{
public:
	FaceDetector();
	~FaceDetector();
	void loadPredictor(const std::string &path);
	std::vector<cv::Point2f> getPoints(const std::string &imagePath);
private:
	dlib::frontal_face_detector _detector;
	dlib::shape_predictor _shapePredictor;
};

	