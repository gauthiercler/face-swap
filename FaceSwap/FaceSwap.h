#pragma once

#include "FaceDetector.h"
#include "Image.h"

class FaceSwap
{
public:
	FaceSwap();
	~FaceSwap();
	void load(const std::string &firstImagePath,
		const std::string &secondImagePath);
	void process();
	void display(const std::string &windowName);

private:
	FaceDetector _faceDetector;
	Image _firstImage;
	Image _secondImage;
	cv::Mat _output;
	std::vector<std::vector<int>> _triangles;

	void getConvexPolygon();
	void applyDelaunayTriangulation();
	void getTrianglePoints();
	void warpTriangle();
	void calculateMask();
};

