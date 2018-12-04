#include "pch.h"
#include "FaceSwap.h"


FaceSwap::FaceSwap() : _faceDetector()
{

}


FaceSwap::~FaceSwap()
{
}

void FaceSwap::load(const std::string &firstImagePath,
					const std::string &secondImagePath)
{
	_firstImage.load(firstImagePath);
	_secondImage.load(secondImagePath);
	_firstImage._warped = _secondImage.getImage().clone();
	_faceDetector.loadPredictor("shape_predictor_68_face_landmarks.dat");
	std::cout << "Loading complete" << std::endl;
}

void FaceSwap::process()
{
	std::cout << "Starting processing" << std::endl;
	_firstImage.points = _faceDetector.getPoints(_firstImage.getPath());
	_secondImage.points = _faceDetector.getPoints(_secondImage.getPath());
	_firstImage.getImage().convertTo(_firstImage.getImage(), CV_32F);
	_firstImage.getWarpedImage().convertTo(_firstImage.getWarpedImage(), CV_32F);
	this->getConvexPolygon();
	this->applyDelaunayTriangulation();
	this->getTrianglePoints();
	this->calculateMask();
	std::cout << "Processing complete" << std::endl;
}

void FaceSwap::display(const std::string &windowName)
{
	std::cout << "Display result" << std::endl;
	cv::imshow(windowName, _output);
}

void FaceSwap::getConvexPolygon()
{
	std::vector<int> convexIndexes;

	std::cout << "Getting convex polygon" << std::endl;
	cv::convexHull(_secondImage.points, convexIndexes, false, false);
	for (auto &&index : convexIndexes)
	{
		_firstImage.hull.push_back(_firstImage.points.at(index));
		_secondImage.hull.push_back(_secondImage.points.at(index));
	}
}

void FaceSwap::applyDelaunayTriangulation()
{
	std::vector<cv::Vec6f> triangleBatch;
	cv::Rect rectangle(0, 0, _firstImage.getWarpedImage().cols, _firstImage.getWarpedImage().rows);
	cv::Subdiv2D subdivision(rectangle);
	std::vector<cv::Point2f> points;
	std::vector<int> out(3);

	std::cout << "Applying delaunay triangulation" << std::endl;
	for (auto &&point : _secondImage.hull)
		subdivision.insert(point);
	subdivision.getTriangleList(triangleBatch);
	for (auto &&triangle : triangleBatch)
	{
		points.push_back(cv::Point2f(triangle[0], triangle[1]));
		points.push_back(cv::Point2f(triangle[2], triangle[3]));
		points.push_back(cv::Point2f(triangle[4], triangle[5]));
		if (rectangle.contains(points[0])
			&& rectangle.contains(points[1])
				&& rectangle.contains(points[2])) {
			for (unsigned int idx = 0; idx < points.size(); idx++) {
				for (unsigned int ptsIndex = 0; ptsIndex < _secondImage.hull.size(); ptsIndex++) {
					if (std::abs(points.at(idx).x - _secondImage.hull.at(ptsIndex).x) < 1.0f
						&& std::abs(points.at(idx).y - _secondImage.hull.at(ptsIndex).y) < 1.0f) {
						out[idx] = (int)(ptsIndex);
					}
				}
				_triangles.push_back(out);
			}
		}
		points.clear();
	}
}

void FaceSwap::getTrianglePoints()
{
	std::cout << "Applying affine transformation" << std::endl;
	for (auto &&triangle : _triangles) {
		for (auto &&point : triangle) {
			_firstImage.triangles.push_back(_firstImage.hull[point]);
			_secondImage.triangles.push_back(_secondImage.hull[point]);
		}
		this->warpTriangle();
		_firstImage.triangles.clear();
		_secondImage.triangles.clear();
	}
}

void FaceSwap::warpTriangle()
{
	cv::Mat mask;
	cv::Mat warpMat;

	_firstImage.boundingRectangle = cv::boundingRect(_firstImage.triangles);
	_secondImage.boundingRectangle = cv::boundingRect(_secondImage.triangles);
	for (int i = 0; i < 3; i++)
	{
		_firstImage.trianglesOffset.push_back(cv::Point2f(_firstImage.triangles.at(i).x - _firstImage.boundingRectangle.x, _firstImage.triangles.at(i).y - _firstImage.boundingRectangle.y));
		_secondImage.trianglesOffset.push_back(cv::Point2f(_secondImage.triangles.at(i).x - _secondImage.boundingRectangle.x, _secondImage.triangles.at(i).y - _secondImage.boundingRectangle.y));
	}

	mask = cv::Mat::zeros(_secondImage.boundingRectangle.height, _secondImage.boundingRectangle.width, CV_32FC3);
	fillConvexPoly(mask, std::vector<cv::Point>(_secondImage.trianglesOffset.begin(), _secondImage.trianglesOffset.end()), cv::Scalar(1.0, 1.0, 1.0), 16, 0);
	_firstImage.getImage()(_firstImage.boundingRectangle).copyTo(_firstImage.patch);
	_secondImage.patch = cv::Mat::zeros(_secondImage.boundingRectangle.height, _secondImage.boundingRectangle.width, _firstImage.patch.type());
	warpMat = cv::getAffineTransform(_firstImage.trianglesOffset, _secondImage.trianglesOffset);
	cv::warpAffine(_firstImage.patch, _secondImage.patch, warpMat, _secondImage.patch.size(), cv::InterpolationFlags::INTER_LINEAR, cv::BorderTypes::BORDER_REFLECT_101);
	cv::multiply(_secondImage.patch, mask, _secondImage.patch);
	cv::multiply(_firstImage.getWarpedImage()(_secondImage.boundingRectangle), cv::Scalar(1.0, 1.0, 1.0) - mask, _firstImage.getWarpedImage()(_secondImage.boundingRectangle));
	_firstImage.getWarpedImage()(_secondImage.boundingRectangle) = _firstImage.getWarpedImage()(_secondImage.boundingRectangle) + _secondImage.patch;
	_firstImage.trianglesOffset.clear();
	_secondImage.trianglesOffset.clear();
}

void FaceSwap::calculateMask()
{
	std::vector<cv::Point> hull8U;

	std::cout << "Calculating mask" << std::endl;
	for (auto &&point : _secondImage.hull)
		hull8U.push_back(cv::Point(point.x, point.y));
	cv::Mat mask = cv::Mat::zeros(_secondImage.getImage().rows, _secondImage.getImage().cols, _secondImage.getImage().depth());
	fillConvexPoly(mask, &hull8U[0], hull8U.size(), cv::Scalar(255, 255, 255));
	cv::Rect r = cv::boundingRect(_secondImage.hull);
	cv::Point center = (r.tl() + r.br()) / 2;
	_firstImage.getWarpedImage().convertTo(_firstImage.getWarpedImage(), CV_8UC3);
	cv::seamlessClone(_firstImage.getWarpedImage(), _secondImage.getImage(), mask, center, _output, cv::NORMAL_CLONE);
}
