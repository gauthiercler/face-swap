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
	std::cout << "Loading completed" << std::endl;
}

void FaceSwap::process()
{
	_firstImage.points = _faceDetector.getPoints(_firstImage.getPath());
	_secondImage.points = _faceDetector.getPoints(_secondImage.getPath());
	_firstImage.getImage().convertTo(_firstImage.getImage(), CV_32F);
	_firstImage.getWarpedImage().convertTo(_firstImage.getWarpedImage(), CV_32F);
	this->getConvexPolygon();
	this->applyDelaunayTriangulation();
	this->getTrianglePoints();
	this->calculateMask();
}

void FaceSwap::display(const std::string &windowName)
{
	cv::imshow(windowName, _output);
}

void FaceSwap::getConvexPolygon()
{
	std::vector<int> convexIndexes;

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
	std::vector<cv::Point2f> points(3);
	std::vector<int> ind(3);

	for (auto &&point : _secondImage.hull)
		subdivision.insert(point);
	subdivision.getTriangleList(triangleBatch);
	for (auto &&triangle : triangleBatch)
	{
		points[0] = cv::Point2f(triangle[0], triangle[1]);
		points[1] = cv::Point2f(triangle[2], triangle[3]);
		points[2] = cv::Point2f(triangle[4], triangle[5]);
		if (rectangle.contains(points[0])
			&& rectangle.contains(points[1])
				&& rectangle.contains(points[2])) {
			for (unsigned int idx = 0; idx < points.size(); ++idx) {
				for (unsigned int ptsIndex = 0; ptsIndex < _secondImage.hull.size(); ++ptsIndex) {
					if (std::abs(points.at(idx).x - _secondImage.hull.at(ptsIndex).x) < 1.0f
						&& std::abs(points.at(idx).x - _secondImage.hull.at(ptsIndex).x) < 1.0f) {
						ind[idx] = static_cast<int>(ptsIndex);
					}
				}
				_triangles.push_back(ind);
			}
		}
	}
}

void FaceSwap::getTrianglePoints()
{
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
	cv::Rect r1;
	cv::Rect r2;
	std::vector<cv::Point2f> t1Rect;
	std::vector<cv::Point2f> t2Rect;
	std::vector<cv::Point> t2RectInt;
	cv::Mat img1Rect;
	cv::Mat img2Rect;
	cv::Mat mask;
	cv::Mat warpMat;

	r1 = cv::boundingRect(_firstImage.triangles);
	r2 = cv::boundingRect(_secondImage.triangles);
	for (int i = 0; i < 3; i++)
	{
		t1Rect.push_back(cv::Point2f(_firstImage.triangles.at(i).x - r1.x, _firstImage.triangles.at(i).y - r1.y));
		t2Rect.push_back(cv::Point2f(_secondImage.triangles.at(i).x - r2.x, _secondImage.triangles.at(i).y - r2.y));
		t2RectInt.push_back(cv::Point(_secondImage.triangles.at(i).x - r2.x, _secondImage.triangles.at(i).y - r2.y));
	}
	mask = cv::Mat::zeros(r2.height, r2.width, CV_32FC3);
	fillConvexPoly(mask, t2RectInt, cv::Scalar(1.0, 1.0, 1.0), 16, 0);
	_firstImage.getImage()(r1).copyTo(img1Rect);
	img2Rect = cv::Mat::zeros(r2.height, r2.width, img1Rect.type());
	warpMat = cv::getAffineTransform(t1Rect, t2Rect);
	cv::warpAffine(img1Rect, img2Rect, warpMat, img2Rect.size(), cv::InterpolationFlags::INTER_LINEAR, cv::BorderTypes::BORDER_REFLECT_101);
	cv::multiply(img2Rect, mask, img2Rect);
	cv::multiply(_firstImage.getWarpedImage()(r2), cv::Scalar(1.0, 1.0, 1.0) - mask, _firstImage.getWarpedImage()(r2));
	_firstImage.getWarpedImage()(r2) = _firstImage.getWarpedImage()(r2) + img2Rect;
}

void FaceSwap::calculateMask()
{
	std::vector<cv::Point> hull8U;

	for (auto &&point : _secondImage.hull)
		hull8U.push_back(cv::Point(point.x, point.y));
	cv::Mat mask = cv::Mat::zeros(_secondImage.getImage().rows, _secondImage.getImage().cols, _secondImage.getImage().depth());
	fillConvexPoly(mask, &hull8U[0], hull8U.size(), cv::Scalar(255, 255, 255));
	cv::Rect r = cv::boundingRect(_secondImage.hull);
	cv::Point center = (r.tl() + r.br()) / 2;
	_firstImage.getWarpedImage().convertTo(_firstImage.getWarpedImage(), CV_8UC3);
	cv::seamlessClone(_firstImage.getWarpedImage(), _secondImage.getImage(), mask, center, _output, cv::NORMAL_CLONE);
}
