#pragma once
#include <opencv2/opencv.hpp>
#include <fstream>


class Image
{
public:
	Image();
	~Image();
	void load(const std::string &path);
	const std::string &getPath() const;
	cv::Mat &getImage();
	cv::Mat &getWarpedImage();
	std::vector<cv::Point2f> points;
	std::vector<cv::Point2f> hull;
	std::vector<cv::Point2f> triangles;
	std::vector<cv::Point2f> trianglesOffset;
	cv::Mat patch;
	cv::Rect boundingRectangle;
	cv::Mat _warped;

private:
	cv::Mat _image;
	std::string _path;
};

