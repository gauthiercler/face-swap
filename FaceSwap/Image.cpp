#include "pch.h"
#include "Image.h"

Image::Image()
{
}

Image::~Image()
{
}

void Image::load(const std::string &path)
{
	_path = path;
	std::cout << "Loading " << _path << std::endl;
	_image = cv::imread(_path);
}

const std::string & Image::getPath() const
{
	return _path;
}

cv::Mat & Image::getImage()
{
	return _image;
}

cv::Mat & Image::getWarpedImage()
{
	return _warped;
}
