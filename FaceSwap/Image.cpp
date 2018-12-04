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
	if (_image.data == NULL) {
		std::cerr << "Unable to open file " << path << ", aborting" << std::endl;
		exit(1);
	}
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
