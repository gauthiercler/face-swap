#include "pch.h"
#include "FaceSwap.h"

int main(int argc, char *argv[])
{
	FaceSwap faceSwap;

	faceSwap.load(argv[1], argv[2]);
	faceSwap.process();
	faceSwap.display("First");

	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}