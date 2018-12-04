#include "pch.h"
#include "FaceSwap.h"

void displayUsage()
{
	std::cerr << "USAGE:" << std::endl;
	std::cerr << "FaceSwap.exe firstInputIMage secondInputImage." << std::endl;
	std::cerr << "Be sure that file face_prediction.dat is in same directory as the binary." << std::endl;
}

int main(int argc, char *argv[])
{
	FaceSwap faceSwap;

	if (argc != 3) {
		displayUsage();
		exit(1);
	}
	faceSwap.load(argv[1], argv[2]);
	faceSwap.process();
	faceSwap.display("First");

	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}