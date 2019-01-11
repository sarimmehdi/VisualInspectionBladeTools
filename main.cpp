#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "inspection.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	cout << "                     ******************************************************************" << endl;
	cout << "                     ||                       AUTHOR: Sarim Mehdi                    ||" << endl;
	cout << "                     ||                   CREATED ON: 9 January 2019                 ||" << endl;
	cout << "                     ||           PURPOSE: Visual Inspection of Blade Tools          ||" << endl;
	cout << "                     ******************************************************************" << endl;

	//Load source image and convert it to gray
	Mat src_gray = imread("saw_07.png", IMREAD_GRAYSCALE);
	Inspector myImg(src_gray); myImg.makeTrackbar();

	waitKey(0);
	return(0);
}
