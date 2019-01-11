#pragma once
#ifndef INS_H
#define INS_H

class Inspector
{
public:
	Inspector(cv::Mat &yourImg) : src_gray(yourImg) {};

	//create interface
	void makeTrackbar();
private:
	cv::Mat src_gray;                                         //input image
	int thresh = 109, max_thresh = 255;                       //parameters for Canny
	int epsilon = 1, max_epsilon = 100;                       //parameters for Douglas-Peucker
	
	//parameters for Harris corner detector
	int cornerThresh = 115, maxCornerThresh = 255;
	int cornerHeight = 20, maxCornerHeight = 300;

	//min and max angle you want
	int minAngle = 37, maxMinAngle = 39;
	int maxAngle = 42, maxMaxAngle = 50;

	//parameters for detecting corners
	int maxLength = 70, maxMaxLength = 300;
	int maxWidth = 40, maxMaxWidth = 300;

	//parameters for hough line detector
	int houghThresh = 18, maxHoughThresh = 100;
	int minLineGapThresh = 23, maxminLineGapThresh = 100;
	int maxLineGapThresh = 10, maxmaxLineGapThresh = 100;

	//parameter for detecting imperfections
	int theThresh = 15, maxTheThresh = 100;

	//in the beginning just move this slider
	int start = 0, maxStart = 1;

	//set to true if you want to see imperfections
	int checkQuality = 1, maxCheckQuality = 1;

	//distance between two points
	double pointDist(cv::Point &a, cv::Point &b) { return (norm(a - b)); }

	//arrange points in ascending order of their x value
	struct compStructX
	{
		bool operator() (cv::Point &i, cv::Point &j) { return (i.x < j.x); }
	} pointXFunction;

	//arrange points in ascending order of their y value
	struct compStructY
	{
		bool operator() (cv::Point &i, cv::Point &j) { return (i.y < j.y); }
	} pointYFunction;

	//draw triangles at each peak
	void drawTrigs(cv::Mat &binaryImg, cv::Mat &normalImg, cv::Mat &binaryTemp, std::vector<cv::Point> &corners, 
		std::vector<cv::Point> &helperPoints);

	//compute gradient of a line throught two given points
	double grad(cv::Vec4i &line);

	//find imperfections by convolving with a filter of 1's. The pixel at which you get a value greater than selected threshold is imperfect
	void checkImp(cv::Mat &input, cv::Mat &output, int margin, int startingPoint, int endPoint, double threshold);

	static void callbackfunc(int v, void* ptr);

	void doInspection();
};

#endif
