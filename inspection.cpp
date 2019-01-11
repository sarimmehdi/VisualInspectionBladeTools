#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <numeric>
#include <cstdlib>

#include "inspection.h"

//uncomment this line if you want to see debug images
//#define DEBUG

double Inspector::grad(cv::Vec4i &line)
{
	double x1 = (double)line[0], x2 = (double)line[2]; double y1 = (double)line[1], y2 = (double)line[3];
	if (x2 - x1 == 0) { x2++; }
	return ((y2 - y1) / (x2 - x1));
}

void Inspector::callbackfunc(int v, void* ptr)
{
	// resolve 'this':
	Inspector *that = (Inspector*)ptr;
	//that->thresh_callback();
	that->doInspection();
}

void Inspector::checkImp(cv::Mat &input, cv::Mat &output, int margin, int startingPoint, int endPoint, double threshold)
{
	double sum; cv::Vec3b color = cv::Vec3b(255,0,255);
	int highC, lowRow, highRow, lowCol, highCol;
	std::cout << startingPoint << std::endl; std::cout << endPoint << std::endl;

	if (input.cols - margin <= input.cols) { highC = input.cols - margin; }
	else { highC = input.cols; }
	for (int r = startingPoint; r < endPoint; r++)
	{
		for (int c = margin; c < highC; c++)
		{
			sum = 0.0;
			if (r - margin >= 0) { lowRow = r - margin; }
			else { lowRow = 0; }

			if (r + margin + 1 <= input.rows) { highRow = r + margin + 1; }
			else { highRow = input.rows; }

			if (c - margin >= 0) { lowCol = c - margin; }
			else { lowCol = 0; }

			if (c + margin + 1 <= input.cols) { highCol = c + margin + 1; }
			else { highCol = input.cols; }
			for (int R = lowRow; R < highRow; R++)
			{
				for (int C = lowCol; C < highCol; C++)
				{
					sum += input.at<uchar>(R, C);
				}
			}
			sum = sum / ((margin + margin + 1) * (margin + margin + 1) * 255);
			if (sum >= threshold) { output.at<cv::Vec3b>(r, c) = color; }
		}
	}
}

void Inspector::drawTrigs(cv::Mat &binaryImg, cv::Mat &normalImg, cv::Mat &binaryTemp, std::vector<cv::Point> &corners, 
	std::vector<cv::Point> &helperPoints)
{
	std::vector<cv::Point> candidates; int lowR, highR, lowC, highC; cv::Vec4i l1, l2;
	cv::Point write; double m1, m2, angle;
	cv::Point2f vtx[4]; cv::RotatedRect box; std::vector<cv::Point2f> triangle;
	putText(normalImg, "BAD ANGLE", cv::Point(50, 50), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0, 0, 255));
	putText(normalImg, "GOOD ANGLE", cv::Point(50, 100), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(255, 0, 0));
	putText(normalImg, "RANGE " + std::to_string(minAngle) + " TO " + std::to_string(maxAngle), cv::Point(50, 150), 
		cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(255, 0, 0));
	for (size_t i = 0; i < corners.size(); i++)
	{
		if (corners[i].y + maxLength <= binaryImg.rows) { highR = corners[i].y + maxLength; }
		else { highR = binaryImg.rows; }

		if (corners[i].x - maxWidth >= 0) { lowC = corners[i].x - maxWidth; }
		else { lowC = 0; }

		if (corners[i].x + maxWidth <= binaryImg.cols) { highC = corners[i].x + maxWidth; }
		else { highC = binaryImg.cols; }

		for (int r = corners[i].y; r < highR; r++)
		{
			for (int c = lowC; c < highC; c++)
			{
				if (binaryImg.at<uchar>(r, c) > 0) 
				{ 
					candidates.push_back(cv::Point(c, r)); 
				}
			}
		}

		if (!candidates.empty()) 
		{ 
			box = cv::minAreaRect(candidates);
			box.points(vtx);
			minEnclosingTriangle(candidates, triangle);
			for (int j = 0; j < 3; j++) 
			{
				line(normalImg, triangle[j], triangle[(j + 1) % 3], cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
				line(binaryTemp, triangle[j], triangle[(j + 1) % 3], cv::Scalar(255), 1, cv::LINE_AA);
				helperPoints.push_back(triangle[j]);
			}
			
			//https://www.mathstopia.net/coordinate-geometry/angle-two-lines
			sort(triangle.begin(), triangle.end(), [](const cv::Point2f &a, const cv::Point2f &b) {
				return (a.y < b.y);
			});
			write = triangle[0]; write.y -= 50;
			l1 = cv::Vec4i(triangle[0].x, triangle[0].y, triangle[1].x, triangle[1].y);
			l2 = cv::Vec4i(triangle[0].x, triangle[0].y, triangle[2].x, triangle[2].y);
			m1 = grad(l1); m2 = grad(l2); angle = atan((m1 - m2) / (1 + m1 * m2)); angle = abs(angle * (180 / CV_PI));
			if (angle > minAngle && angle < maxAngle)
			{
				putText(normalImg, std::to_string(angle), write, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(255, 0, 0));
			}
			else if (angle > minAngle - 50 || angle < maxAngle + 50)
			{
				putText(normalImg, std::to_string(angle), write, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(0, 0, 255));
			}
			circle(normalImg, triangle[0], 1, cv::Scalar(0, 0, 255), 2, 8, 0);

			candidates.clear(); 
		}
	}
}

void Inspector::doInspection()
{
	cv::Mat canny_output, dst = cv::Mat::zeros(src_gray.size(), CV_32FC1), dst_norm, srcTemp;
	Canny(src_gray, canny_output, thresh, thresh * 2, 3); //Detect edges using canny
#ifdef DEBUG
	imshow("Canny original", canny_output); imwrite("Canny_original.png", canny_output);
#endif

	//Remove curves using hough line transform
	cv::Mat lineImg = cv::Mat::zeros(src_gray.size(), CV_8UC1);
	std::vector<cv::Vec4i> linesP; cv::Vec4i l;
	HoughLinesP(canny_output, linesP, 1, CV_PI / 180, houghThresh, minLineGapThresh, maxLineGapThresh);
	if (linesP.empty()) { std::cout << "UNABLE TO FIND ANY LINES! PLEASE ADJUST PARAMETERS FOR HOUGH LINES" << std::endl; return; }
	for (size_t i = 0; i < linesP.size(); i++)
	{
		l = linesP[i]; line(lineImg, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255), 3, cv::LINE_AA);
	}
	cv::bitwise_and(canny_output, lineImg, canny_output);
#ifdef DEBUG
	imshow("Hough lines", lineImg); imwrite("Hough_lines.png", lineImg);
	imshow("Canny without curves", canny_output); imwrite("without_curves.png", canny_output);
#endif

	//remove horizontal lines using morphology operation
	int horizontalsize = src_gray.cols / 100;
	cv::Mat cannyTemp = canny_output.clone();
	cv::Mat horizontalStructure = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(horizontalsize, 1));
	erode(cannyTemp, cannyTemp, horizontalStructure, cv::Point(-1, -1));
	dilate(cannyTemp, cannyTemp, horizontalStructure, cv::Point(-1, -1));
	cv::bitwise_xor(canny_output, cannyTemp, canny_output);
#ifdef DEBUG
	imshow("Canny with only horizontal line", cannyTemp); imwrite("only_horizontal_line.png", cannyTemp);
	imshow("Canny without horizontal lines", canny_output); imwrite("without_horizontal_lines.png", canny_output);
#endif

	//Detect corners
	int blockSize = 5, apertureSize = 3; double k = 0.04;
	std::vector<cv::Point> corners;
	cornerHarris(src_gray, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
	normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
	for (int j = 0; j < dst_norm.rows; j++)
	{
		for (int i = 0; i < dst_norm.cols; i++)
		{
			if ((int)dst_norm.at<float>(j, i) > cornerThresh) { corners.push_back(cv::Point(i, j)); }
		}
	}
	if (corners.empty()) { std::cout << "UNABLE TO FIND ANY CORNERS! PLEASE ADJUST PARAMETERS FOR CORNERS" << std::endl; return; }
#ifdef DEBUG
	cv::Mat imgAll1 = src_gray.clone(); cvtColor(imgAll1, imgAll1, cv::COLOR_GRAY2RGB);
	for (size_t i = 0; i < corners.size(); i++) { circle(imgAll1, corners[i], 1, cv::Scalar(0, 0, 255), 2, 8, 0); }
	imshow("All the corners before any filtering", imgAll1); imwrite("filtering.png", imgAll1);
#endif

	//filter corners based on average y value (corners too low will be removed)
	std::vector<cv::Point> newCors;
	cv::Point sum = std::accumulate(corners.begin(), corners.end(), cv::Point(0, 0));
	cv::Point2d average(sum.x / corners.size(), sum.y / corners.size());
	for (size_t i = 0; i < corners.size(); i++)
	{
		if (corners[i].y < average.y + cornerHeight) { newCors.push_back(corners[i]); }
	}
	corners = newCors; newCors.clear();
#ifdef DEBUG
	cv::Mat imgAll2 = src_gray.clone(); cvtColor(imgAll2, imgAll2, cv::COLOR_GRAY2RGB);
	for (size_t i = 0; i < corners.size(); i++) { circle(imgAll2, corners[i], 1, cv::Scalar(0, 0, 255), 2, 8, 0); }
	imshow("All the corners after filtering stage 1", imgAll2); imwrite("filtering1.png", imgAll2);
#endif 

	//one corner for one peak
	sort(corners.begin(), corners.end(), pointXFunction);
	cv::Point prevCor = corners[0]; newCors.push_back(prevCor);
	for (size_t i = 1; i < corners.size(); i++)
	{
		if (pointDist(prevCor, corners[i]) > 10) { prevCor = corners[i]; newCors.push_back(prevCor); }
	}
	corners = newCors;
#ifdef DEBUG
	cv::Mat imgAll3 = src_gray.clone(); cvtColor(imgAll3, imgAll3, cv::COLOR_GRAY2RGB);
	for (size_t i = 0; i < corners.size(); i++) { circle(imgAll3, corners[i], 1, cv::Scalar(0, 0, 255), 2, 8, 0); }
	imshow("All the corners after filtering stage 2", imgAll3); imwrite("filtering2.png", imgAll3);
#endif
	
	//draw triangles with angles
	cvtColor(src_gray, srcTemp, cv::COLOR_GRAY2RGB); 
	cv::Mat newCannyTemp = canny_output.clone(); cv::Mat impImg = srcTemp.clone();
	std::vector<cv::Point> imperfect;
	drawTrigs(canny_output, srcTemp, newCannyTemp, corners, imperfect);

	//detect imperfections
	if (checkQuality)
	{
		if (imperfect.empty()) { std::cout << "UNABLE TO FIND IMPERFECTIONS!" << std::endl; return; }
		double yourThresh = (double)theThresh / (double)maxTheThresh;
		cv::bitwise_xor(newCannyTemp, canny_output, newCannyTemp);
		std::cout << "checking for imperfections" << std::endl;
		sort(imperfect.begin(), imperfect.end(), pointYFunction);
		checkImp(newCannyTemp, impImg, 5, imperfect[0].y, imperfect[imperfect.size() - 1].y, yourThresh);
		std::cout << "found all imperfections" << std::endl;
		imshow("Imperfections", impImg);
#ifdef DEBUG
		imwrite("Imperfections.png", impImg);
#endif
	}

	imshow("Final Image", srcTemp);
#ifdef DEBUG
	imwrite("Final_image.png", srcTemp);
#endif
}

void Inspector::makeTrackbar()
{
	cv::namedWindow("Source");
	cv::createTrackbar("Start", "Source", &start, maxStart, callbackfunc, this);
	cv::createTrackbar("Check\nQuality", "Source", &checkQuality, maxCheckQuality, callbackfunc, this);
	cv::createTrackbar("Canny\nThresh", "Source", &thresh, max_thresh, callbackfunc, this);
	cv::createTrackbar("Doug\nThresh", "Source", &epsilon, max_epsilon, callbackfunc, this);
	cv::createTrackbar("Quality\nThresh", "Source", &theThresh, maxTheThresh, callbackfunc, this);
	cv::createTrackbar("Min\nAngle", "Source", &minAngle, maxMinAngle, callbackfunc, this);
	cv::createTrackbar("Max\nAngle", "Source", &maxAngle, maxMaxAngle, callbackfunc, this);
	cv::createTrackbar("Max\nLength", "Source", &maxLength, maxMaxLength, callbackfunc, this);
	cv::createTrackbar("Max\nWidth", "Source", &maxWidth, maxMaxWidth, callbackfunc, this);
	cv::createTrackbar("Corner", "Source", &cornerThresh, maxCornerThresh, callbackfunc, this);
	cv::createTrackbar("Corner\nFilter", "Source", &cornerHeight, maxCornerHeight, callbackfunc, this);
	cv::createTrackbar("Hough\nThresh", "Source", &houghThresh, maxHoughThresh, callbackfunc, this);
	cv::createTrackbar("Hough\nMin", "Source", &minLineGapThresh, maxminLineGapThresh, callbackfunc, this);
	cv::createTrackbar("Hough\nMax", "Source", &maxLineGapThresh, maxmaxLineGapThresh, callbackfunc, this);
}
