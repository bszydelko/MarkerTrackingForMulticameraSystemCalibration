#pragma once
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class Blob
{
public:
	Blob();
	Blob(Point2d _centroid);
	Blob(const vector<Point>& _contour, const Mat& _image, const Mat& _mask, uint8_t _ID);

	uint8_t getIntensity() const;
	double getCircularity() const;
	double getArea() const;
	Point2d getCentroid() const;
	vector<Point> getContour() const;

	double getEquivalentDiameter() const;

	bool isConvex() const;
	bool isOnMask(const Mat& mask) const;
	bool isCorrect() const;

	uint8_t getID();

	void includeWindowOffset(const Rect& window);



private:
	uint8_t ID;

	Mat mask;
	Mat image;
	vector<Point> contour;
	Moments cntMoments;

	Scalar intensityRGB;
	double intensitySum;

	double circularity;
	double perimeter;
	double area;

	Point2d centroid;

	bool convex;
	bool correct;

	Rect2d rect;
};

