#include "Blob.h"

Blob::Blob()
{
	contour.clear();
	intensitySum = 0.0;
	circularity = 0.0;
	perimeter = 0.0;
	area = 0.0;
	centroid = Point2d(-1, -1);
	convex = false;
	correct = false;





}

Blob::Blob(Point2d _centroid)
{
	centroid = _centroid;
}


Blob::Blob(const vector<Point>& _contour, const Mat& _image, const Mat& _mask, uint8_t _ID)
{
	ID = _ID;
	contour = _contour;
	cntMoments = moments(contour);

	if (cntMoments.m00 != 0)
	{
		centroid = Point2d((cntMoments.m10 / cntMoments.m00), (cntMoments.m01 / cntMoments.m00));

		rect = boundingRect(contour);

		image = _image(rect);
		mask = _mask(rect);

		intensityRGB = mean(image, mask);
		intensitySum = intensityRGB[0] + intensityRGB[1] + intensityRGB[2];

		area = contourArea(contour);
		perimeter = arcLength(contour, true);
		circularity = 4 * CV_PI * area / (perimeter * perimeter);

		convex = isContourConvex(contour);
	}
	else 
	{

		centroid = Point2d(-1, -1);
		intensityRGB = { 0,0,0,0 };
		area = 0;
		perimeter = 0;
		circularity = 0;
		convex = false;

	}

}

uint8_t Blob::getIntensity() const
{
	return uint8_t(intensityRGB[0]);
}

double Blob::getCircularity() const
{
	return circularity;
}

double Blob::getArea() const
{
	return area;
}

Point2d Blob::getCentroid() const
{
	return Point2d(centroid);
}

vector<Point> Blob::getContour() const
{
	return vector<Point>(contour);
}

double Blob::getEquivalentDiameter() const
{
	return sqrt(4 * area / CV_PI);
}

bool Blob::isConvex() const
{
	return convex;
}

bool Blob::isOnMask(const Mat& mask) const
{
	for (const auto& point : contour)
	{
		if (mask.at<uint8_t>(point)) return true;
	}

	return false;
}

bool Blob::isCorrect() const
{
	return correct;
}

uint8_t Blob::getID()
{
	return uint8_t(ID);
}

void Blob::includeWindowOffset(const Rect& window)
{
	centroid.x += window.tl().x;
	centroid.y += window.tl().y;
}


