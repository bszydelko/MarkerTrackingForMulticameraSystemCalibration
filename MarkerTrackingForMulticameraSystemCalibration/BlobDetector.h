#pragma once
#include <opencv2/opencv.hpp>
#include "Blob.h"

using namespace cv;
using namespace std;

class BlobDetector
{
public:
	BlobDetector() {};
	void init(const Mat& _mask);
	void apply(const Mat& _image, const Rect& _window);

	Blob getBestBlob(const Rect2d& window);

	const Mat& getMask();
	const Mat& getMaskInv();
	const Mat& getGray();
	const Mat& getBinary();
	const Mat& getEdges();

private:
	Rect window;

	Mat mask;
	Mat maskInv;

	Mat rawImage;
	Mat image;
	Mat gray;
	Mat binary;
	Mat edges;

	uint8_t thresh{ 250 };
	Mat dilateKernel{ getStructuringElement(MorphShapes::MORPH_ELLIPSE, Size(5,5)) };
	vector<vector<Point>> contours;

	vector<Blob> blobs;
	vector<Blob> filteredBlobs;
	vector<Blob> sortedByIntensivity;
	vector<Blob> sortedByCircularity;
	vector<Blob> sortedByArea;

	vector<Blob> history;


	void filterBlobs();
	void sortBlobs();

};

