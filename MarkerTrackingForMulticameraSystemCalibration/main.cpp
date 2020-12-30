#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "CaptureYUV/CaptureYUV.h"
#include <chrono>
#include "BlobDetector.h"
#include "BlobTracker.h"
#include "FileManager.h"
#include "utils.h"

#define DEBUG 1


using namespace std;
using namespace cv;
using namespace bs;


int main(int argc, char** argv)
{

	String keys =
		"{@sc| | number of sequences		}"
		"{@fc| | number of frames		}"
		"{@sl| | path to sequence list	}"
		"{@ml| | path to mask list		}"
		"{@w| | width of sequence		}"
		"{@h| | height of sequence		}"
		"{@chs| | chroma subsampling		}"
		"{help usage| | print this message		}";

	CommandLineParser parser(argc, argv, keys);

	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}
#if !DEBUG
	int seqCount = parser.get<int>(0);
	int frameCount = parser.get<int>(1);
	String seqListPath = parser.get<String>(2);
	String maskListPath = parser.get<String>(3);
	int width = parser.get<int>(4);
	int height = parser.get<int>(5);
	int chromaSubsampling = parser.get<int>(6);
#else
	int seqCount = 34;
	int frameCount = 750;
	String seqListPath = "D:\\inz_data\\_SEQ\\seq.txt";
	String maskListPath = "D:\\inz_data\\_SEQ\\mask.txt";
	int width = 1920;
	int height = 1080;
	int chromaSubsampling = 420;
#endif
	
	

	FileManager fileManager(seqListPath, maskListPath);

	Size seqResolution(width, height);

	Mat frame;
	Mat mask;

	BlobDetector blobDetector;
	Blob bestBlob;
	vector<Blob> blobHistory;
	Rect2d detectWindow(Point(0, 0), seqResolution);


	BlobTracker blobTracker;
	vector<Point2d> initPts;
	vector<Point> predictedPts;
	Rect2d trackWindow;
	int maxWindowScaler  = 10;
	int windowScaler = 1;
	int lastWindowScaler = 0;
	const cv::Point2d NOT_DETECTED(-1, -1);

	int key = 0;
	int waitTime = 1;

	cv::Point2d** points = new cv::Point2d * [frameCount];
	for (size_t i = 0; i < frameCount; i++)
		points[i] = new cv::Point2d[seqCount];

	int seqNum = 0;
	int frameNum = 0;
	int initialFrames = 4;

	auto start = chrono::high_resolution_clock::now();

	while (!fileManager.eof() && seqNum < seqCount)
	{
		string sequenceName = fileManager.getSequencePath();
		string maskName = fileManager.getMaskPath();

		CaptureYUV sequenceCap(sequenceName, width, height, chromaSubsampling, 1);
		CaptureYUV maskCap(maskName, width, height, chromaSubsampling, 1);

		maskCap.read(mask);
		blobDetector.init(mask);
		blobHistory.clear();

		detectWindow = Rect2d(Point(0, 0), seqResolution);

		cout << sequenceName << endl;

		//initial frames
		initPts.clear();
		frameNum = 0;
		for (size_t i = 0; i < initialFrames; i++, frameNum++)
		{
			sequenceCap.read(frame);
			blobDetector.apply(frame, detectWindow);
			bestBlob = blobDetector.getBestBlob(detectWindow);
			initPts.push_back(bestBlob.getCentroid());
			blobHistory.push_back(bestBlob.getCentroid());
		}
		predictedPts.clear();
		blobTracker.init(initPts, seqResolution);

		

		while (sequenceCap.read(frame) && frameNum < frameCount)
		{
			Point predictedPt = blobTracker.predict();

			do
			{
				blobDetector.apply(frame, detectWindow);

				if (lastWindowScaler <= maxWindowScaler)
					trackWindow = blobTracker.window(windowScaler);
				else
					trackWindow = detectWindow;

				bestBlob = blobDetector.getBestBlob(trackWindow);

				windowScaler++;

			} while (windowScaler <= maxWindowScaler && bestBlob.getCentroid() == NOT_DETECTED);

			lastWindowScaler = windowScaler;
			windowScaler = 1;

			if (bestBlob.getCentroid() != NOT_DETECTED)
				blobTracker.update(bestBlob.getCentroid());
			else
				blobTracker.update(predictedPt);

			predictedPts.push_back(predictedPt);
			blobHistory.push_back(bestBlob);

			drawLines(frame, blobHistory, 50);
			drawLines(frame, predictedPts, 50);

			circle(frame, predictedPt, 10, Scalar(0, 255, 255), 2);
			circle(frame, bestBlob.getCentroid(), 10, Scalar(0, 255, 0), 2);

			rectangle(frame, trackWindow, Scalar(255, 0, 0), 2);

			drawInfo(frame, seqNum, frameNum, bestBlob.getCentroid(),predictedPt);
			imshow("frame", frame);

			key = waitKey(waitTime);
			if (key == 27) break;
			if (key == 'n') break;
			if (key == 32) {
				waitTime = 0;
				waitKey(waitTime);
				waitTime = 1;
			}
			frameNum++;

		}
		
		if (key != 'n' && key != 27) {
			for (size_t i = 0; i < frameCount; i++)
			{
				points[i][seqNum] = blobHistory[i].getCentroid();
			}
		}
		seqNum++;

		if (key == 27) break;
	}

	auto finish = chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;

	std::ofstream file_final_pos("points-detector-windowTracker.txt", std::ios::out);

	//save to file

	for (size_t i = 0; i < frameCount; i++)
	{
		for (size_t j = 0; j < seqNum; j++)
		{
			file_final_pos << std::setprecision(3) << std::fixed;
			if (points[i][j] == NOT_DETECTED)
				file_final_pos << points[i][j].x << "\t" << points[i][j].y << "\t" << 0 << "\t";
			else
				file_final_pos << points[i][j].x << "\t" << points[i][j].y << "\t" << 1 << "\t";
		}
		file_final_pos << "\n";
	}

	std::cout << "POINTS SAVED\n";
	std::cout << "Elapsed time: " << elapsed.count();
	file_final_pos.close();

	return EXIT_SUCCESS;
}