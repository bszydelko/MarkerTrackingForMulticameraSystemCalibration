﻿#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include "../MarkerTrackingForMulticameraSystemCalibration/CaptureYUV/CaptureYUV.h"


using namespace bs;
using namespace cv;
using namespace std;

int main()
{
    string filename = "D:\\inz_data\\_SEQ\\1\\cam1_1920x1080.yuvdist.yuv";
    CaptureYUV yuvcap(filename,1920, 1080, 420, 1);
    Mat frame, roi, hsv_roi, mask;
    // take first frame of the video
    yuvcap.read(frame);
    // setup initial location of window
    Rect track_window(300, 200, 100, 50); // simply hardcoded the values
    track_window = selectROI(frame, true, true);
    // set up the ROI for tracking
    roi = frame(track_window);
    cvtColor(roi, hsv_roi, COLOR_BGR2HSV);
    inRange(hsv_roi, Scalar(0, 60, 32), Scalar(180, 255, 255), mask);
    float range_[] = { 0, 180 };
    const float* range[] = { range_ };
    Mat roi_hist;
    int histSize[] = { 180 };
    int channels[] = { 0 };
    calcHist(&hsv_roi, 1, channels, mask, roi_hist, 1, histSize, range);
    normalize(roi_hist, roi_hist, 0, 255, NORM_MINMAX);
    // Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    TermCriteria term_crit(TermCriteria::EPS | TermCriteria::COUNT, 10, 100);
    while (true) {
        Mat hsv, dst;
        yuvcap.read(frame);
        if (frame.empty())
            break;
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        calcBackProject(&hsv, 1, channels, roi_hist, dst, range);
        // apply camshift to get the new location
        RotatedRect rot_rect = CamShift(dst, track_window, term_crit);
        // Draw it on image
        Point2f points[4];
        rot_rect.points(points);
        for (int i = 0; i < 4; i++)
            line(frame, points[i], points[(i + 1) % 4], 255, 2);
        imshow("img2", frame);
        int keyboard = waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;
    }
}