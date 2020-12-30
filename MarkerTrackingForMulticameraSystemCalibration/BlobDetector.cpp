#include "BlobDetector.h"

void BlobDetector::init(const Mat& _mask)
{
    assert(_mask.empty() == false);
    _mask.copyTo(mask);

    GaussianBlur(mask, mask, Size(3, 3), 0);
    cvtColor(mask, mask, COLOR_BGR2GRAY);
    threshold(mask, mask, 220, 255, ThresholdTypes::THRESH_BINARY_INV);
    erode(mask, maskInv, dilateKernel);
    bitwise_not(maskInv, mask);
    dilate(mask, mask, dilateKernel);

    Rect bound(10, 10, mask.size().width - 20, mask.size().height - 20);
    Mat maskBound = 255 * Mat::ones(mask.size(), CV_8UC1);

    rectangle(maskBound, bound, Scalar(0), -1);

    bitwise_or(maskBound, mask, mask);
}

void BlobDetector::apply(const Mat& _image, const Rect& _window)
{
    assert(_image.empty() == false);

    window = _window;
    //wyntij oknem
    Mat maskInvWindow = maskInv(window);
    Mat maskWindow = mask(window);
    
    rawImage = _image(window);
    //wytnij mask¹
    rawImage.copyTo(image, maskInvWindow);
    

    
    cvtColor(image, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, gray, Size(5, 5), 0);
    threshold(gray, binary, thresh, 255, THRESH_BINARY);
    dilate(binary, binary, dilateKernel);

    //find contours
    contours.clear();
    Canny(binary, edges, thresh, 255);
    findContours(edges, contours, RetrievalModes::RETR_EXTERNAL, ContourApproximationModes::CHAIN_APPROX_NONE);

    blobs.clear();
    uint8_t ID = 0;
    for (const auto& contour : contours)
    {
        Blob blob(contour, rawImage, binary, ID++);
        blobs.push_back(blob);
    }

    filterBlobs();
    //sortBlobs();
}

Blob BlobDetector::getBestBlob(const Rect2d& window)
{
    vector<Blob> best;
    //w calej ramce nic nie znaleziono
   

    for (auto blob : filteredBlobs)
    {
        for (auto pt : blob.getContour())
            if (window.contains(pt)) 
            {
                best.push_back(blob);
                break;
            }
    }
    filteredBlobs.clear();
    filteredBlobs = best;
    if(filteredBlobs.empty())
            return Blob(Point2d(-1, -1));

    sortBlobs();

    if (sortedByArea.front().getID() == sortedByCircularity.front().getID() && sortedByArea.front().getID() == sortedByIntensivity.front().getID())
        return sortedByArea.front();
    else 
        return Blob(Point2d(-1, -1));

}

const Mat& BlobDetector::getMask()
{
    return mask;
}

const Mat& BlobDetector::getMaskInv()
{
    return maskInv;
}

const Mat& BlobDetector::getGray()
{
    return gray;
}

const Mat& BlobDetector::getBinary()
{   
    return binary;
}

const Mat& BlobDetector::getEdges()
{
    return edges;
}

void BlobDetector::filterBlobs()
{

    filteredBlobs.clear();

    //filter by mask overlap
    for (const auto& blob : blobs)
    {
        if (blob.isOnMask(mask))continue;//|| blob.getCircularity() < 0.75) continue;
            
        filteredBlobs.push_back(blob);
    }

}

void BlobDetector::sortBlobs()
{
    auto sortByIntensivity = [&](const Blob& a, const Blob& b) -> bool
    {
        return a.getIntensity() > b.getIntensity();
    };

    auto sortByCircularity = [&](const Blob& a, const Blob& b) -> bool
    {
        return a.getCircularity() < b.getCircularity();
    };

    auto sortByArea = [&](const Blob& a, const Blob& b) -> bool
    {
        return a.getArea() > b.getArea();
    };

    sortedByIntensivity = filteredBlobs;
    sort(sortedByIntensivity.begin(), sortedByIntensivity.end(), sortByIntensivity);

    sortedByCircularity = filteredBlobs;
    sort(sortedByCircularity.begin(), sortedByCircularity.end(), sortByCircularity);

    sortedByArea = filteredBlobs;
    sort(sortedByArea.begin(), sortedByArea.end(), sortByArea);
    
}

