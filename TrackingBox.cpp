#include "TrackingBox.h"

using namespace std;
using namespace cv;


TrackingBox::TrackingBox(string id, float minX, float maxX, float minY, float maxY, float minZ, float maxZ) :
        id(id), minX(minX), maxX(maxX), minY(minY), maxY(maxY), minZ(minZ), maxZ(maxZ) {}

bool TrackingBox::isInside(Point3f& p) {
    return (p.x > minX && p.x < maxX &&
            p.y > minY && p.y < maxY &&
            p.z > minZ && p.z < maxZ);
}

bool TrackingBox::checkAndInsert(Point3f& p) {
    if (isInside(p)) {
        points.push_back(p);
        return true;
    }
    return false;
}

void TrackingBox::reset() {
    top = Point3f();
    points.clear();
}

bool TrackingBox::comparePoints(Point3f p0, Point3f p1) {
    return (p0.z < p1.z);
}

void TrackingBox::sort() {
    std::sort(points.begin(), points.end(), comparePoints);
}

Point3f TrackingBox::computePosition(int averageCnt) {

    if (points.size() < averageCnt) {
        top = Point3f();
        return top;
    }

    vector<Point3f> topPoints(points.end() - averageCnt, points.end());

    Mat covar, mean;
    Mat samples = Mat(topPoints).reshape(1).t();
    calcCovarMatrix(samples, covar, mean, CV_COVAR_NORMAL | CV_COVAR_COLS);

    double min, max;
    minMaxIdx(covar, &min, &max);

    if (max > 0.1) {
        top = Point3f();
        return top;
    }

    top = Point3f(mean);

    return top;
}

void TrackingBoxList::insert(std::initializer_list<TrackingBox> list) {
    for (auto b : list) insert(b);
}

void TrackingBoxList::insert(TrackingBox b) {
    boxes.push_back(b);
}

void TrackingBoxList::resetAll() {
    for (auto& b : boxes) {
        b.reset();
    }
}

void TrackingBoxList::sortAll() {
    for (auto& b : boxes) {
        b.sort();
    }
}

void TrackingBoxList::computePositions(int averageCnt) {
    for (auto& b : boxes) {
        b.computePosition(averageCnt);
    }
}

Point3f unprojectPoint(float rawDepth, int u, int v, Mat& params) {

    static auto fx = static_cast<float>(params.at<double>(0, 0));
    static auto fy = static_cast<float>(params.at<double>(1, 1));
    static auto cx = static_cast<float>(params.at<double>(0, 2));
    static auto cy = static_cast<float>(params.at<double>(1, 2));

    float z = rawDepth / 1000.0f;
    float x = z * (u - cx) / fx;
    float y = z * (v - cy) / fy;

    return Point3f(x, y, z);
}

void TrackingBoxList::fill(const uchar* depthFrame, Mat cameraMatrix, Affine3f transformation, float zThresh) {

    float thresh = zThresh * 1000.0f;

    int width = 512;
    int height = 424;

    for (int u = 0; u < width; ++u) {
        for (int v = 0; v < height; ++v) {
            float d = ((float*) depthFrame)[u + v * width];

            if (d > 0.0f && d < thresh) {
                cv::Point3f p = transformation * unprojectPoint(d, u, v, cameraMatrix);

                for (auto& b : boxes) {
                    if (b.checkAndInsert(p)) {
                        break;
                    }
                }
            }
        }
    }
}

vector<vector<float>> TrackingBoxList::getTrackingData() {
    vector<vector<float>> trackingData;
    for (size_t i = 0; i < boxes.size(); ++i) {
        vector<float> v = {(float) i, boxes[i].top.x, boxes[i].top.y, boxes[i].top.z};
        trackingData.push_back(v);
    }
    return trackingData;
}


