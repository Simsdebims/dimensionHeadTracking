#include "TrackingBox.h"

#include <thread>

using namespace std;
using namespace cv;


TrackingBox::TrackingBox(string id, float minX, float maxX, float minY, float maxY, float minZ, float maxZ) :
        id(id), minX(minX), maxX(maxX), minY(minY), maxY(maxY), minZ(minZ), maxZ(maxZ), minCnt(40) {}

bool TrackingBox::isInside(const Point3f& p) const {
    return (p.x > minX && p.x < maxX &&
            p.y > minY && p.y < maxY &&
            p.z > minZ && p.z < maxZ);
}

bool TrackingBox::checkAndInsert(const Point3f& p) {
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

bool TrackingBox::comparePoints(const Point3f& p0, const Point3f& p1) {
    return (p0.z < p1.z);
}

void TrackingBox::sort() {
    std::sort(points.begin(), points.end(), comparePoints);
}

Point3f TrackingBox::computePosition() {

    if (points.size() < minCnt) {
        top = Point3f();
        return top;
    }

    sort();

    vector<Point3f> topPoints(points.end() - minCnt, points.end());

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

void TrackingBoxList::computePositions() {
    for (auto& b : boxes) {
        b.computePosition();
    }
}

Point3f unprojectPoint(float rawDepth, int u, int v, const Mat& params) {

    static auto fx = static_cast<float>(params.at<double>(0, 0));
    static auto fy = static_cast<float>(params.at<double>(1, 1));
    static auto cx = static_cast<float>(params.at<double>(0, 2));
    static auto cy = static_cast<float>(params.at<double>(1, 2));

    float z = rawDepth / 1000.0f;
    float x = z * (u - cx) / fx;
    float y = z * (v - cy) / fy;

    return Point3f(x, y, z);
}

void TrackingBoxList::fill(const uchar* depthFrame, const Mat& cameraMatrix, const Affine3f& transformation, float zThresh) {

    float thresh = zThresh * 1000.0f;

    int width = 512;
    int height = 424;

    int numberOfThreads = 4;
    int columnCnt = height / numberOfThreads;
    int from = 0;
    vector<thread> workers;
    for (int i = 0; i < numberOfThreads; i++) {
        int to = min(from + columnCnt, width);

        workers.emplace_back([&, from, to]() {
            for (int u = 0; u < width; ++u) {
                for (int v = from; v < to; ++v) {

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
        });
        from = to + 1;
    }

    for_each(workers.begin(), workers.end(), [](thread& t) {
        t.join();
    });
}

vector<vector<float>> TrackingBoxList::getTrackingData() const {
    vector<vector<float>> trackingData;
    for (size_t i = 0; i < boxes.size(); ++i) {
        vector<float> v = {(float) i, boxes[i].top.x, boxes[i].top.y, boxes[i].top.z};
        trackingData.push_back(v);
    }
    return trackingData;
}


