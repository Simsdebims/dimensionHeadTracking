#pragma once

#include <opencv2/opencv.hpp>

class TrackingBox {

public:

    TrackingBox(std::string id, float minX, float maxX, float minY, float maxY, float minZ, float maxZ);

    bool isInside(const cv::Point3f& p) const;

    bool checkAndInsert(const cv::Point3f& p);

    void reset();

    static bool comparePoints(const cv::Point3f& p0, const cv::Point3f& p1);

    cv::Point3f computePosition();

    float minX, maxX, minY, maxY, minZ, maxZ;
    cv::Point3f top;
    std::vector<cv::Point3f> points;
    std::string id;

private:

    void sort();

    int minCnt;
};

class TrackingBoxList {

public:

    void insert(std::initializer_list<TrackingBox> list);

    void insert(TrackingBox b);

    void resetAll();

    void computePositions();

    void fill(const uchar* depthFrame, const cv::Mat& cameraMatrix, const cv::Affine3f& transformation, float zThresh = 5.0f);

    std::vector<std::vector<float>> getTrackingData() const;

    std::vector<TrackingBox> boxes;

};