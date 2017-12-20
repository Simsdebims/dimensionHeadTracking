#pragma once

#include <opencv2/opencv.hpp>

class TrackingBox {

public:

    TrackingBox(std::string id, float minX, float maxX, float minY, float maxY, float minZ, float maxZ);

    bool isInside(cv::Point3f& p);

    bool checkAndInsert(cv::Point3f& p);

    void reset();

    static bool comparePoints(cv::Point3f p0, cv::Point3f p1);

    void sort();

    cv::Point3f computePosition(int averageCnt);

    float minX, maxX, minY, maxY, minZ, maxZ;
    cv::Point3f top;
    std::vector<cv::Point3f> points;
    std::string id;
};

class TrackingBoxList {

public:

    void insert(std::initializer_list<TrackingBox> list);

    void insert(TrackingBox b);

    void resetAll();

    void sortAll();

    void computePositions(int averageCnt = 50);

    void fill(const uchar* depthFrame, cv::Mat cameraMatrix, cv::Affine3f transformation, float zThresh = 5.0f);

    std::vector<std::vector<float>> getTrackingData();

    std::vector<TrackingBox> boxes;

};