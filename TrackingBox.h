#pragma once

#include <opencv2/opencv.hpp>

#define VIS3D

# define TABLE_WIDTH 0.946f
# define TABLE_LENGTH 1.562f

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

    TrackingBoxList(cv::Affine3f transformation);

    void insert(std::initializer_list<TrackingBox> list);

    void insert(TrackingBox b);

    void resetAll();

    void computePositions();

    void fill(const uchar* depthFrame, const cv::Mat& cameraMatrix, float zThresh = 5.0f);

    std::vector<std::vector<float>> getTrackingData() const;

    std::vector<TrackingBox> boxes;

private:

    cv::Affine3f transformation;

#ifdef VIS3D

    cv::viz::Viz3d window3D{"Viz"};
    cv::viz::WPlane table_w{cv::Point3d(0, 0, 0), cv::Vec3d(0, 0, 1), cv::Vec3d(0, 1, 0),
                            cv::Size2d(TABLE_WIDTH, TABLE_LENGTH)};
    cv::viz::WCameraPosition origin_w;
    cv::viz::WCameraPosition camera_w{cv::Vec2d(1.22, 1.04)};
    cv::viz::WPlane floor_w{cv::Point3d(0, 0, -1), cv::Vec3d(0, 0, 1), cv::Vec3d(0, 1, 0), cv::Size2d(5, 5),
                            cv::viz::Color::gray()};
    std::vector<cv::viz::WSphere> positions_w;
    cv::viz::WText fps_w{"0.0", cv::Point(10, 10)};

    cv::RNG rng{2345};

    clock_t time = clock();

public:

    void visualize();

#endif

};