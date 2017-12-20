#include <iostream>
#include <thread>
#include <algorithm>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/logger.h>
#include <libfreenect2/registration.h>

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

#include "server.h"

# define TABLE_WIDTH 0.946
# define TABLE_LENGTH 1.562

#define VIS3D
#define VIS2D

using namespace std;
using namespace libfreenect2;

typedef Freenect2Device::IrCameraParams DepthCameraParams;
typedef Freenect2Device::ColorCameraParams ColorCameraParams;

bool running = true;
bool calibrated = false;
cv::Ptr<cv::aruco::Dictionary> dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
cv::Mat colorCamMat;
cv::Mat colorDistCoeffs;
cv::RNG rng(2345);

struct TrackingBox {

    TrackingBox(string id, float minX, float maxX, float minY, float maxY, float minZ, float maxZ) :
            id(id), minX(minX), maxX(maxX), minY(minY), maxY(maxY), minZ(minZ), maxZ(maxZ) {}

    bool isInside(cv::Point3f& p) {
        return (p.x > minX && p.x < maxX &&
                p.y > minY && p.y < maxY &&
                p.z > minZ && p.z < maxZ);
    }

    bool checkAndInsert(cv::Point3f& p) {
        if (isInside(p)) {
            points.push_back(p);
            return true;
        }
        return false;
    }

    void reset() {
        top = cv::Point3f();
        points.clear();
    }

    static bool comparePoints(cv::Point3f p0, cv::Point3f p1) {
        return (p0.z < p1.z);
    }

    void sort() {
        ::sort(points.begin(), points.end(), comparePoints);
    }

    cv::Point3f computePosition(int averageCnt) {

        if (points.size() < averageCnt) {
            top = cv::Point3f();
            return top;
        }

        vector<cv::Point3f> topPoints(points.end() - averageCnt, points.end());

        cv::Mat covar, mean;
        cv::Mat samples = cv::Mat(topPoints).reshape(1).t();
        cv::calcCovarMatrix(samples, covar, mean, CV_COVAR_NORMAL | CV_COVAR_COLS);

        double min, max;
        cv::minMaxIdx(covar, &min, &max);

        if (max > 0.1) {
            top = cv::Point3f();
            return top;
        }

        top = cv::Point3f(mean);

        return top;
    }

    float minX, maxX, minY, maxY, minZ, maxZ;
    cv::Point3f top;
    vector<cv::Point3f> points;
    string id;
};

cv::Point3f unprojectPoint(float rawDepth, int u, int v, DepthCameraParams& params) {

    float z = rawDepth / 1000.0f;
    float x = z * (u - params.cx) / params.fx;
    float y = z * (v - params.cy) / params.fy;

    return cv::Point3f(x, y, z);
}

void transformAndSplit(Frame* depthFrame, DepthCameraParams& params, cv::Affine3f transformation, float zThresh,
                       vector<TrackingBox>& boxes) {

    float thresh = zThresh * 1000.0f;

    for (size_t u = 0; u < depthFrame->width; ++u) {
        for (size_t v = 0; v < depthFrame->height; ++v) {
            float d = ((float*) depthFrame->data)[u + v * 512];

            if (d > 0.0f && d < thresh) {
                cv::Point3f p = transformation * unprojectPoint(d, u, v, params);

                for (auto& b : boxes) {
                    if (b.checkAndInsert(p)) {
                        break;
                    }
                }
            }
        }
    }
}

bool
findTransformation(cv::Mat& colorImage, cv::Mat& cameraMatrix, cv::Mat distCoeffs, float markerLength,
                   cv::Affine3f& result) {

    // Flip color image
    cv::Mat tmp;
    cv::flip(colorImage, tmp, 1);

    // Find markers in rgb image
    vector<vector<cv::Point2f>> corners;
    vector<int> ids;
    cv::aruco::detectMarkers(tmp, dict, corners, ids);
    if (corners.size() < 4) return false;

    int id0, id1, id2, id3;
    for (int i = 0; i < ids.size(); ++i) {
        if (ids[i] == 0) id0 = i;
        else if (ids[i] == 1) id1 = i;
        else if (ids[i] == 2) id2 = i;
        else if (ids[i] == 3) id3 = i;
        else return false;
    }

    cout << "Found markers." << endl;

    // Flip marker points for usage in original image
    for (auto& a : corners) {
        for (auto& c: a) {
            c.x = colorImage.cols - c.x;
        }
    }

    // Estimate marker positions in 3D
    vector<cv::Vec3d> tvecs, rvecs;
    cv::aruco::estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs);

    // Compute center
    cv::Vec3f center;
    for (auto& t: tvecs) center += t;
    center /= (float) tvecs.size();

    // Compute axes
    // TODO: consistent coordinate systems!
    cv::Vec3f x = cv::normalize(tvecs[id1] - tvecs[id0]);
    cv::Vec3f y = cv::normalize(tvecs[id3] - tvecs[id0]);
    cv::Vec3f z = x.cross(y);
    x = y.cross(z); // ensure perpendicularity

    // Create transformation matrix
    result = cv::Affine3f::Identity();
    result = result.translate(-center);
    vector<cv::Vec3f> m = {x, y, z};
    cv::Mat3f r(m);
    result = result.rotate(cv::Matx33f((float*) r.ptr()).t());

    cout << "Camera transformation:" << endl;
    cout << result.matrix << endl;

#ifdef VIS2D
    cv::aruco::drawDetectedMarkers(colorImage, corners, ids);
    cv::Vec3f rvec;
    cv::Rodrigues(result.rotation(), rvec);
    cv::aruco::drawAxis(colorImage, cameraMatrix, distCoeffs, rvec, center, 0.1f);
    cv::imshow("markers", colorImage);
#endif

    cv::FileStorage fs("currentCameraTransformation.yaml", cv::FileStorage::WRITE);
    fs << "cameraTransformation" << cv::Mat(result.matrix);
    fs.release();

    return true;
}


void siginthandler(int s) {
    running = false;
}

void loadCalibration(string serial, cv::Mat& cameraMat, cv::Mat& distCoeffs) {
    string fileName = "calib_data/" + serial + "/calib_color.yaml";
    cv::FileStorage fs(fileName, cv::FileStorage::READ);

    if (!fs.isOpened()) {
        cerr << "Could not open calibration file " << fileName << endl;
        return;
    }

    fs["cameraMatrix"] >> cameraMat;
    fs["distortionCoefficients"] >> distCoeffs;

    fs.release();
}

int main(int argc, char* argv[]) {

    cv::Affine3f transformation = cv::Affine3f::Identity();

    if (argc > 1) {
        string fileName = argv[1];
        cv::FileStorage fs(fileName, cv::FileStorage::READ);

        if (!fs.isOpened()) {
            cerr << "Could not open camera transformation file " << fileName << endl;
            return 1;
        }

        cv::Mat tmp;
        fs["cameraTransformation"] >> tmp;
        transformation.matrix = tmp;
        fs.release();
        cout << "Loaded camera transformation from " << fileName << endl;
        calibrated = true;
    }

    Freenect2 freenect2;
    Freenect2Device* dev = 0;

    setGlobalLogger(createConsoleLogger(Logger::Warning));

    if (freenect2.enumerateDevices() == 0) {
        std::cerr << "No kinect connected!" << std::endl;
        return -1;
    }

    string serial = freenect2.getDefaultDeviceSerialNumber();

    PacketPipeline* pipeline = 0;

    pipeline = new libfreenect2::CudaPacketPipeline;

    dev = freenect2.openDevice(serial, pipeline);

    if (dev == 0) {
        std::cerr << "Failure opening kinect!" << std::endl;
        return -1;
    }

    SyncMultiFrameListener listener(libfreenect2::Frame::Color | Frame::Depth | libfreenect2::Frame::Ir);
    dev->setIrAndDepthFrameListener(&listener);
    dev->setColorFrameListener(&listener);

    dev->start();

    DepthCameraParams depthParams = dev->getIrCameraParams();
    ColorCameraParams colorParams = dev->getColorCameraParams();
    loadCalibration(serial, colorCamMat, colorDistCoeffs); // TODO

    FrameMap frames;

    Server server;
    server.start();


    vector<TrackingBox> boxes;
    boxes.push_back({"B0", -TABLE_WIDTH / 2 - 0.4f, 0, 0.0f, TABLE_LENGTH / 2, 0.3f, 1.5f});
    boxes.push_back({"B1", -TABLE_WIDTH / 2 - 0.4f, 0, -TABLE_LENGTH / 2, 0.0f, 0.3f, 1.5f});
    boxes.push_back({"B2", 0, TABLE_WIDTH / 2 + 0.4f, 0.0f, TABLE_LENGTH / 2, 0.3f, 1.5f});
    boxes.push_back({"B3", 0, TABLE_WIDTH / 2 + 0.4f, -TABLE_LENGTH / 2, 0.0f, 0.3f, 1.5f});


#ifdef VIS3D
    cv::viz::Viz3d window3D("Viz");
    cv::viz::WPlane table_w(cv::Point3d(0, 0, 0), cv::Vec3d(0, 0, 1), cv::Vec3d(0, 1, 0),
                            cv::Size2d(TABLE_WIDTH, TABLE_LENGTH));
    cv::viz::WCameraPosition origin_w;
    cv::viz::WCameraPosition camera_w(cv::Vec2d(1.22, 1.04));
    cv::viz::WPlane floor_w(cv::Point3d(0, 0, -1), cv::Vec3d(0, 0, 1), cv::Vec3d(0, 1, 0), cv::Size2d(5, 5),
                            cv::viz::Color::gray());
    window3D.showWidget("table", table_w);
    window3D.showWidget("floor", floor_w);
    window3D.showWidget("origin", origin_w);
    window3D.showWidget("camera", camera_w);
    vector<cv::viz::WSphere> positions_w;
    for (auto& b: boxes) {
        cv::viz::WSphere s(cv::Point3d(0, 0, 0), 0.1, 10, cv::viz::Color(rng(255), rng(255), rng(255)));
        window3D.showWidget(b.id + "_position", s);
        positions_w.push_back(s);
        cv::viz::WCube c(cv::Point3d(b.minX, b.minY, b.minZ), cv::Point3d(b.maxX, b.maxY, b.maxZ));
        window3D.showWidget(b.id + "_cube", c);
    }
#endif

    signal(SIGINT, siginthandler);

    while (running) {

        listener.waitForNewFrame(frames);

        Frame* depth = frames[Frame::Depth];
        Frame* color = frames[Frame::Color];

        cv::Mat colorImage(color->height, color->width, CV_8UC4, color->data);
        cv::Mat depthImage(depth->height, depth->width, CV_32FC1, depth->data);

        cv::Mat rgb;
        cv::cvtColor(colorImage, rgb, CV_RGBA2RGB);

#ifdef VIS2D
        cv::imshow("Depth", depthImage / 4096.0f);
        cv::imshow("Color", rgb);
        cv::waitKey(1);
#endif

        if (!calibrated) {
            calibrated = findTransformation(rgb, colorCamMat, colorDistCoeffs, 0.059f, transformation);
            listener.release(frames);
            continue;
        }

        cv::medianBlur(depthImage, depthImage, 3);

        for (auto& b : boxes) b.reset();
        transformAndSplit(depth, depthParams, transformation, 5.0f, boxes);

        for (auto& b : boxes) {
            b.sort();
            b.computePosition(50);
        }


#ifdef VIS3D
        for (auto& b : boxes) {
            cv::Point3f pos = cv::Vec3f(b.top);
            if (pos == cv::Point3f()) continue;
            cv::Affine3d pose;
            pose.translation(cv::Vec3f(b.top));
            window3D.setWidgetPose(b.id + "_position", pose);
            vector<cv::Point3f> points(1, cv::Point3f());
            cv::viz::WCloud cloud(points);
            if (!b.points.empty()) {
                cloud = cv::viz::WCloud(b.points);
            }
            window3D.showWidget(b.id + "_cloud", cloud);
        }
        camera_w.setPose(transformation);

        window3D.spinOnce(1, true);
#endif

        listener.release(frames);

        vector<vector<float>> trackingData;
        for (size_t i = 0; i < boxes.size(); ++i) {
            vector<float> v = {(float) i, boxes[i].top.x, boxes[i].top.y, boxes[i].top.z};
            trackingData.push_back(v);
        }
        server.send(trackingData);


    }

    server.stop();
    dev->stop();
    dev->close();

    return 0;

}
