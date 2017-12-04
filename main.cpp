#include <iostream>
#include <thread>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/logger.h>
#include <libfreenect2/registration.h>

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

#include "server.h"

using namespace std;
using namespace libfreenect2;

typedef Freenect2Device::IrCameraParams DepthCameraParams;
typedef Freenect2Device::ColorCameraParams ColorCameraParams;

struct BoundingBox {

    BoundingBox(string id, float minX, float maxX, float minY, float maxY, float minZ, float maxZ) :
            id(id), minX(minX), maxX(maxX), minY(minY), maxY(maxY), minZ(minZ), maxZ(maxZ) { }

    bool isInside(cv::Point3f& p) {
        return (p.x > minX && p.x < maxX &&
                p.y > minY && p.y < maxY &&
                p.z > minZ && p.z < maxZ);
    }

    bool checkAndInsert(cv::Point3f& p) {
        if (isInside(p)) {
            if (p.y > top.y) {
                top = p;
            }
            points.push_back(p);
            return true;
        }
        return false;
    }

    float minX, maxX, minY, maxY, minZ, maxZ;
    cv::Point3f top{0.0f, 0.0f, 0.0f};
    vector<cv::Point3f> points;
    string id;
};

cv::Point3f unprojectPoint(float depth, int u, int v, DepthCameraParams& params) {

    float z = depth / 1000.0f;
    float x = z * (u - params.cx) / params.fx;
    float y = z * (v - params.cy) / params.fy;

    return cv::Point3f(x, y, z);
}

void transformAndSplit(Frame* depthFrame, DepthCameraParams& params, cv::Affine3f transformation, float zThresh,
                       vector<BoundingBox>& boxes) {

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
findTransformation(cv::Mat& colorImage, cv::Mat& registeredDepthImage, DepthCameraParams& depthParams,
                   cv::Affine3f& result) {

    cv::Ptr<cv::aruco::Dictionary> dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250); // TODO
    vector<vector<cv::Point2f>> corners;
    vector<int> ids;

    while (corners.size() < 4) {
        corners.clear();
        ids.clear();
        cv::aruco::detectMarkers(colorImage, dict, corners, ids);
    }

    if (corners.size() < 4) return false;

    cout << "Found markers." << endl;

    vector<cv::Point3f> markers;

    for (int i = 0; i < 4; ++i) {
        int u = (int) corners[i][0].x;
        int v = (int) corners[i][0].y;
        float d = ((float*) registeredDepthImage.data)[u + (v + 1) * 512];
        if (d <= 0.0f || d > 10.0f) return false;
        markers.push_back(unprojectPoint(d, u, v, depthParams));
    }

    cv::Point3f from = (markers[0] + markers[1] + markers[2] + markers[3]) / 4.0f;
    cv::Vec3f dir = cv::normalize(cv::Vec3f(markers[2] - markers[1]));
    cv::Vec3f up = dir.cross(markers[0] - markers[1]);

    dir = cv::normalize(dir);
    cv::Vec3f right = up.cross(dir);
    right = cv::normalize(right);
    cv::Vec3f newup = dir.cross(right);

    // Create transformation matrix
    cv::Matx44f transform = cv::Matx44f::eye();
    vector<cv::Vec3f> m = {right, newup, dir};
    cv::Mat3f r(m);
    result.rotation(cv::Matx33f((float*) r.ptr()).t());
    result.translation(from);

    cout << "Success. Transformation: " << endl;
    cout << result.matrix << endl;

    return true;
}


bool running = true;
bool calibrated = false;

void siginthandler(int s) {
    running = false;
}

int main() {

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
//    pipeline = new libfreenect2::OpenGLPacketPipeline;
//    pipeline = new libfreenect2::CpuPacketPipeline();

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

    Registration* registration = new Registration(depthParams, colorParams);
    Frame undistorted(512, 424, 4);
    Frame registered(512, 424, 4);
    Frame depth2rgb(1920, 1080 + 2, 4);

    FrameMap frames;


    float angle = (45.0f * 3.14159265f) / 180.0f;
    float r[3][3] =
            {
                    {1, 0,           0},
                    {0, cos(angle),  sin(angle)},
                    {0, -sin(angle), cos(angle)}
            };
    cv::Mat m(3, 3, CV_32FC1, r);
    cv::Affine3f transformation = cv::Affine3f::Identity();

    Server server;
    server.start();

    signal(SIGINT, siginthandler);

    while (running) {

        listener.waitForNewFrame(frames);

        Frame* depth = frames[Frame::Depth];
        Frame* color = frames[Frame::Color];
        Frame* ir = frames[Frame::Ir];

        registration->apply(color, depth, &undistorted, &registered, true, &depth2rgb);

        cv::Mat colorImage(color->height, color->width, CV_8UC4, color->data);
        cv::Mat depthImage(depth->height, depth->width, CV_32FC1, depth->data);
        cv::Mat registeredImage(registered.height, registered.width, CV_8UC4, registered.data);
        cv::Mat depth2rgbImage(depth2rgb.height, depth2rgb.width, CV_32FC1, depth2rgb.data);

        cv::Mat rgb;
        cv::cvtColor(colorImage, rgb, CV_RGBA2RGB);

/*
        cv::imshow("Color", colorImage);
        cv::imshow("Color2", rgb);
        cv::imshow("Registered", registeredImage);
        cv::imshow("Depth", depthImage / 4096.0f);
        cv::imshow("depth2rgb", depth2rgbImage / 4096.0f);
        cv::waitKey(1);
*/


        if (!calibrated) {
            cout << "Searching for markers" << endl;
            cout << rgb.channels() << endl;
            calibrated = findTransformation(rgb, depth2rgbImage, depthParams, transformation);
            listener.release(frames);
            continue;
        }

/*
        vector<BoundingBox> boxes;
        boxes.push_back({"B0", -0.5f, -1.5f, 0.0f, 2.5f, 0.0f, 1.0f});
        boxes.push_back({"B1", -0.5f, -1.5f, 0.0f, 2.5f, -1.0f, 0.0f});
        boxes.push_back({"B2", 0.5f, 1.5f, 0.0f, 2.5f, 0.0f, 1.0f});
        boxes.push_back({"B3", 0.5f, 1.5f, 0.0f, 2.5f, -1.0f, 0.0f});
        transformAndSplit(depth, depthParams, transformation, 5.0f, boxes);
*/

        listener.release(frames);
/*
        vector<vector<float>> trackingData;
        for (size_t i = 0; i < boxes.size(); ++i) {
            vector<float> v = {(float) i, boxes[i].top.x, boxes[i].top.y, boxes[i].top.z};
            trackingData.push_back(v);
        }
        server.send(trackingData);
 */

    }

    server.stop();
    dev->stop();
    dev->close();

    delete registration;

    return 0;

}
