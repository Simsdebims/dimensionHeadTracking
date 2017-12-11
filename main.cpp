#include <iostream>
#include <thread>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/logger.h>
#include <libfreenect2/registration.h>

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
//#include <opencv2/viz.hpp>

#include "server.h"

using namespace std;
using namespace libfreenect2;

typedef Freenect2Device::IrCameraParams DepthCameraParams;
typedef Freenect2Device::ColorCameraParams ColorCameraParams;

bool running = true;
bool calibrated = false;
cv::Ptr<cv::aruco::Dictionary> dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
cv::Mat colorCamMat;
cv::Mat colorDistCoeffs;

struct TrackingBox {

    TrackingBox(string id, float minX, float maxX, float minY, float maxY, float minZ, float maxZ) :
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

    void reset() {
        top = cv::Point3f();
        points.clear();
    }

    void refine(float topRange = 0.05f) {
        float thresh = top.y - topRange;
        cv::Point3f sum;
        int cnt = 0;
        for (auto& p : points) {
            if (p.y > thresh) {
                sum += p;
                cnt++;
            }
        }
        top = sum / cnt;
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
findTransformation(cv::Mat& colorImage, cv::Mat& cameraMatrix, cv::Mat distCoeffs, float markerLength, cv::Affine3f& result) {

    // Find markers in rgb image

    vector<vector<cv::Point2f>> corners;
    vector<int> ids;

    cv::aruco::detectMarkers(colorImage, dict, corners, ids);

    if (corners.size() < 4) return false;

    cout << "Found markers." << endl;

    cv::aruco::drawDetectedMarkers(colorImage, corners, ids);
    cv::imshow("markers", colorImage);

    vector<cv::Vec3d> tvecs, rvecs;
    cv::aruco::estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs);

    cv::Vec3f translation(0.0f,0.0f,0.0f);
    for (auto& t: tvecs) {
        translation += t;
    }
    translation /= (float) tvecs.size();

    // TODO: average rotations?

    result.rotation(rvecs[0]);
    result.translation(translation);

    cout << "Camera transformation:" << endl;
    cout << result.matrix << endl;

    return true;
}


void siginthandler(int s) {
    running = false;
}

void loadCalibration(string serial, cv::Mat& cameraMat, cv::Mat& distCoeffs) {
    string filenName = "calib_data/" + serial + "/calib_color.yaml";
    cv::FileStorage fs(filenName, cv::FileStorage::READ);

    if (!fs.isOpened()) {
        cerr << "Could not open calibration file " << filenName << endl;
        return;
    }

    fs["cameraMatrix"] >> cameraMat;
    fs["distortionCoefficients"] >> distCoeffs;

    fs.release();
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
    loadCalibration(serial, colorCamMat, colorDistCoeffs); // TODO


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


    vector<TrackingBox> boxes;
    boxes.push_back({"B0", -1.5f, -0.5f, 0.0f, 2.5f, 0.0f, 1.0f});
    boxes.push_back({"B1", -1.5f, -0.5f, 0.0f, 2.5f, -1.0f, 0.0f});
    boxes.push_back({"B2", 0.5f, 1.5f, 0.0f, 2.5f, 0.0f, 1.0f});
    boxes.push_back({"B3", 0.5f, 1.5f, 0.0f, 2.5f, -1.0f, 0.0f});


    // Visualization
//    cv::viz::Viz3d window3D("Viz");
//    cv::viz::WPlane table_w(cv::Point3d(0, 0, 0), cv::Vec3d(0, 1, 0), cv::Vec3d(0, 0, 1), cv::Size2d(1, 2));
//    window3D.showWidget("table", table_w);
//    vector<cv::viz::WSphere> positions_w;
//    for (auto& b: boxes) {
//        cv::viz::WSphere s(cv::Point3d(0, 0, 0), 0.1);
//        window3D.showWidget(b.id, s);
//        positions_w.push_back(s);
//    }

    signal(SIGINT, siginthandler);

    while (running) {

        listener.waitForNewFrame(frames);

        Frame* depth = frames[Frame::Depth];
        Frame* color = frames[Frame::Color];
        Frame* ir = frames[Frame::Ir];

        registration->apply(color, depth, &undistorted, &registered, true, &depth2rgb);

        cv::Mat colorImage(color->height, color->width, CV_8UC4, color->data);
        cv::Mat depthImage(depth->height, depth->width, CV_32FC1, depth->data);
        cv::Mat undistortedImage(depth->height, depth->width, CV_32FC1, depth->data);
        cv::Mat registeredImage(registered.height, registered.width, CV_8UC4, registered.data);
        cv::Mat depth2rgbImage(depth2rgb.height, depth2rgb.width, CV_32FC1, depth2rgb.data);

        cv::Mat rgb, mappedDepth;
        cv::cvtColor(colorImage, rgb, CV_RGBA2RGB);
        cv::flip(rgb, rgb, 1);
        cv::flip(depth2rgbImage, mappedDepth, 1);


        cv::imshow("Depth", depthImage / 4096.0f);

        cv::waitKey(1);

        if (!calibrated) {
            calibrated = findTransformation(rgb, colorCamMat, colorDistCoeffs, 0.016f, transformation);
            listener.release(frames);
            continue;
        }

        for (auto& b : boxes) b.reset();

        transformAndSplit(depth, depthParams, transformation, 5.0f, boxes);

        for (auto& b : boxes) b.refine();

        // Visualization
//        for (auto& b : boxes) {
//            cv::Point3f pos = cv::Vec3f(b.top);
//            if (pos == cv::Point3f()) continue;
//            cv::Affine3d pose;
//            pose.translation(cv::Vec3f(b.top));
//            window3D.setWidgetPose(b.id, pose);
//        }
//        window3D.spinOnce(1, true);


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
