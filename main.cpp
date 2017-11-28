#include <iostream>
#include <thread>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>

#include <opencv2/opencv.hpp>

#include "server.h"


using namespace std;

libfreenect2::Freenect2Device::IrCameraParams cameraParams;

struct Box {

    Box(string id, float minX, float maxX, float minY, float maxY, float minZ, float maxZ) :
            id(id), minX(minX), maxX(maxX), minY(minY), maxY(maxY), minZ(minZ), maxZ(maxZ) { }

    bool isInside(cv::Point3f& p) {
        if (p.x > minX && p.x < maxX &&
            p.y > minY && p.y < maxY &&
            p.z > minZ && p.z < maxZ) {
            return true;
        }
        return false;
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
    cv::Point3f top{0, -100.0f, 0};
    vector<cv::Point3f> points;
    string id;
};

void checkAndSplit(libfreenect2::Frame* depthFrame, float xTilt, float offsetZ, float offsetY, vector<Box>& boxes) {

    float angle = (xTilt * 3.14159265f) / 180.0f;
    float r[3][3] =
            {
                    {1, 0,           0},
                    {0, cos(angle),  sin(angle)},
                    {0, -sin(angle), cos(angle)}
            };
    cv::Mat m(3, 3, CV_32FC1, r);
    cv::Affine3f transformation = cv::Affine3f::Identity();
    transformation.rotation(m);
    transformation.translation(cv::Point3f(0.0f, offsetY, offsetZ));

    for (size_t u = 0; u < depthFrame->width; ++u) {
        for (size_t v = 0; v < depthFrame->height; ++v) {
            float d = ((float*) depthFrame->data)[u + v * 512];

            if (d > 0.0f && d < 5000.0f) {
                float z = d / 1000.0f;
                float x = z * (u - cameraParams.cx) / cameraParams.fx;
                float y = z * (v - cameraParams.cy) / cameraParams.fy;

                cv::Point3f p(x, y, z);
                p = transformation * p;

                for (auto& b : boxes) {
                    if (b.checkAndInsert(p)) {
                        break;
                    }
                }


            }
        }
    }
}

bool running = true;

void siginthandler(int s) {
    running = false;
}

int main() {

    libfreenect2::Freenect2 freenect2;
    libfreenect2::Freenect2Device* dev = 0;

    if (freenect2.enumerateDevices() == 0) {
        std::cout << "no device connected!" << std::endl;
        return -1;
    }

    string serial = freenect2.getDefaultDeviceSerialNumber();

    std::cout << "SERIAL: " << serial << std::endl;

    dev = freenect2.openDevice(serial);

    if (dev == 0) {
        std::cout << "failure opening device!" << std::endl;
        return -1;
    }

    libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Depth |
                                                  libfreenect2::Frame::Ir);
    dev->setIrAndDepthFrameListener(&listener);

    libfreenect2::FrameMap frames;


    dev->start();

    cameraParams = dev->getIrCameraParams();

    cv::Mat depthImage(424, 512, CV_32FC1);

    cv::namedWindow("depth", cv::WINDOW_AUTOSIZE);

    Server server;
    server.start();

    signal(SIGINT, siginthandler);

    while (running) {

        listener.waitForNewFrame(frames);

        libfreenect2::Frame* ir = frames[libfreenect2::Frame::Ir];
        libfreenect2::Frame* depth = frames[libfreenect2::Frame::Depth];

        vector<Box> boxes;
        boxes.push_back(Box("B1", -1.5f, -0.5f, 0.0f, 2.5f, 0.0f, 1.0f));
        boxes.push_back(Box("B2", 0.5f, 1.5f, 0.0f, 2.5f, 0.0f, 1.0f));

        checkAndSplit(depth, 45.0f, -1.0f, 1.0f, boxes);

        mempcpy(depthImage.data, depth->data, 424 * 512 * sizeof(float));

        cv::imshow("depth", depthImage);
        cv::waitKey(1);


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