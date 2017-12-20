#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/logger.h>

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

#include "Server.h"
#include "TrackingBox.h"

# define TABLE_WIDTH 0.946f
# define TABLE_LENGTH 1.562f

#define VIS3D
//#define VIS2D

using namespace std;
using namespace cv;
using namespace libfreenect2;

bool running = true;

Ptr<aruco::Dictionary> dict = aruco::getPredefinedDictionary(aruco::DICT_4X4_50);
Mat colorCamMat;
Mat depthCamMat;
Mat colorDistCoeffs;
RNG rng(2345);

bool findTransformation(Mat& colorImage, Mat& cameraMatrix, Mat& distCoeffs, float markerLength, Affine3f& result) {

    // Flip color image
    Mat tmp;
    flip(colorImage, tmp, 1);

    // Find markers in rgb image
    vector<vector<Point2f>> corners;
    vector<int> ids;
    aruco::detectMarkers(tmp, dict, corners, ids);
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
    vector<Vec3d> tvecs, rvecs;
    aruco::estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs);

    // Compute center
    Vec3f center;
    for (auto& t: tvecs) center += t;
    center /= (float) tvecs.size();

    // Compute axes
    Vec3f x = normalize(tvecs[id1] - tvecs[id0]);
    Vec3f y = normalize(tvecs[id3] - tvecs[id0]);
    Vec3f z = x.cross(y);
    x = y.cross(z); // ensure perpendicularity

    // Create transformation matrix
    result = Affine3f::Identity();
    result = result.translate(-center);
    vector<Vec3f> m = {x, y, z};
    Mat3f r(m);
    result = result.rotate(Matx33f((float*) r.ptr()).t());

    cout << "Camera transformation:" << endl;
    cout << result.matrix << endl;

#ifdef VIS2D
    aruco::drawDetectedMarkers(colorImage, corners, ids);
    Vec3f rvec;
    Rodrigues(result.rotation(), rvec);
    aruco::drawAxis(colorImage, cameraMatrix, distCoeffs, rvec, center, 0.1f);
    imshow("markers", colorImage);
#endif

    FileStorage fs("currentCameraTransformation.yaml", FileStorage::WRITE);
    fs << "cameraTransformation" << Mat(result.matrix);
    fs.release();

    return true;
}


void siginthandler(int s) {
    running = false;
}

void loadCalibration(string serial, Mat& colorCameraMat, Mat& distCoeffs, Mat& depthCameraMat) {
    string colorFileName = "calib_data/" + serial + "/calib_color.yaml";
    string depthFileName = "calib_data/" + serial + "/calib_ir.yaml";

    FileStorage fs(colorFileName, FileStorage::READ);

    if (!fs.isOpened()) {
        cerr << "Could not open color calibration file " << colorFileName << endl;
        return;
    }

    fs["cameraMatrix"] >> colorCameraMat;
    fs["distortionCoefficients"] >> distCoeffs;

    fs.release();

    fs.open(depthFileName, FileStorage::READ);

    if (!fs.isOpened()) {
        cerr << "Could not open depth calibration file " << depthFileName << endl;
        return;
    }

    fs["cameraMatrix"] >> depthCameraMat;

    fs.release();
}

int main(int argc, char* argv[]) {

    Affine3f transformation = Affine3f::Identity();

    bool calibrated = false;

    if (argc > 1) {
        string fileName = argv[1];
        FileStorage fs(fileName, FileStorage::READ);

        if (!fs.isOpened()) {
            cerr << "Could not open camera transformation file " << fileName << endl;
            return 1;
        }

        Mat tmp;
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

    PacketPipeline* pipeline = new libfreenect2::CudaPacketPipeline;

    dev = freenect2.openDevice(serial, pipeline);

    if (dev == 0) {
        std::cerr << "Failure opening kinect!" << std::endl;
        return -1;
    }

    loadCalibration(serial, colorCamMat, colorDistCoeffs, depthCamMat);

    FrameMap frames;
    SyncMultiFrameListener listener(libfreenect2::Frame::Color | Frame::Depth | libfreenect2::Frame::Ir);
    dev->setIrAndDepthFrameListener(&listener);
    dev->setColorFrameListener(&listener);
    dev->start();

    Server server;
    server.start();

    TrackingBoxList boxes;
    boxes.insert({"B0", -TABLE_WIDTH / 2 - 0.4f, 0, 0.0f, TABLE_LENGTH / 2, 0.3f, 1.5f});
    boxes.insert({"B1", -TABLE_WIDTH / 2 - 0.4f, 0, -TABLE_LENGTH / 2, 0.0f, 0.3f, 1.5f});
    boxes.insert({"B2", 0, TABLE_WIDTH / 2 + 0.4f, 0.0f, TABLE_LENGTH / 2, 0.3f, 1.5f});
    boxes.insert({"B3", 0, TABLE_WIDTH / 2 + 0.4f, -TABLE_LENGTH / 2, 0.0f, 0.3f, 1.5f});


#ifdef VIS3D
    viz::Viz3d window3D("Viz");
    viz::WPlane table_w(Point3d(0, 0, 0), Vec3d(0, 0, 1), Vec3d(0, 1, 0),
                        Size2d(TABLE_WIDTH, TABLE_LENGTH));
    viz::WCameraPosition origin_w;
    viz::WCameraPosition camera_w(Vec2d(1.22, 1.04));
    viz::WPlane floor_w(Point3d(0, 0, -1), Vec3d(0, 0, 1), Vec3d(0, 1, 0), Size2d(5, 5),
                        viz::Color::gray());
    window3D.showWidget("table", table_w);
    window3D.showWidget("floor", floor_w);
    window3D.showWidget("origin", origin_w);
    window3D.showWidget("camera", camera_w);
    vector<viz::WSphere> positions_w;
    for (auto& b: boxes.boxes) {
        viz::WSphere s(Point3d(0, 0, 0), 0.1, 10, viz::Color(rng(255), rng(255), rng(255)));
        window3D.showWidget(b.id + "_position", s);
        positions_w.push_back(s);
        viz::WCube c(Point3d(b.minX, b.minY, b.minZ), Point3d(b.maxX, b.maxY, b.maxZ));
        window3D.showWidget(b.id + "_cube", c);
    }
#endif

    signal(SIGINT, siginthandler);

    while (running) {

        listener.waitForNewFrame(frames);

        Frame* depth = frames[Frame::Depth];
        Frame* color = frames[Frame::Color];

        Mat colorImage(color->height, color->width, CV_8UC4, color->data);
        Mat depthImage(depth->height, depth->width, CV_32FC1, depth->data);

        Mat rgb;
        cvtColor(colorImage, rgb, CV_RGBA2RGB);

#ifdef VIS2D
        imshow("Depth", depthImage / 4096.0f);
        imshow("Color", rgb);
        waitKey(1);
#endif

        if (!calibrated) {
            calibrated = findTransformation(rgb, colorCamMat, colorDistCoeffs, 0.059f, transformation);
            listener.release(frames);
            continue;
        }

        medianBlur(depthImage, depthImage, 3);

        boxes.resetAll();
        boxes.fill(depth->data, depthCamMat, transformation);
        boxes.sortAll();
        boxes.computePositions();


#ifdef VIS3D
        for (auto& b : boxes.boxes) {
            Point3f pos = Vec3f(b.top);
            if (pos == Point3f()) continue;
            Affine3d pose;
            pose.translation(Vec3f(b.top));
            window3D.setWidgetPose(b.id + "_position", pose);
            vector<Point3f> points(1, Point3f());
            viz::WCloud cloud(points);
            if (!b.points.empty()) {
                cloud = viz::WCloud(b.points);
            }
            window3D.showWidget(b.id + "_cloud", cloud);
        }
        camera_w.setPose(transformation);

        window3D.spinOnce(1, true);
#endif

        listener.release(frames);

        server.send(boxes.getTrackingData());

    }

    server.stop();

    dev->stop();
    dev->close();

    return 0;

}
