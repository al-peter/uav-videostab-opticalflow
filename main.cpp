// ===== FULL FILE: videostab_optflow_fixed.cpp =====
// Fix: apply the INVERSE motion (cancel camera movement) so yaw/pitch directions look correct.
// All changes are marked with:  // **** CHANGED: ...

#include <opencv2/opencv.hpp>
#include <deque>
#include <cmath>
#include <chrono>
#include <iostream>
#include <thread>

using namespace cv;
using namespace std;

// ===================== USER SETTINGS =====================
struct Config {
    double downSample      = 0.25; // LK runs on downscaled frames
    double zoomFactor      = 0.95; // <1.0 shows more black borders
    int    smoothingWindow = 5;    // EMA buffer size
    double alpha           = 0.85; // EMA blend
    bool   saveFrames      = false;
    int    frameSkip       = 1;    // 2/15fps

    bool   showRectROI     = false;
    bool   showTrackingPts = false;
    bool   maskFrame       = false;
    double roiDiv          = 1.0;  // 1.0 => full frame; 0<roiDiv<1 => centered crop
} CFG;

// ===================== LK / RANSAC =====================
struct LKParams {
    Size  win   = Size(15,15); // 11,11/12,12
    int   levels= 3;           // 2
    TermCriteria term{TermCriteria::COUNT | TermCriteria::EPS, 20, 0.01}; // 15, 0.01
    int   maxFeatures = 800; // 300
} LKP;

static inline Rect centeredRoi(const Size& s, double frac) {
    if (!(frac > 0.0 && frac < 1.0)) return Rect(0,0, s.width, s.height);
    const int cw = max(1, (int)std::round(s.width  * frac));
    const int ch = max(1, (int)std::round(s.height * frac));
    return Rect( (s.width - cw)/2, (s.height - ch)/2, cw, ch );
}

static inline bool frameLooksCorrupt(const Mat& bgr, int expectedW, int expectedH) {
    if (bgr.empty() || bgr.cols != expectedW || bgr.rows != expectedH) return true;
    Scalar meanVal, stdVal;
    meanStdDev(bgr, meanVal, stdVal);
    const double meanSum = meanVal[0]+meanVal[1]+meanVal[2];
    const double stdSum  = stdVal[0] +stdVal[1] +stdVal[2];
    return (stdSum < 1.0 && (meanSum < 3.0 || meanSum > 3.0*252.0));
}

static inline string buildGstPipeline(int w, int h, double fps, const string& host, int port) {
    // Use actual caps to avoid unexpected converters in the pipeline.
    // NV12 path + mpph264enc (RK HW) to keep latency low on RK3588.
    std::ostringstream oss;
    oss << "appsrc format=time is-live=true do-timestamp=true "
        << "caps=video/x-raw,format=BGR,width=" << w
        << ",height=" << h << ",framerate=" << (int)fps << "/1 ! "
        << "queue max-size-buffers=2 leaky=downstream ! "
        << "videoconvert ! video/x-raw,format=NV12 ! "
        << "mpph264enc bps=2000000 gop=30 ! h264parse ! "
        << "rtph264pay pt=96 config-interval=1 mtu=1200 ! "
        << "udpsink host=" << host << " port=" << port << " sync=false";
    return oss.str();

}

int main(int argc, char** argv) {
    // cv::setUseOptimized(true);
    // cv::setNumThreads(4);

    const string src = (argc > 1) ? argv[1] : ""; // e.g. /home/orangepi/Downloads/raw.mp4
    VideoCapture cap;
    const bool isCamera = src.empty();

    int    realWidth  = 1280;
    int    realHeight = 720;
    double realFPS    = 30.0;

    // ---------------- Open source ----------------
    if (isCamera) {
        if (!cap.open("/dev/video0", CAP_V4L2)) {
            cerr << "Failed to open /dev/video0, trying /dev/video1..." << endl;
            cap.open("/dev/video1", CAP_V4L2);
        }
        if (cap.isOpened()) {
            cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G'));
            cap.set(CAP_PROP_FRAME_WIDTH,  realWidth);
            cap.set(CAP_PROP_FRAME_HEIGHT, realHeight);
            cap.set(CAP_PROP_FPS,          30);
            cap.set(CAP_PROP_BUFFERSIZE,   2);

            realWidth  = (int)cap.get(CAP_PROP_FRAME_WIDTH);
            realHeight = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
            realFPS    = cap.get(CAP_PROP_FPS);
            if (realFPS <= 0) realFPS = 30.0;
        }
    } else {
        cout << "Opening video file: " << src << endl;
        cap.open(src);
        if (cap.isOpened()) {
            realWidth  = (int)cap.get(CAP_PROP_FRAME_WIDTH);
            realHeight = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
            realFPS    = cap.get(CAP_PROP_FPS);
            if (realFPS <= 0) realFPS = 30.0;
        }
    }

    if (!cap.isOpened()) {
        cerr << "Failed to open video source!" << endl;
        return -1;
    }

    // ---------------- GStreamer sink ----------------
    const double fps = (cap.get(CAP_PROP_FPS) > 0) ? cap.get(CAP_PROP_FPS) : realFPS;
    const string gstPipeline = buildGstPipeline(realWidth, realHeight, fps, "169.254.28.59", 5601);  //192.168.51.213

    VideoWriter writer(gstPipeline, CAP_GSTREAMER, fps, Size(realWidth, realHeight), true);
    if (!writer.isOpened()) {
        cerr << "Failed to open GStreamer pipeline!" << endl;
        return -1;
    }

    // ====== State & reusable buffers ======
    Mat prevFrameFull, prevGrayDS, lastAffine;
    Mat frameFull, frameDS, currGrayDS;
    Mat fWarped, fStabilized, postMask;

    deque<Point3d> motion; // (dx, dy, da)

    // Watchdogs & counters
    int frameIdx = 0, fpsCount = 0;
    auto t0 = chrono::high_resolution_clock::now();

    int badRun = 0;
    const int BAD_RUN_RESET_STATE = 10;
    auto lastGoodTs = chrono::steady_clock::now();

    int v4lTimeouts = 0;
    const int MAX_V4L_TIMEOUTS = 5;

    auto reset_state = [&](){
        prevFrameFull.release();
        prevGrayDS.release();
        lastAffine.release();
        motion.clear();
    };

    auto reopen_camera = [&](){
        cap.release();
        if (!cap.open("/dev/video0", CAP_V4L2)) cap.open("/dev/video1", CAP_V4L2);
        if (cap.isOpened()) {
            cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G'));
            cap.set(CAP_PROP_FRAME_WIDTH,  realWidth);
            cap.set(CAP_PROP_FRAME_HEIGHT, realHeight);
            cap.set(CAP_PROP_FPS,          30);
            cap.set(CAP_PROP_BUFFERSIZE,   2);
            reset_state();
            v4lTimeouts = 0;
        }
    };

    // =================== MAIN LOOP =================== //
    while (true) {
        // ---- low-latency capture ----
        if (!cap.grab()) {
            if (++v4lTimeouts >= MAX_V4L_TIMEOUTS) {
                cerr << "V4L2: too many timeouts, reopening camera..." << endl;
                reopen_camera();
            } else {
                this_thread::sleep_for(chrono::milliseconds(5));
            }
            continue;
        }
        if (!cap.retrieve(frameFull) || frameFull.empty()) {
            if (++v4lTimeouts >= MAX_V4L_TIMEOUTS) {
                cerr << "V4L2: repeated empty retrieves, reopening..." << endl;
                reopen_camera();
            }
            continue;
        }
        v4lTimeouts = 0;

        // Optional frame skip
        if (++frameIdx % CFG.frameSkip != 0) continue;

        // Quick integrity check
        if (frameLooksCorrupt(frameFull, realWidth, realHeight)) {
            if (++badRun >= BAD_RUN_RESET_STATE) { reset_state(); badRun = 0; }
            continue;
        }
        badRun = 0;

        // FPS meter
        if (++fpsCount, chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - t0).count() >= 1000) {
            auto t1 = chrono::high_resolution_clock::now();
            double ms = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
            cout << "Actual FPS: " << fpsCount * 1000.0 / ms << endl;
            fpsCount = 0; t0 = t1;
        }

        // Prepare DS + ROI + GRAY
        if (CFG.downSample != 1.0)
            resize(frameFull, frameDS, Size(), CFG.downSample, CFG.downSample, INTER_AREA);
        else
            frameDS = frameFull;

        Rect roiDS = centeredRoi(frameDS.size(), CFG.roiDiv);
        Mat workDS = frameDS(roiDS);
        cvtColor(workDS, currGrayDS, COLOR_BGR2GRAY);

        // Gap watchdog: long capture stalls → drop history
        {
            auto now = chrono::steady_clock::now();
            if (chrono::duration_cast<chrono::milliseconds>(now - lastGoodTs).count() > 500.0) {
                reset_state();
            }
            lastGoodTs = now;
        }

        // First valid frame → initialize "prev" (MUST clone to avoid aliasing)
        if (prevFrameFull.empty()) {
            prevFrameFull = frameFull.clone();
            prevGrayDS    = currGrayDS.clone();
            continue;
        }

        // ROI rectangle (optional)
        if (CFG.showRectROI && CFG.roiDiv > 0.0 && CFG.roiDiv < 1.0) {
            Rect roiFull((int)std::round(roiDS.x / CFG.downSample),
                         (int)std::round(roiDS.y / CFG.downSample),
                         (int)std::round(roiDS.width  / CFG.downSample),
                         (int)std::round(roiDS.height / CFG.downSample));
            rectangle(prevFrameFull, roiFull, Scalar(211,211,211), 1);
        }

        // =================== OPTICAL FLOW =================== //
        vector<Point2f> prevPts;
        prevPts.reserve(LKP.maxFeatures);
        goodFeaturesToTrack(prevGrayDS, prevPts, LKP.maxFeatures, 0.01, 10, Mat(), 3, false, 0.04);

        if (!prevPts.empty()) {
            vector<Point2f> currPts(prevPts.size());
            vector<uchar>   status(prevPts.size());
            vector<float>   err(prevPts.size());

            calcOpticalFlowPyrLK(prevGrayDS, currGrayDS, prevPts, currPts, status, err,
                                 LKP.win, LKP.levels, LKP.term, 0, 1e-4);

            // Map DS/ROI → FULL
            const float invDS = (CFG.downSample != 0.0) ? float(1.0 / CFG.downSample) : 1.0f;
            const Point2f roiOff((float)roiDS.x, (float)roiDS.y);

            vector<Point2f> prevGood; prevGood.reserve(currPts.size());
            vector<Point2f> currGood; currGood.reserve(currPts.size());

            for (size_t i=0; i<status.size(); ++i) if (status[i]) {
                Point2f p0 = (prevPts[i] + roiOff) * invDS;
                Point2f p1 = (currPts[i] + roiOff) * invDS;
                prevGood.push_back(p0);
                currGood.push_back(p1);
                if (CFG.showTrackingPts) circle(prevFrameFull, p0, 3, Scalar(211,211,211), -1);
            }

            Mat m;
            if (!prevGood.empty()) {
                Mat inliers;
                // Slightly relaxed reprojection threshold improves robustness on noisy UAV video
                m = estimateAffinePartial2D(prevGood, currGood, inliers,
                                            RANSAC, 4.0, 200, 0.995, 10);
                // NOTE: m maps prev -> curr (forward motion)
            }
            if (m.empty() && !lastAffine.empty()) m = lastAffine;

            if (!m.empty()) {
                double dx = m.at<double>(0,2);
                double dy = m.at<double>(1,2);
                double da = atan2(m.at<double>(1,0), m.at<double>(0,0));

                // EMA smoothing on forward motion
                if (!motion.empty()) {
                    dx = CFG.alpha * motion.back().x + (1.0 - CFG.alpha) * dx;
                    dy = CFG.alpha * motion.back().y + (1.0 - CFG.alpha) * dy;
                    da = CFG.alpha * motion.back().z + (1.0 - CFG.alpha) * da;
                }

                motion.emplace_back(dx,dy,da);
                if ((int)motion.size() > CFG.smoothingWindow) motion.pop_front();

                double sx=0, sy=0, sa=0;
                for (auto &p : motion){ sx+=p.x; sy+=p.y; sa+=p.z; }
                sx/=motion.size(); sy/=motion.size(); sa/=motion.size();

                // **** CHANGED: apply the INVERSE transform to CANCEL motion
                // forward (prev->curr): [sx, sy, sa]
                // inverse we want:      [-sx, -sy, -sa]
                const double cx = -sx;   // **** CHANGED
                const double cy = -sy;   // **** CHANGED
                const double ca = -sa;   // **** CHANGED

                Mat warpMat = (Mat_<double>(2,3) <<
                               cos(ca), -sin(ca), cx,   // **** CHANGED: ca,cx
                               sin(ca),  cos(ca), cy);  // **** CHANGED: ca,cy

                // 1) stabilize previous frame with inverse motion
                warpAffine(prevFrameFull, fWarped, warpMat,
                           Size(frameFull.cols, frameFull.rows),
                           INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0,0));

                // 2) zoom-in to hide borders
                Mat T = getRotationMatrix2D(Point2f(frameFull.cols*0.5f, frameFull.rows*0.5f),
                                            0.0, CFG.zoomFactor);
                warpAffine(fWarped, fStabilized, T,
                           Size(frameFull.cols, frameFull.rows),
                           INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0,0));

                if (CFG.maskFrame) {
                    if (postMask.empty()) {
                        postMask = Mat::zeros(fStabilized.size(), CV_8UC1);
                        rectangle(postMask, Point(100,200),
                                  Point(frameFull.cols-100, frameFull.rows-100), Scalar(255), -1);
                    }
                    bitwise_and(fStabilized, fStabilized, fStabilized, postMask);
                }

                writer.write(fStabilized);
                if (CFG.saveFrames) {
                    imwrite(string("frame_") + to_string(frameIdx) + ".jpg", fStabilized);
                }

                lastAffine = warpMat.clone();
            }
        }

        // Roll-forward (MUST clone to avoid aliasing)
        prevFrameFull = frameFull.clone();
        prevGrayDS    = currGrayDS.clone();
    }

    cap.release();
    writer.release();
    return 0;
}
