# VideoStab — Optical-Flow (Inverse Motion) Stabilizer
Real-time, low-latency **digital** stabilization that cancels camera motion by
estimating an affine transform with OpenCV LK+RANSAC and applying the **inverse**
warp. Frames are H.264-encoded via Rockchip **mpph264enc** and streamed over UDP
using GStreamer (RK3588 / Orange Pi 5 friendly).

## Features
- Pyramidal Lucas–Kanade optical flow on a downscaled ROI
- RANSAC affine (translation + rotation) with outlier rejection
- **Inverse motion** application (cancels camera shake)
- EMA smoothing + windowed averaging
- Optional ROI, track-point visualization, border-hiding zoom
- Low-latency UDP H.264 using GStreamer (`mpph264enc`)

## Receiver
gst-launch-1.0 -v udpsrc port=5601 caps="application/x-rtp,
media=video, encoding-name=H264, payload=96, clock-rate=90000" !
rtpjitterbuffer latency=80 ! rtph264depay ! h264parse ! avdec_h264 !
videoconvert ! autovideosink sync=false
