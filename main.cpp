#include <iostream>
#include <vector>

#include <cv.h>
#include <highgui.h>

using namespace cv;
using namespace std;

struct mouse_info_struct { int x, y; };
struct mouse_info_struct mouse_info = { -1, -1 }, last_mouse, lastKalmanMouse;

vector<Point> mousev, kalmanv;

void on_mouse(int event, int x, int y, int flags, void* param) {
  //if (event == CV_EVENT_LBUTTONUP)
  {
    last_mouse = mouse_info;
    mouse_info.x = x;
    mouse_info.y = y;

    //		cout << "got mouse " << x <<","<< y <<endl;
  }
}

void cvKalmanNoObservation(KalmanFilter& KF) {
  KF.errorCovPost = KF.errorCovPre.clone();
  KF.statePost = KF.statePre.clone();
}

#include <iostream>
int main(int argc, char** argv)
{
  Mat img(500, 500, CV_8UC3);
  KalmanFilter KF(4, 2, 0);
  Mat_<float> state(4, 1);
  /* (x, y, Vx, Vy) */
  Mat processNoise(4, 1, CV_32F);
  Mat_<float> measurement(2, 1); measurement.setTo(Scalar(0));
  Mat_<float> predictMeasurement(2, 1); measurement.setTo(Scalar(0));
  char code = (char)-1;
  namedWindow("mouse kalman");
  setMouseCallback("mouse kalman", on_mouse, 0);
  bool predictOnly = false;

  for (;;)
  {
    if (mouse_info.x < 0 || mouse_info.y < 0) {
      imshow("mouse kalman", img);
      waitKey(30);
      continue;
    }
    KF.statePre.at<float>(0) = mouse_info.x;
    KF.statePre.at<float>(1) = mouse_info.y;
    KF.statePre.at<float>(2) = 0;
    KF.statePre.at<float>(3) = 0;
    KF.transitionMatrix = *(Mat_<float>(4, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);

    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
    setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
    setIdentity(KF.errorCovPost, Scalar::all(.1));

    mousev.clear();
    kalmanv.clear();
    //randn(KF.statePost, Scalar::all(0), Scalar::all(0.1));

    for (;;)
    {
      //            Point statePt(state(0),state(1));

      Mat prediction = KF.predict();
      Point predictPt(prediction.at<float>(0), prediction.at<float>(1));
      predictMeasurement(0) = predictPt.x;
      predictMeasurement(1) = predictPt.y;

      Point statePt, measPt;
      if (predictOnly)
      {
        //cvKalmanNoObservation(KF);

        // next measurement: take last known measurement direction and multiply
        // with the predicted acceleration
        vector<Point>::iterator it2 = mousev.end();
        std::advance(it2, -1);
        Point lastMeasurement = *it2;
        std::advance(it2, -1);
        Point previousMeasurement = *it2;
        Point2f directionMeasurementVector = lastMeasurement - previousMeasurement;
        double length = cv::norm(directionMeasurementVector);
        cout << "directionMeasurementVector.x: " << directionMeasurementVector.x << std::endl;
        cout << "directionMeasurementVector.y: " << directionMeasurementVector.y << std::endl;

        directionMeasurementVector.x = static_cast<float>(directionMeasurementVector.x) / static_cast<float>(length);
        directionMeasurementVector.y = static_cast<float>(directionMeasurementVector.y) / static_cast<float>(length);

        vector<Point>::iterator it = kalmanv.end();
        std::advance(it, -1);
        Point lastState = *it;
        std::advance(it, -1);
        Point previousState = *it;
        Point2f directionStateVector = lastState - previousState;
        length = cv::norm(directionStateVector);

        cout << "directionMeasurementVector.x: " << directionMeasurementVector.x << std::endl;
        cout << "directionMeasurementVector.y: " << directionMeasurementVector.y << std::endl;
        cout << "length: " << length << std::endl;

        Point2f newPoint;
        newPoint.x = lastMeasurement.x + static_cast<float>(length)* static_cast<float>(directionMeasurementVector.x);
        newPoint.y = lastMeasurement.y + static_cast<float>(length)* static_cast<float>(directionMeasurementVector.y);

        cout << "newPoint.x: " << newPoint.x << std::endl;
        cout << "newPoint.y: " << newPoint.y << std::endl;
        //system("pause");

        measurement(0) = newPoint.x;
        measurement(1) = newPoint.y;
        //cvKalmanNoObservation(KF);*/
      }
      else
      {
        measurement(0) = mouse_info.x;
        measurement(1) = mouse_info.y;
      }

      measPt = Point(measurement(0), measurement(1));
      mousev.push_back(measPt);
      Mat estimated = KF.correct(measurement);

      statePt = Point(estimated.at<float>(0), estimated.at<float>(1));
      kalmanv.push_back(statePt);

      // plot points
#define drawCross( center, color, d )                                 \
  line( img, Point( center.x - d, center.y - d ),                \
  Point( center.x + d, center.y + d ), color, 2, CV_AA, 0); \
  line( img, Point( center.x + d, center.y - d ),                \
  Point( center.x - d, center.y + d ), color, 2, CV_AA, 0 )

      img = Scalar::all(0);
      drawCross(statePt, Scalar(255, 255, 255), 5);
      drawCross(measPt, Scalar(0, 0, 255), 5);
      //            drawCross( predictPt, Scalar(0,255,0), 3 );
      //			line( img, statePt, measPt, Scalar(0,0,255), 3, CV_AA, 0 );
      //			line( img, statePt, predictPt, Scalar(0,255,255), 3, CV_AA, 0 );

      for (int i = 0; i < mousev.size() - 1; i++) {
        line(img, mousev[i], mousev[i + 1], Scalar(255, 255, 0), 1);
      }
      for (int i = 0; i < kalmanv.size() - 1; i++) {
        line(img, kalmanv[i], kalmanv[i + 1], Scalar(0, 255, 0), 1);
      }

      //            randn( processNoise, Scalar(0), Scalar::all(sqrt(KF.processNoiseCov.at<float>(0, 0))));
      //            state = KF.transitionMatrix*state + processNoise;

      imshow("mouse kalman", img);
      code = (char)waitKey(100);

      if (code == 'p')
        predictOnly = !predictOnly;
      else if (code > 0)
        break;
    }
    if (code == 27 || code == 'q' || code == 'Q')
      break;
  }

  return 0;

  /*
  std::cout << "hello" << std::endl;
  system("pause");
  */
}