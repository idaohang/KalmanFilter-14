#include "test.h"

using namespace cv;
using namespace std;

int main2(int argc, char** argv)
{
  size_t numPoints = 5;
  cv::Size imgSize(1024, 768);
  ImagePoints points;
  float range = 10.0f;
  ImagePointsFromRandomPlacement generator;
  generator.SetImagePoints(&points);
  generator.SetImageSize(&imgSize);
  generator.SetNumPoints(&numPoints);
  generator.SetRange(&range);

  bool predict = false;
  ImagePoints predictedPoints;
  ImagePointsFromImagePointsPrediction predictor;
  predictor.SetImagePoints(&points);
  predictor.SetNumPoints(&numPoints);
  predictor.SetOutputImagePoints(&predictedPoints);
  predictor.SetForcePrediction(&predict);

  cv::Mat img;
  ImageFromImagePointsDrawing drawing;
  drawing.SetImage(&img);
  drawing.SetImagePoints(&predictedPoints);
  drawing.SetImageSize(&imgSize);

  char code = 0;

  while (true)
  {
    generator.Update();
    predictor.Update();
    drawing.Update();

    imshow("mouse kalman", img);
    code = (char)waitKey(1500);

    if (code == 'p')
      predict = !predict;

    if (code == 27 || code == 'q' || code == 'Q')
      break;
  }

  return EXIT_SUCCESS;
}

struct ImagePointsFromImagePointsPredictionData
{
  ImagePoints previousPoints;
  std::vector<cv::Vec2f> currentSpeed;
  std::vector<cv::Vec2f> previousSpeed;
  size_t currentNumPoints;
  bool predictedLastUpate;
};

ImagePointsFromImagePointsPrediction::ImagePointsFromImagePointsPrediction()
  : d(new ImagePointsFromImagePointsPredictionData)
{
  d->currentNumPoints = 0;
  d->predictedLastUpate = false;
}

ImagePointsFromImagePointsPrediction::~ImagePointsFromImagePointsPrediction()
{
  delete d;
}
void ImagePointsFromImagePointsPrediction::Update()
{
  if (d->currentNumPoints != *m_NumPoints)
  {
    d->currentSpeed.clear();
    d->previousSpeed.clear();
    d->previousPoints.clear();
    d->predictedLastUpate = false;
    d->currentNumPoints = *m_NumPoints;
  }
  // prediction case (probably include more heuristic, i.e false positive detection, compare movements of other points)
  if (*m_NumPoints != m_ImagePoints->size() || (m_ForcePrediction != 0 && *m_ForcePrediction == true))
  {
    // no prediction possible
    if (d->previousPoints.size() == 0 || d->currentSpeed.size() == 0 || d->previousSpeed.size() == 0)
    {
      *m_OutputImagePoints = *m_ImagePoints;
      return;
    }

    ImagePoints predictedPoints;
    // calculate speed and add it to the predicted point
    for (size_t i = 0; i < d->previousPoints.size(); ++i)
    {
      const cv::Vec2f& previousSpeed = d->previousSpeed.at(i);
      const cv::Vec2f& speed = d->currentSpeed.at(i);
      const cv::Point2f& oldPoint = d->previousPoints.at(i);

      // only predict length of acceleration and not movement
      float accelerationAmount = static_cast<float>(cv::norm(speed)) - static_cast<float>(cv::norm(previousSpeed));
      // take the direction of the current speed
      float speedAmount = static_cast<float>(cv::norm(speed));
      cv::Vec2f normedSpeed;
      if (speedAmount != 0)
      {
        normedSpeed[0] = speed[0] / speedAmount;
        normedSpeed[1] = speed[1] / speedAmount;
      }

      float predictedLength = speedAmount + accelerationAmount;
      if (predictedLength < 0)
        predictedLength = 0;
      cv::Vec2f predictedSpeed = predictedLength * normedSpeed;

      cout << "point " << i << ", acc=" << accelerationAmount << endl;

      cv::Point2f predictedPoint = oldPoint + cv::Point2f(predictedSpeed[0], predictedSpeed[1]);
      predictedPoints.push_back(predictedPoint);
    }

    d->predictedLastUpate = true;
    *m_OutputImagePoints = predictedPoints;
  }
  // normal case
  else
  {
    // falling back from predicting to normal case
    if (d->predictedLastUpate)
    {
      d->previousPoints.clear();
      d->currentSpeed.clear();
      d->predictedLastUpate = false;
    }

    *m_OutputImagePoints = *m_ImagePoints;
  }

  // if we have enough and valid output image points
  // (from prediction or measurement), do the speed and acceleration update
  if (m_OutputImagePoints->size() == *m_NumPoints)
  {
    // speed is only possible if we have previous information
    if (d->previousPoints.size() > 0)
    {
      // memorize previous speed for acceleration calculations
      if (d->currentSpeed.size() > 0)
        d->previousSpeed = d->currentSpeed;

      // recalculate current speed
      d->currentSpeed.clear();
      for (size_t i = 0; i < m_OutputImagePoints->size(); ++i)
      {
        const cv::Point2f& newPoint = m_OutputImagePoints->at(i);
        const cv::Point2f& oldPoint = d->previousPoints.at(i);

        cv::Vec2f speedVector = newPoint - oldPoint;
        d->currentSpeed.push_back(speedVector);
      }
    }
    d->previousPoints = *m_OutputImagePoints;
  }
}