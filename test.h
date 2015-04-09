#ifndef test_H
#define test_H

#include <iostream>
#include <vector>

#include <cv.h>
#include <highgui.h>

int main2(int argc, char** argv);

struct Algorithm
{
  virtual void Update() = 0;
};

#define CreateInput(macroVarName, macroVarType) \
  public: \
  void Set##macroVarName(const macroVarType* _##macroVarName) { m_##macroVarName = _##macroVarName; } \
  private: \
  const macroVarType* m_##macroVarName;
#define CreateOutput(macroVarName, macroVarType) \
  public: \
  void Set##macroVarName(macroVarType* _##macroVarName) { m_##macroVarName = _##macroVarName; } \
  private: \
  macroVarType* m_##macroVarName;

typedef std::vector<cv::Point2f> ImagePoints;
struct ImagePointsFromRandomPlacement : public ::Algorithm
{
  CreateOutput(ImagePoints, ImagePoints);
  CreateInput(ImageSize, cv::Size);
  CreateInput(NumPoints, size_t);
  CreateInput(Range, float);
public:
  void Update()
  {
    if (m_ImagePoints->size() != *m_NumPoints)
      m_ImagePoints->clear();

    if (m_ImagePoints->empty())
    {
      for (size_t i = 0; i < *m_NumPoints; ++i)
      {
        cv::Point2f p;
        p.x = rng.uniform(0, m_ImageSize->width - 1);
        p.y = rng.uniform(0, m_ImageSize->height - 1);
        m_ImagePoints->push_back(p);
      }
    }
    else
    {
      for (size_t i = 0; i < *m_NumPoints; ++i)
      {
        cv::Point2f& p = m_ImagePoints->at(i);
        p.x = p.x + rng.uniform(-(*m_Range), (*m_Range));
        while (p.x < 0 || p.x >= m_ImageSize->width)
          p.x = p.x + rng.uniform(-(*m_Range), (*m_Range));

        p.y = p.y + rng.uniform(-(*m_Range), (*m_Range));
        while (p.y < 0 || p.y >= m_ImageSize->height)
          p.y = p.y + rng.uniform(-(*m_Range), (*m_Range));
      }
    }
  }

  ImagePointsFromRandomPlacement()
    : rng(cv::RNG(0xFFFFFFFF)) {}
private:
  cv::RNG rng;
};

struct ImageFromImagePointsDrawing : public ::Algorithm
{
  CreateInput(ImagePoints, ImagePoints);
  CreateInput(ImageSize, cv::Size);
  CreateOutput(Image, cv::Mat);
public:
  void drawCross(const cv::Point2f& center, const cv::Scalar& color, float d, cv::Mat& img)
  {
    cv::line(img, cv::Point2f(center.x - d, center.y - d), cv::Point2f(center.x + d, center.y + d), color, 2, CV_AA, 0);
    cv::line(img, cv::Point2f(center.x + d, center.y - d), cv::Point2f(center.x - d, center.y + d), color, 2, CV_AA, 0);
  }
  void Update()
  {
    *m_Image = cv::Mat::zeros(*m_ImageSize, CV_8UC3);
    if (m_ImagePoints->size() == 0)
      return;

    if (previousPoints.size() > 0 && previousPoints.back().size() != m_ImagePoints->size())
    {
      previousPoints.clear();
      colors.clear();
    }
    previousPoints.push_back(*m_ImagePoints);

    for (size_t i = 0; i < m_ImagePoints->size(); ++i)
    {
      const cv::Point2f& p = m_ImagePoints->at(i);
      cv::Scalar color;
      if (colors.size() < m_ImagePoints->size())
      {
        color = cv::Scalar(rng.uniform(128, 255), rng.uniform(128, 255), rng.uniform(128, 255));
        colors.push_back(color);
      }
      else
      {
        color = colors.at(i);
      }

      drawCross(p, color, 5, *m_Image);

      if (previousPoints.size() > 0)
      {
        for (size_t j = 0; j < previousPoints.size() - 1; j++)
        {
          cv::line(*m_Image, previousPoints.at(j).at(i), previousPoints.at(j + 1).at(i), color, 1);
        }
      }
    }
  }

  ImageFromImagePointsDrawing()
    : rng(cv::RNG(0xFFFFFFFF)) {}
private:
  cv::RNG rng;
  cv::Mat image;
  std::vector<ImagePoints> previousPoints;
  std::vector<cv::Scalar> colors;
};

struct ImagePointsFromImagePointsPredictionData;
struct ImagePointsFromImagePointsPrediction : public ::Algorithm
{
  ImagePointsFromImagePointsPrediction();
  virtual ~ImagePointsFromImagePointsPrediction();
  void Update();

  CreateInput(ImagePoints, ImagePoints);
  CreateInput(NumPoints, size_t);
  CreateInput(ForcePrediction, bool);
  CreateOutput(OutputImagePoints, ImagePoints);

  ImagePointsFromImagePointsPredictionData* d;
};

#endif