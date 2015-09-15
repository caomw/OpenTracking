/*
 * FerNNClassifier.h
 *
 *  Created on: Jun 14, 2011
 *      Author: alantrrs
 */

#include <opencv2/opencv.hpp>
#include <stdio.h>
class FerNNClassifier{
private:
  float thr_fern; //0.6����ʼ�������л������������ܽ�һ�����
  int structSize;//13 ����ʵ����
  int nstructs; //10 �����Ҷ�ӵĸ���
  float valid; // ��������ǰ valid%
  float ncc_thesame;//���ϵ����ֵ0.95
  float thr_nn; //0.65����ʼ�������п������
  int acum;
public:
  //Parameters
  float thr_nn_valid;//��ʼ��Ϊ0.7,���ֵ������ڵķ�����ֵthr_nn���ϸ񣬳�ʼ�������п������

  void read(const cv::FileNode& file);
  void prepare(const std::vector<cv::Size>& scales);
  void getFeatures(const cv::Mat& image,const int& scale_idx,std::vector<int>& fern);
  void update(const std::vector<int>& fern, int C, int N);
  float measure_forest(std::vector<int> fern);
  void trainF(const std::vector<std::pair<std::vector<int>,int> >& ferns,int resample);
  void trainNN(const std::vector<cv::Mat>& nn_examples);
  void NNConf(const cv::Mat& example,std::vector<int>& isin,float& rsconf,float& csconf);
  void evaluateTh(const std::vector<std::pair<std::vector<int>,int> >& nXT,const std::vector<cv::Mat>& nExT);
  void show();
  //Ferns Members
  int getNumStructs(){return nstructs;}
  float getFernTh(){return thr_fern;}
  float getNNTh(){return thr_nn;}
  struct Feature//һ������ //�����ṹ��  
      {
          uchar x1, y1, x2, y2;
          Feature() : x1(0), y1(0), x2(0), y2(0) {}
          Feature(int _x1, int _y1, int _x2, int _y2)
          : x1((uchar)_x1), y1((uchar)_y1), x2((uchar)_x2), y2((uchar)_y2)
          {}
		  //��ά��ͨ��Ԫ�ؿ�����Mat::at(i, j)���ʣ�i������ţ�j�������  
		  //���ص�patchͼ��Ƭ��(y1,x1)��(y2, x2)������رȽ�ֵ������0����1  
          bool operator ()(const cv::Mat& patch) const
          { return patch.at<uchar>(y1,x1) > patch.at<uchar>(y2, x2); }
      };
   //Ferns��ާ��ֲ��и�������Ҷ֮�֣����߻���features �����飿  
  std::vector<std::vector<Feature> > features; //Ferns features (one std::vector for each scale)���Ƚϵ�Ե�����
  std::vector< std::vector<int> > nCounter; //negative counter
  std::vector< std::vector<int> > pCounter; //positive counter
  std::vector< std::vector<float> > posteriors; //Ferns posteriors
  float thrN; //Negative threshold  50%��5.3.1��
  float thrP;  //Positive thershold ��thr_fern*nstructs=0.6*10
  //NN Members
  std::vector<cv::Mat> pEx; //NN positive examples
  std::vector<cv::Mat> nEx; //NN negative examples
};