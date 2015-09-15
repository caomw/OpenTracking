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
  float thr_fern; //0.6，初始化过程中会修正，即可能进一步提高
  int structSize;//13 随机厥的深度
  int nstructs; //10 随机厥叶子的个数
  float valid; // 正样本的前 valid%
  float ncc_thesame;//相关系数阈值0.95
  float thr_nn; //0.65，初始化函数中可能提高
  int acum;
public:
  //Parameters
  float thr_nn_valid;//初始化为0.7,这个值比最近邻的分类阈值thr_nn更严格，初始化函数中可能提高

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
  struct Feature//一对坐标 //特征结构体  
      {
          uchar x1, y1, x2, y2;
          Feature() : x1(0), y1(0), x2(0), y2(0) {}
          Feature(int _x1, int _y1, int _x2, int _y2)
          : x1((uchar)_x1), y1((uchar)_y1), x2((uchar)_x2), y2((uchar)_y2)
          {}
		  //二维单通道元素可以用Mat::at(i, j)访问，i是行序号，j是列序号  
		  //返回的patch图像片在(y1,x1)和(y2, x2)点的像素比较值，返回0或者1  
          bool operator ()(const cv::Mat& patch) const
          { return patch.at<uchar>(y1,x1) > patch.at<uchar>(y2, x2); }
      };
   //Ferns（蕨类植物：有根、茎、叶之分，不具花）features 特征组？  
  std::vector<std::vector<Feature> > features; //Ferns features (one std::vector for each scale)，比较点对的索引
  std::vector< std::vector<int> > nCounter; //negative counter
  std::vector< std::vector<int> > pCounter; //positive counter
  std::vector< std::vector<float> > posteriors; //Ferns posteriors
  float thrN; //Negative threshold  50%【5.3.1】
  float thrP;  //Positive thershold ，thr_fern*nstructs=0.6*10
  //NN Members
  std::vector<cv::Mat> pEx; //NN positive examples
  std::vector<cv::Mat> nEx; //NN negative examples
};
