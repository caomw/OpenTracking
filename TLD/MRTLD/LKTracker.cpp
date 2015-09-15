#include "LKTracker.h"
using namespace cv;
//金字塔LK光流法跟踪   
//Media Flow 中值光流跟踪 加 跟踪错误检测   
//构造函数，初始化成员变量   
LKTracker::LKTracker(){////该类变量需要3个参数，一个是类型，第二个参数为迭代的最大次数，最后一个是特定的阈值。 
  term_criteria = TermCriteria( TermCriteria::COUNT+TermCriteria::EPS, 20, 0.03);
  window_size = Size(4,4);// size of the search window at each pyramid level
  level = 5;//maxLevel
  lambda = 0.5;
}

//points1->points2，由于调用了filterPts，所以只有通过筛选的point对还保留在points1，points2
bool LKTracker::trackf2f(const Mat& img1, const Mat& img2,vector<Point2f> &points1, vector<cv::Point2f> &points2){
  //TODO!:implement c function cvCalcOpticalFlowPyrLK() or Faster tracking function
  //1. Track points,Forward-Backward tracking
	//基于Forward-Backward Error的中值流跟踪方法  
	//金字塔LK光流法跟踪   
	//forward trajectory 前向轨迹跟踪  
	//跟踪的原理基于Forward-Backward Error的中值流跟踪方法，对于points1中的每个点，使用前向跟踪，即上一帧的点A(由于点A是在lastbox行生成的，
	//所以确实是上一帧的点)在当前帧的跟踪结果为B，然后使用后向跟踪，即当前帧的点B反向跟踪得到上一帧的跟踪点C，这样就产生了前向和后向两条跟踪轨迹，
	//理想的情况应该是两条轨迹重合，即A和C是重合的，所以计算A和C的距离FB_error，得到一个FB_error[]数组。之后调用normCrossCorrelation()计算A和B的similarity，
	//这个similarity是以A和B为中心的，分别在上一帧和当前帧截取的10*10的区域调用matchTemplate()函数计算匹配度，将匹配度值赋给similarity，得到一个similarity[]数组。
  calcOpticalFlowPyrLK( img1,img2, points1, points2, status,similarity, window_size, level, term_criteria, lambda, 0);
  //backward trajectory 后向轨迹跟踪  
  calcOpticalFlowPyrLK( img2,img1, points2, pointsFB, FB_status,FB_error, window_size, level, term_criteria, lambda, 0);
  //2. Estimate tracking error,Compute the real FB-error
  //Compute the real FB-error   
  //原理很简单：从t时刻的图像的A点，跟踪到t+1时刻的图像B点；然后倒回来，从t+1时刻的图像的B点往回跟踪，  
  //假如跟踪到t时刻的图像的C点，这样就产生了前向和后向两个轨迹，比较t时刻中 A点 和 C点 的距离，如果距离  
  //小于一个阈值，那么就认为前向跟踪是正确的；这个距离就是FB_error  
  //计算 前向 与 后向 轨迹的误差   
  for( int i= 0; i<points1.size(); ++i ){
        FB_error[i] = norm(pointsFB[i]-points1[i]);//残差为欧氏距离【ICPR 2】
  }
  //3.Filter out outliers
  //Filter out points with FB_error[i] > median(FB_error) && points with sim_error[i] > median(sim_error)
  //调用normCrossCorrelation()计算A和B的similarity，这个similarity是以A和B为中心的，分别在上一帧和当前帧
  //截取的10*10的区域调用matchTemplate()函数计算匹配度，将匹配度值赋给similarity，得到一个similarity[]数组
  normCrossCorrelation(img1,img2,points1,points2);
  //接下来调用filterPts(vector& points1,vector& points2)对所得到的点进行过滤，
  //其中points1是所有点A组成的集合，points2是所有点B组成的集合。
  return filterPts(points1,points2);
}
//利用NCC把跟踪预测的结果周围取10*10的小图片与原始位置周围10*10的小图片（使用函数getRectSubPix得到）进  
//行模板匹配（调用matchTemplate）  
void LKTracker::normCrossCorrelation(const Mat& img1,const Mat& img2, vector<Point2f>& points1, vector<Point2f>& points2) {
        Mat rec0(10,10,CV_8U);
        Mat rec1(10,10,CV_8U);
        Mat res(1,1,CV_32F);
        for (int i = 0; i < points1.size(); i++) {
                if (status[i] == 1) {//跟踪到了//为1表示该特征点跟踪成功  
                        getRectSubPix( img1, Size(10,10), points1[i],rec0 );//以points1[i]为中心，提取10*10的小块
                        getRectSubPix( img2, Size(10,10), points2[i],rec1);
                        matchTemplate( rec0,rec1, res, CV_TM_CCOEFF_NORMED);//Cross Correlation //匹配前一帧和当前帧中提取的10x10象素矩形，得到匹配后的映射图像
                        similarity[i] = ((float *)(res.data))[0];//得到各个特征点的

                } else {
                        similarity[i] = 0.0;
                }
        }
        rec0.release();
        rec1.release();
        res.release();
}
//筛选出 FB_error[i] <= median(FB_error) 和 sim_error[i] > median(sim_error) 的特征点  
//得到NCC和FB error结果的中值，分别去掉中值一半的跟踪结果不好的点  
//Filter out points with FB_error[i] > median(FB_error) && points with sim_error[i] > median(sim_error)
//首先计算similarity[]数组中所有数的中值simmed，对于similarity超过simmed的点进行保留，其余的剔除，
//这样FB_error[]数组的规模也相应减小。之后计算这个减小规模后的FB_error[]数组的中值fbmed，对于FB_error小于fbmed的点进行保留，其余的剔除
bool LKTracker::filterPts(vector<Point2f>& points1,vector<Point2f>& points2){
  //Get Error Medians
  simmed = median(similarity);//NCC中值//找到相似度的中值  
  size_t i, k;
  for( i=k = 0; i<points2.size(); ++i ){//前向筛选，没跟踪到的不要
        if( !status[i])
          continue;//剩下 similarity[i]> simmed 的特征点  
        if(similarity[i]> simmed){//normalized crosscorrelation (NCC)筛选，比对前后两点周围的小块
          points1[k] = points1[i];
          points2[k] = points2[i];
          FB_error[k] = FB_error[i];
          k++;
        }
    }
  if (k==0)
    return false;
  points1.resize(k);
  points2.resize(k);
  FB_error.resize(k);

  fbmed = median(FB_error);//残差中值 //找到FB_error的中值  
  for( i=k = 0; i<points2.size(); ++i ){//后向筛选，找到了，但是偏离太多 //再对上一步剩下的特征点进一步筛选，剩下 FB_error[i] <= fbmed 的特征点  
      if( !status[i])
        continue;
      if(FB_error[i] <= fbmed){
        points1[k] = points1[i];
        points2[k] = points2[i];
        k++;
      }
  }
  points1.resize(k);
  points2.resize(k);
  if (k>0)
    return true;
  else
    return false;
}




/*
 * old OpenCV style
void LKTracker::init(Mat img0, vector<Point2f> &points){
  //Preallocate
  //pyr1 = cvCreateImage(Size(img1.width+8,img1.height/3),IPL_DEPTH_32F,1);
  //pyr2 = cvCreateImage(Size(img1.width+8,img1.height/3),IPL_DEPTH_32F,1);
  //const int NUM_PTS = points.size();
  //status = new char[NUM_PTS];
  //track_error = new float[NUM_PTS];
  //FB_error = new float[NUM_PTS];
}


void LKTracker::trackf2f(..){
  cvCalcOpticalFlowPyrLK( &img1, &img2, pyr1, pyr1, points1, points2, points1.size(), window_size, level, status, track_error, term_criteria, CV_LKFLOW_INITIAL_GUESSES);
  cvCalcOpticalFlowPyrLK( &img2, &img1, pyr2, pyr1, points2, pointsFB, points2.size(),window_size, level, 0, 0, term_criteria, CV_LKFLOW_INITIAL_GUESSES | CV_LKFLOW_PYR_A_READY | CV_LKFLOW_PYR_B_READY );
}
*/

