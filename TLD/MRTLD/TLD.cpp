/*
 * TLD.cpp
 *
 *  Created on: Jun 9, 2011
 *      Author: alantrrs
 */

#include "TLD.h"
#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;


TLD::TLD()
{
}
TLD::TLD(const FileNode& file){
  read(file);
}

void TLD::read(const FileNode& file){
  ///Bounding Box Parameters
  min_win = (int)file["min_win"];
  ///Genarator Parameters
  //initial parameters for positive examples
  patch_size = (int)file["patch_size"];
  num_closest_init = (int)file["num_closest_init"];
  num_warps_init = (int)file["num_warps_init"];
  noise_init = (int)file["noise_init"];
  angle_init = (float)file["angle_init"];
  shift_init = (float)file["shift_init"];
  scale_init = (float)file["scale_init"];
  //update parameters for positive examples
  num_closest_update = (int)file["num_closest_update"];
  num_warps_update = (int)file["num_warps_update"];
  noise_update = (int)file["noise_update"];
  angle_update = (float)file["angle_update"];
  shift_update = (float)file["shift_update"];
  scale_update = (float)file["scale_update"];
  //parameters for negative examples
  bad_overlap = (float)file["overlap"];
  bad_patches = (int)file["num_patches"];
  classifier.read(file);
}
 //此函数根据传入的box（目标边界框）在传入的图像frame1中构建全部的扫描窗口，并计算重叠度
void TLD::init(const Mat& frame1,const Rect& box,FILE* bb_file){
  //bb_file = fopen("bounding_boxes.txt","w");
  //Get Bounding Boxes
  // 1.预先计算好所有位置，所有尺度的bb,并计算每一个bb与初始跟踪区域的面积交/并
  // 问题：好多呀，有必要吗？一劳永逸的事情，倒也耽误不了多少工夫
  buildGrid(frame1,box);
  printf("Created %d bounding boxes\n",(int)grid.size());
  //Preparation
  //allocation
  //积分图像，用以计算2bitBP特征（类似于haar特征的计算）   
  //Mat的创建，方式有两种：1.调用create（行，列，类型）2.Mat（行，列，类型（值））。
  //为各种变量或容器分配存储空间，包括积分图iisum、平方积分图iisqsum、存放good_boxes和存放bad_boxes的容器，等等
  iisum.create(frame1.rows+1,frame1.cols+1,CV_32F);
  iisqsum.create(frame1.rows+1,frame1.cols+1,CV_64F);
  //Detector data中定义：std::vector<float> dconf;  检测确信度？？  
  //vector 的reserve增加了vector的capacity，但是它的size没有改变！而resize改变了vector  
  //的capacity同时也增加了它的size！reserve是容器预留空间，但在空间内不真正创建元素对象，  
  //所以在没有添加新的对象之前，不能引用容器内的元素。   
  //不管是调用resize还是reserve，二者对容器原有的元素都没有影响。  
  //myVec.reserve( 100 );     // 新元素还没有构造, 此时不能用[]访问元素  
  //myVec.resize( 100 );      // 用元素的默认构造函数构造了100个新的元素，可以直接操作新元素
  dconf.reserve(100);
  dbb.reserve(100);
  bbox_step =7;
  //tmp.conf.reserve(grid.size());
  //以下在Detector data中定义的容器都给其分配grid.size()大小（这个是一幅图像中全部的扫描窗口个数）的容量  
  //Detector data中定义TempStruct tmp;    
  //tmp.conf.reserve(grid.size());   
  tmp.conf = vector<float>(grid.size());
  tmp.patt = vector<vector<int> >(grid.size(),vector<int>(10,0));
  //tmp.patt.reserve(grid.size());
  dt.bb.reserve(grid.size());//预留空间
  good_boxes.reserve(grid.size());
  bad_boxes.reserve(grid.size());
  //TLD中定义：cv::Mat pEx;  //positive NN example 大小为15*15图像片  
  pEx.create(patch_size,patch_size,CV_64F);
  //Init Generator
  generator = PatchGenerator(0,0,noise_init,true,1-scale_init,1+scale_init,-angle_init*CV_PI/180,angle_init*CV_PI/180,-angle_init*CV_PI/180,angle_init*CV_PI/180);
  // 2. 依据1中计算的overlap,选出10(num_closest_init)个good box，1个best_box,其余的都作为bad box（N个）
  //【5.6.1】select 10 bounding boxes on the  scanning grid that are closest to the initial bounding box.
  //此函数根据传入的box（目标边界框），在整帧图像中的全部窗口中寻找与该box距离最小（即最相似，  
  //重叠度最大）的num_closest_init个窗口，然后把这些窗口 归入good_boxes容器  
  //同时，把重叠度小于0.2的，归入 bad_boxes 容器  
  //首先根据overlap的比例信息选出重复区域比例大于60%并且前num_closet_init= 10个的最接近box的RectBox，  
  //相当于对RectBox进行筛选。并通过BBhull函数得到这些RectBox的最大边界。 
  //这个函数输入是初始跟踪窗口box，以及good_boxes中要放入的扫描窗口的数量closest_init
  getOverlappingBoxes(box,num_closest_init);
  printf("Found %d good boxes, %d bad boxes\n",(int)good_boxes.size(),(int)bad_boxes.size());
  printf("Best Box: %d %d %d %d\n",best_box.x,best_box.y,best_box.width,best_box.height);
  printf("Bounding box hull: %d %d %d %d\n",bbhull.x,bbhull.y,bbhull.width,bbhull.height);
  //Correct Bounding Box
  lastbox=best_box;//注意是 best_box
  lastconf=1;
  lastvalid=true;
  //Print
  fprintf(bb_file,"%d,%d,%d,%d,%f\n",lastbox.x,lastbox.y,lastbox.br().x,lastbox.br().y,lastconf);
  //Prepare Classifier
  //3. 确定随机森林分类器特征的计算方式， 随机产生了 130对点，比较其相互的大小关系（二值化）->作为10个列向量（向量化）
  //Prepare Classifier 准备分类器   
  //scales容器里是所有扫描窗口的尺度，由buildGrid()函数初始化 
  //这个函数输入是buildGrid函数中得到的scales容器，主要功能是进行分类器的准备
  classifier.prepare(scales);
  ///Generate Data
  // Generate positive data【5.6.1】
  // generate 20 warped versions by geometric transformations(shift 1%,scale change 1%,in-plane rotation 10)
  // and add them with Gaussian noise( ¼ 5) on pixels.The result is 200 synthetic positive patches.
  // 4. 得到最近邻分类器的正样本pEx，提取的是best_box的patch;随机森林分类器的正样本 由 good box 变形繁衍（1->20）得到
  //这个函数的输入是当前帧frame、要进行仿射变换的次数num_warps_init(初始值20是从myParam.yml中获取)，
  //主要功能是将good_boxes中的10个扫描窗口进行num_warps_init次仿射变换，这样共得到10*20个窗口，做为正样本数据
  generatePositiveData(frame1,num_warps_init);
  // 5. Set variance threshold
  Scalar stdev, mean;
  //统计best_box的均值和标准差，取var为“标准差平方的一半”做为方差分类器的阈值
  meanStdDev(frame1(best_box),mean,stdev);//注意是best_box，而不是我们自己框定的box
  //利用积分图像去计算每个待检测窗口的方差   
  //cvIntegral( const CvArr* image, CvArr* sum, CvArr* sqsum=NULL, CvArr* tilted_sum=NULL );  
  //计算积分图像，输入图像，sum积分图像, W+1×H+1，sqsum对象素值平方的积分图像，tilted_sum旋转45度的积分图像  
  //利用积分图像，可以计算在某象素的上－右方的或者旋转的矩形区域中进行求和、求均值以及标准方差的计算，  
  //并且保证运算的复杂度为O(1)。 
  //计算积分图和平方积分图，并调用getVar(best_box, iisum, iisqsum)利用积分图和平方积分图进行快速计算以得到best_box中的方差，取方差的一半做为检测方差vr
  integral(frame1,iisum,iisqsum);
  //级联分类器模块一：方差检测模块，利用积分图计算每个待检测窗口的方差，方差大于var阈值（目标patch方差的50%）的，  
  //则认为其含有前景目标方差；var 为标准差的平方   
  var = pow(stdev.val[0],2)*0.5; //【5.3.1】50 percent of variance of the patch that was selected  for tracking
  cout << "variance: " << var << endl;
  //check variance
   //getVar函数通过积分图像计算输入的best_box的方差  
  double vr =  getVar(best_box,iisum,iisqsum)*0.5;
  cout << "check variance: " << vr << endl;
  // 6. Generate negative data，得到随机森林的负样本集:nX（特征fern，N多），最近邻的负样本集：nEx（图像块patch,100个）
  //由于在跟踪时需要注意到跟踪的目标是否发生远离或靠近镜头，以及旋转和位移的变化所以加入了仿射变换，但是对于负样本而言，它本身就是负样本，进行仿射变换没有什么意义，所以无需进行
  generateNegativeData(frame1);
  //generatePositiveData和generateNegativeData的区别在于：前者进行仿射变换，后者没有；
  //前者无需打乱good_boxes，而后者打乱了bad_boxes，目的是为了随机获取前bad_patches个负样本作为后面近邻数据集的负样本训练集（关于近邻数据集的正样本训练集只有一个数据就是best_box）
  // 7. 构造训练集和测试集（负样本五五分）
  //		 |	           训练集			      |	      测试集（都只有负样本）
  // NN分类器 |  [pEx(1个) nEx(N/2)]-->nn_data     |      nExT （N/2） 
  // 随机森林 |  [nX(N/2) + pX(200)]--> ferns_data |      nXT(N/2)
  //Split Negative Ferns into Training and Testing sets (they are already shuffled)
  //将负样本放进 训练和测试集   
  //将负样本集nX一分为二成为训练集nX和测试集nXT，将负样本近邻数据集nEx也一分为二成为近邻数据训练集nEx和近邻数据测试集nExT，而正样本集pX和正样本近邻数据集pEx无需划分
  int half = (int)nX.size()*0.5f;//保留后一半作为测试
  nXT.assign(nX.begin()+half,nX.end());
  nX.resize(half);
  //Split Negative NN Examples into Training and Testing sets
  //将正样本集pX(good_boxes的区域仿射变换后的200个结果)和负样本集nX(bad_boxes中方差较大的区域选取一半出来)顺序打乱放到ferns_data[]中，
  //用于训练Fern分类器，注意这里对于每个box所存的数据是10棵树分别对这个box的特征：也即10个长度为13的二进制编码
  half = (int)nEx.size()*0.5f;
  nExT.assign(nEx.begin()+half,nEx.end());
  nEx.resize(half);
  //Merge Negative Data with Positive Data and shuffle it
  // [nX + pX]-->ferns_data
  vector<pair<vector<int>,int> > ferns_data(nX.size()+pX.size());
  vector<int> idx = index_shuffle(0,ferns_data.size());//再一次打乱+-样本的顺序
  int a=0;
  for (int i=0;i<pX.size();i++){//pX 是在 generatePositiveData中产生 10*20
      ferns_data[idx[a]] = pX[i];
      a++;
  }
  for (int i=0;i<nX.size();i++){
      ferns_data[idx[a]] = nX[i];
      a++;
  }
  //Data already have been shuffled, just putting it in the same vector
  //[pEx(1个) nEx(N多)]->nn_data
  //将正样本近邻数据集pEx(其实只有一个数据，就是best_box所得到的pattern)和负样本近邻数据集nEx(bad_boxes打乱顺序后前bad_patches/2=50个数据所得到的pattern)放到nn_data[]中，用于训练最近邻分类器
  vector<cv::Mat> nn_data(nEx.size()+1);
  nn_data[0] = pEx;
  for (int i=0;i<nEx.size();i++){
      nn_data[i+1]= nEx[i];
  }
  // 8.Training，决策森林 和 最近邻
  //训练 集合分类器（森林） 和 最近邻分类器  
  //输入是正负样本集pX和nX所存储的ferns_data[]，第二个参数2表示bootstrap=2(但在函数中并没有看出其作用)
  classifier.trainF(ferns_data,2); //bootstrap = 2
  //这个函数的输入是正样本近邻数据集pEx和负样本近邻数据集nEx所存储的nn_data[]。
  classifier.trainNN(nn_data);
  //负样本集的另一个nXT和负样本近邻数据集的另一半nExT组合起来，作为测试集，用于评价并修改得到最好的分类器阈值
  // 9.Threshold Evaluation on testing sets，检查是否要提高阈值thr_fern，thr_nn，thr_nn_valid
  //用样本在上面得到的 集合分类器（森林） 和 最近邻分类器 中分类，评价得到最好的阈值
  //这个函数的输入是负样本数据集的后半部分nXT(存放feature)和负样本近邻数据集(存放pattern)的后半部分nExT，
  //主要功能是将这两个作为测试集用来校正阈值thr_fern、thr_nn、thr_nn_valid
  classifier.evaluateTh(nXT,nExT);//仅此一次调用，后面都不会提高了
}

/* Generate Positive data
 * Inputs:
 * - good_boxes (bbP)
 * - best_box (bbP0)
 * - frame (im0)
 * - num_warps：		
 * Outputs:
 * - Positive fern features (pX)//先clear,然后由good_boxes变形而来
 * - Positive NN examples (pEx)//由best_box产生
 */
void TLD::generatePositiveData(const Mat& frame, int num_warps){
  Scalar mean;
  Scalar stdev;
  //此函数将frame图像best_box区域的图像片归一化为均值为0的15*15大小的patch，存在pEx正样本
  //调用getPattern(frame(best_box),pEx,mean,stdev)，对于frame的best_box区域，
  //将其缩放为patch_size*patch_size(15*15)大小然后计算均值与方差，将每个元素的像素值减去均值使得得到的patern的均值为0
  getPattern(frame(best_box),pEx,mean,stdev);//pEx
  //Get Fern features on warped patches
  Mat img;
  Mat warped;
  // 调用GaussianBlur(frame,img,Size(9,9),1.5)，对整个frame进行高斯平滑以得到img，高斯模板大小为9*9，X方向的方差为1.5，同时利用第3步得到的bbhull获取img的该区域
  GaussianBlur(frame,img,Size(9,9),1.5);
  warped = img(bbhull);//注意是浅拷贝，wraped并没有单独的空间
  RNG& rng = theRNG();
  Point2f pt(bbhull.x+(bbhull.width-1)*0.5f,bbhull.y+(bbhull.height-1)*0.5f);//水平，垂直中心，即旋转的中心
  //nstructs树木（由一个特征组构建，每组特征代表图像块的不同视图表示）的个数  
  //fern[nstructs] nstructs棵树的森林的数组？？   
  vector<int> fern(classifier.getNumStructs());
  pX.clear();//
  Mat patch;
  //pX为处理后的RectBox最大边界处理后的像素信息，pEx最近邻的RectBox的Pattern，bbP0为最近邻的RectBox。  
  if (pX.capacity()<num_warps*good_boxes.size())
    pX.reserve(num_warps*good_boxes.size());//pX正样本个数为 仿射变换个数 * good_box的个数，故需分配至少这么大的空间  
  int idx;
  for (int i=0;i<num_warps;i++){//每一个good_boxes都生num_warps个pX
     if (i>0)//PatchGenerator类用来对图像区域进行仿射变换，先RNG一个随机因子，再调用（）运算符产生一个变换后的正样本。  
       //其中frame是当前的帧，pt是bbhull窗口的中心坐标，warped是进行仿射变换后的结果，bbhull.size()是bbhull的宽高度，
	   //rng是一个随机数。这个函数是根据bbhull.size()的尺寸和rng生成一个仿射矩阵，对frame进行仿射变换得到结果放到warped中，也即变换前尺寸是原始frame的尺寸，变换后就是bbhull的尺寸了
		 generator(frame,pt,warped,bbhull.size(),rng);//仿射变换，问题：这一块是不是错啦？没错！！！
     for (int b=0;b<good_boxes.size();b++){
         idx=good_boxes[b]; //good_boxes容器保存的是 grid 的索引  
		 patch = img(grid[idx]);//把img的 grid[idx] 区域（也就是bounding box重叠度高的）这一块图像片提取出来  
         classifier.getFeatures(patch,grid[idx].sidx,fern); //getFeatures函数得到输入的patch的用于树的节点，也就是特征组的特征fern（13位的二进制代码）  
         pX.push_back(make_pair(fern,1));
     }
  }
  printf("Positive examples generated: ferns:%d NN:1\n",(int)pX.size());
}
//Output: 尺寸归一化，零均值 patch，要不要再加个方差归一化呢？
//先对最接近box的RectBox区域得到其patch ,然后将像素信息转换为Pattern，  
//具体的说就是归一化RectBox对应的patch的size（放缩至patch_size = 15*15），将2维的矩阵变成一维的向量信息，  
//然后将向量信息均值设为0，调整为zero mean and unit variance（ZMUV）  
//Output: resized Zero-Mean patch  
void TLD::getPattern(const Mat& img, Mat& pattern,Scalar& mean,Scalar& stdev){
 //将img放缩至patch_size = 15*15，存到pattern中  
  resize(img,pattern,Size(patch_size,patch_size));
   //计算pattern这个矩阵的均值和标准差   
  meanStdDev(pattern,mean,stdev);
  pattern.convertTo(pattern,CV_32F);
  pattern = pattern-mean.val[0];
}

/* Fun
*	1.对于所有的bad_boxes，筛选出var>var*0.5f的样本（方差这一关奈何不得它），作为重点堤防对象
*   2.提取重点堤防对象的特征存到 nX,特征是nstructs个structSize位的二进制
*   3.随机保留bad_patches个bad_boxes(筛选前)
* Inputs:
* - Image
* - bad_boxes (Boxes far from the bounding box)
* - variance (pEx variance)
* Outputs
* - Negative fern features (nX)
* - Negative NN examples (nEx)
*/
void TLD::generateNegativeData(const Mat& frame){
	//由于之前重叠度小于0.2的，都归入 bad_boxes了，所以数量挺多，下面的函数用于打乱顺序，也就是为了  
	//后面随机选择bad_boxes   
	//由于在3步中重叠率小于0.2的都放到bad_boxes中，所以其规模很大，这里就先把bad_boxes中的元素顺序打乱，
	//之后把方差大于var*0.5的bad_boxes放到pEx中作为负样本，而把方差较小的进行剔除。
	//和第5步一样，也调用getFeature()获取负样本数据并放到nX中。
	//另外，它还对打乱顺序的bad_boxes取了前bad_patches(固定值为100，从myParam.yml中读取)个，
	//通过getPattern()将获取的pattern放到nEx中，作为近邻数据集的训练集用于后面对近邻分类器的训练
  random_shuffle(bad_boxes.begin(),bad_boxes.end());//Random shuffle bad_boxes indexes
  int idx;
  //Get Fern Features of the boxes with big variance (calculated using integral images)
  int a=0;
  //int num = std::min((int)bad_boxes.size(),(int)bad_patches*100); //limits the size of bad_boxes to try
  printf("negative data generation started.\n");
  vector<int> fern(classifier.getNumStructs());// nstructs(10)
  nX.reserve(bad_boxes.size());
  Mat patch;
  //将bad_boxes中能顺利通过第一关：方差的样本的特征（fern）加入随机森林的负样本集nX
  for (int j=0;j<bad_boxes.size();j++){//把方差较大的bad_boxes加入负样本 
      idx = bad_boxes[j];
          if (getVar(grid[idx],iisum,iisqsum)<var*0.5f)
            continue;
      patch =  frame(grid[idx]);
	  classifier.getFeatures(patch,grid[idx].sidx,fern);
      nX.push_back(make_pair(fern,0));
      a++;
  }
  printf("Negative examples generated: ferns: %d ",a);
  //random_shuffle(bad_boxes.begin(),bad_boxes.begin()+bad_patches);//Randomly selects 'bad_patches' and get the patterns for NN;
  //将所有bad_boxes尺寸归一化，零均值化后，随机选择bad_patches(100)个作为最近邻分类器的负样本集nEx
  Scalar dum1, dum2;
  nEx=vector<Mat>(bad_patches);// bad_patches 100
  for (int i=0;i<bad_patches;i++){
      idx=bad_boxes[i];
	  patch = frame(grid[idx]); //具体的说就是归一化RectBox对应的patch的size（放缩至patch_size = 15*15）  
	  //由于负样本不需要均值和方差，所以就定义dum，将其舍弃  
      getPattern(patch,nEx[i],dum1,dum2);//尺寸归一化，零均值化
  }
  printf("NN: %d\n",(int)nEx.size());
}
//该函数通过积分图像计算输入的box的方差   
double TLD::getVar(const BoundingBox& box,const Mat& sum,const Mat& sqsum){
  double brs = sum.at<int>(box.y+box.height,box.x+box.width);
  double bls = sum.at<int>(box.y+box.height,box.x);
  double trs = sum.at<int>(box.y,box.x+box.width);
  double tls = sum.at<int>(box.y,box.x);
  double brsq = sqsum.at<double>(box.y+box.height,box.x+box.width);
  double blsq = sqsum.at<double>(box.y+box.height,box.x);
  double trsq = sqsum.at<double>(box.y,box.x+box.width);
  double tlsq = sqsum.at<double>(box.y,box.x);
  double mean = (brs+tls-trs-bls)/((double)box.area());
  double sqmean = (brsq+tlsq-trsq-blsq)/((double)box.area());
  return sqmean-mean*mean; //方差=E(X^2)-(EX)^2   EX表示均值   
}
/*
* In:
*	-img1:	img1last_gray
*	-img2:	current_gray
*/
void TLD::processFrame(const cv::Mat& img1,const cv::Mat& img2,vector<Point2f>& points1,vector<Point2f>& points2,BoundingBox& bbnext,bool& lastboxfound, bool tl, FILE* bb_file){
  vector<BoundingBox> cbb;//聚类之后的 bounding box
  vector<float> cconf;
  int confident_detections=0;//小D的结果聚类之后，分数比小T高的数目
  int didx; //detection index
  /// 1.Track///Track  跟踪模块   
  if(lastboxfound && tl){//前一帧目标出现过，我们才跟踪，否则只能检测了
	  //这个函数的输入是上一帧灰度图img1和当前帧灰度图img2，以及跟踪点的坐标points1和points2
      track(img1,img2,points1,points2);
  }
  else{
      tracked = false;
  }
  /// 2.Detect 检测模块 
  //这个函数的输入是当前帧的灰度图，对该图中的所有扫描窗口，依次用上面提到的方差分类器、Fern分类器、
  //最近邻分类器所形成的级联分类器进行检测，只有通过所有分类器检测的扫描窗口才认为含有前景目标而可能是跟踪区域
  detect(img2);
  /// 3.Integration
  ///Integration   综合模块   
  //TLD只跟踪单目标，所以综合模块综合跟踪器跟踪到的单个目标和检测器检测到的多个目标，然后只输出保守相似度最大的一个目标  
  //这个模块是对跟踪和检测结果进行综合，按照是否检测到、是否跟踪到的两两组合可以分为四种情况：
  //跟踪到检测到、跟踪到但没检测到、没跟踪到但检测到、没跟踪到没检测到。
  //由于需要用到检测模块中的保守相似度，也即必须使用通过检测模块的扫描窗口的信息，
  //对于综合模块，其实就是用跟踪模块和检测模块的结果进行综合考虑，前提是检测器有检测到结果dbb，
  //对dbb进行聚类得到cbb，然后根据跟踪器的结果tbb的情况来分析：如果tbb存在，看tbb与cbb的重叠率，
  //如果重叠率低但是cbb可信度高，用cbb修正tbb；如果重叠率接近，将dbb和cbb加权平均得到当前帧跟踪结果bbnext，
  //但是跟踪器dbb权重较大；如果tbb不存在，看聚类结果tbb是否只有一个中心，是则以其为当前帧跟踪结果bbnext，
  //如果不只一个中心则将聚类结果丢弃，当前帧没有跟踪到。
  if (tracked){
      bbnext=tbb;//   小T， bbnext这是你下一次要跟踪的目标
      lastconf=tconf;//表示相关相似度的阈值   
      lastvalid=tvalid;//表示保守相似度的阈值  
      printf("Tracked\n");
	  //--------- 4. Detetor Vs Trackr-------------
      if(detected)//如果跟踪到并检测到
	  { //   if Detected//通过 重叠度 对检测器检测到的目标bounding box进行聚类，每个类其重叠度小于0.5  
          //对检测到的窗口(通过检测模块的三个分类器后得到的窗口)，调用clusterConf(dbb,dconf,cbb,cconf)进行聚类，其中dbb和dconf是检测模块得到的结果(dbb->detect bounding box)，cbb(cbb->cluster bounding box)和cconf是该函数的输出
		  clusterConf(dbb,dconf,cbb,cconf);//检测的结果太多，所以要进行非极大值抑制，这里是用cluster的方法，时间消耗应该不少吧？？
          printf("Found %d clusters\n",(int)cbb.size());
		  //Get index of a clusters that is far from tracker and are more confident than the tracker
          //接下来对cbb中每个聚类中心进行判断，如果有聚类中心满足如下条件一：“它与tbb(也即跟踪模块中光流跟踪得到的当前帧的跟踪区域)的重叠率小于0.5但是保守相似度却大于tbb的保守相似度，
		  //也即找到一个聚类中心离跟踪器的跟踪结果较远(重叠率小于0.5)但是比跟踪器更为可信(保守相似度高于tbb的)”，那么记录满足条件一的聚类中心的数目，
		  //如果这个数目是1，也即只有一类满足，那么重新初始化跟踪器，并置lastvalid=false表示上一帧在跟踪器中是无效的（也即用检测器的检测结果对跟踪器的跟踪结果进行修正）。
		  //如果没有找到满足条件一的聚类中心，则找出检测到检测结果dbb中所有窗口与跟踪结果tbb重叠率超过0.7的，将其与tbb的信息一起累加再求平均，
		  //但是跟踪器tbb的权重较大，虽然tbb只有一个窗口，但是在计算平均值的时候是用10个tbb加上其它dbb中的信息再除以总个数，所以跟踪器tbb权重大(一抵十)。
		  for (int i=0;i<cbb.size();i++){//找到与跟踪器跟踪到的box距离比较远的类（检测器检测到的box），而且它的相关相似度比跟踪器的要大  
              if (bbOverlap(tbb,cbb[i])<0.5 && cconf[i]>tconf){//小D小T分歧很大，而且小D更有把握（都是用Conservative similarity对比）
                  confident_detections++;//看看小D是不是眼花了//记录满足上述条件，也就是可信度比较高的目标box的个数  
                  didx=i; //detection index
              }
          }
		  /*----------------------------小T向小D学习------------------------------*/
          if (confident_detections==1){ //小D没有眼花，而且看得比小T更清楚 //如果只有一个满足上述条件的box，那么就用这个目标box来重新初始化跟踪器（也就是用检测器的结果去纠正跟踪器） 
			  //L: 这次小D的表现不错(confident_detections==1)，小T你下次跟小D走（重新初始化）
			  //T: o(╯□╰)o，技不如人，只能任人摆布了(bbnext=cbb[didx])			 
			  printf("Found a better match..reinitializing tracking\n"); //if there is ONE such a cluster, re-initialize the tracker
              bbnext=cbb[didx];//重新初始化要跟踪的目标 
              lastconf=cconf[didx];//
              lastvalid=false; // 小T，你这一帧表现不好，所以这次小D就不跟你学习了
			  // 问题：confident_detections>1的情况呢？不算小D赢吗？
			  // 原来：TLD是单目标跟踪，如果>1只能说是小D眼花了……
          }
          else { //找到检测器检测到的box与跟踪器预测到的box距离很近（重叠度大于0.7）的box，对其坐标和大小进行累加  
			  /*-----------------综合小T和小D的检测结果------------------------------*/
			  //L:就算小T赢了，小D结果不如你（confident_detections），小D眼花了（confident_detections>1），但是！下一帧的位置也不能完全听你的，
			  //  我们还要用dbb和你比一下，除非小D的表现实在不像话，明明错了，还自以为是（close_detections==0）
			  //T:为嘛还要让我迁就Detetor的意见(对 bbOverlap(tbb,dbb[i])>0.7 的dbb[i]的位置求平均)
			  //L:小T你还真别别委屈，一般来说 小D都比你靠谱，你没看到目标检测算法能独挑大梁吗？ex:face detection
			  //L:好吧，考虑到这次你的表现好，而且你每次都只能生一个目标区域，就让你的权重为10，小D的权重为1，谁让它生的多，公平吧
              printf("%d confident cluster was found\n",confident_detections);
              int cx=0,cy=0,cw=0,ch=0;
              int close_detections=0;
              for (int i=0;i<dbb.size();i++){
                  if(bbOverlap(tbb,dbb[i])>0.7){  // Get mean of close detections
                      cx += dbb[i].x;
                      cy +=dbb[i].y;
                      cw += dbb[i].width;
                      ch += dbb[i].height;
                      close_detections++;
                      printf("weighted detection: %d %d %d %d\n",dbb[i].x,dbb[i].y,dbb[i].width,dbb[i].height);
                  }
              }
			  if (close_detections>0){ //对与跟踪器预测到的box距离很近的box 和 跟踪器本身预测到的box 进行坐标与大小的平均作为最终的  
				  //目标bounding box，但是跟踪器的权值较大
                  bbnext.x = cvRound((float)(10*tbb.x+cx)/(float)(10+close_detections));// weighted average trackers trajectory with the close detections
                  bbnext.y = cvRound((float)(10*tbb.y+cy)/(float)(10+close_detections));
                  bbnext.width = cvRound((float)(10*tbb.width+cw)/(float)(10+close_detections));
                  bbnext.height =  cvRound((float)(10*tbb.height+ch)/(float)(10+close_detections));
                  printf("Tracker bb: %d %d %d %d\n",tbb.x,tbb.y,tbb.width,tbb.height);
                  printf("Average bb: %d %d %d %d\n",bbnext.x,bbnext.y,bbnext.width,bbnext.height);
                  printf("Weighting %d close detection(s) with tracker..\n",close_detections);
              }
              else{
                printf("%d close detections were found\n",close_detections);

              }
          }
      }
  }
  else{                                       //   If NOT tracking
      printf("Not tracking..\n");
	  //此时没有tbb只有dbb，置lastboxfound和lastvalid为false表示上一帧的box无效
      lastboxfound = false;
      lastvalid = false;
	  //如果跟踪器没有跟踪到目标，但是检测器检测到了一些可能的目标box，那么同样对其进行聚类，但只是简单的  
	  //将聚类的cbb[0]作为新的跟踪目标box（不比较相似度了？？还是里面已经排好序了？？），重新初始化跟踪器  
      if(detected){ //如果没跟踪到但检测到   //  and detector is defined   
		  //同样对dbb进行聚类得到cbb，如果聚类中心只有一个，将聚类中心的信息作为当前帧的处理结果bbnext，
		  //如果聚类中心有多个，丢弃检测器的检测结果。
          clusterConf(dbb,dconf,cbb,cconf);   //  cluster detections
          printf("Found %d clusters\n",(int)cbb.size());
          if (cconf.size()==1){// 大于1呢？，眼花了呗
              bbnext=cbb[0];//注意使用cbb来初始化
              lastconf=cconf[0];
              printf("Confident detection..reinitializing tracker\n");
              lastboxfound = true;
          }
      }
  }
  lastbox=bbnext; //
  if (lastboxfound)
    fprintf(bb_file,"%d,%d,%d,%d,%f\n",lastbox.x,lastbox.y,lastbox.br().x,lastbox.br().y,lastconf);
  else
    fprintf(bb_file,"NaN,NaN,NaN,NaN,NaN\n");
  if (lastvalid && tl)//tvalid
	  //------------- 5. learn ---------------
	  ///learn 学习模块 
	  //这个函数的输入是当前帧的灰度图，主要功能是进行学习
    learn(img2);
}

/*
* Inputs:
*	-current frame(img2), last frame(img1), last Bbox(bbox_f[0]).
* Outputs:
*	-Confidence(tconf), Predicted bounding box(tbb),Validity(tvalid), points2 (for display purposes only)
*/
void TLD::track(const Mat& img1, const Mat& img2,vector<Point2f>& points1,vector<Point2f>& points2){
  //【5.4】
  // 1.Generate points
	  //网格均匀撒点（均匀采样），在lastbox中共产生最多10*10=100个特征点，存于points1  
	//这个函数主要功能就是进行点的生成，在上一帧的跟踪区域lastbox中进行均匀采样，
	//得到不超过10*10=100个点放到points1中(因为采样步长是用ceil进一法得到，所以每行或每列得到的点可能无法达到10个)。
  bbPoints(points1,lastbox);
  if (points1.size()<1){//问题：何时会出现这种情况？？
      printf("BB= %d %d %d %d, Points not generated\n",lastbox.x,lastbox.y,lastbox.width,lastbox.height);
      tvalid=false;
      tracked=false;
      return;
  }
  vector<Point2f> points = points1;
  //Frame-to-frame tracking with forward-backward error cheking
  // 2. 推断上一帧的points，在当前帧的位置，points->points2
  // 注意:只有通过筛选的point对 还保留在points，points2
  //trackf2f函数完成：跟踪、计算FB error和匹配相似度sim，然后筛选出 FB_error[i] <= median(FB_error) 和   
  //sim_error[i] > median(sim_error) 的特征点（跟踪结果不好的特征点），剩下的是不到50%的特征
  //这个函数输入是上一帧灰度图img1和当前帧灰度图img2，bbPoints()生成的点序列points1，输出是points2，
  //主要功能是完成：跟踪、计算FB error和匹配相似度sim，然后剔除 匹配度小于匹配度中值的(sim_error[i] > median(sim_error))，再剔除跟踪误差大于误差中值的(FB_error[i] <= median(FB_error))， 也即把跟踪结果不好的特征点去掉，剩下的是不到50%的特征点，对应地留在points1和points2中。
  tracked = tracker.trackf2f(img1,img2,points,points2);
  if (tracked){//只要有一个点跟到了，就算跟到了……，是不是应该严格一点呢？？
      // 3. Bounding box prediction
	  //利用剩下的这不到一半的跟踪点输入来预测bounding box在当前帧的位置和大小 tbb  
	  //这个函数输入中的point1和points2是前面用光流法跟踪并剔除跟踪效果不好的特征点而剩下的点集，lastbox是上一帧的跟踪结果，
	  //tbb是用于记录当前帧的跟踪结果。主要功能是利用剩下的这不到一半的跟踪点作为输入来预测bounding box在当前帧的位置和大小 并放到tbb中。 
      bbPredict(points,points2,lastbox,tbb);//此时，lastbox，还是依据上一帧预测的目标在当前帧的位置
	  // 4. Failure detection,检测 getFB()>10 || 完全出轨 
	  //跟踪失败检测：如果FB error的中值大于10个像素（经验值），或者预测到的当前box的位置移出图像，则  
	  //认为跟踪错误，此时不返回bounding box；Rect::br()返回的是右下角的坐标  
	  //getFB()返回的是FB error的中值  
	  //对跟踪结果进行判断，如果fbmed超过10(固定经验值)、或者窗口坐标位于图像外，说明跟踪的结果不稳定，将跟踪结果丢弃，置tvalid和tracked为false并进入下一帧，否则继续进行下面的过程。
	  if (tracker.getFB()>10 || tbb.x>img2.cols ||  tbb.y>img2.rows || tbb.br().x < 1 || tbb.br().y <1){//br() bottom right坐标
          tvalid =false; //too unstable prediction or bounding box out of image
          tracked = false;
          printf("Too unstable predictions FB error=%f\n",tracker.getFB());
          return;
      }
      // 5. Estimate Confidence and Validity
	   //评估跟踪确信度和有效性   
      Mat pattern;
      Scalar mean, stdev;
      BoundingBox bb;
      bb.x = max(tbb.x,0);
      bb.y = max(tbb.y,0);
      bb.width = min(min(img2.cols-tbb.x,tbb.width),min(tbb.width,tbb.br().x));
      bb.height = min(min(img2.rows-tbb.y,tbb.height),min(tbb.height,tbb.br().y));
       //归一化img2(bb)对应的patch的size（放缩至patch_size = 15*15），存入pattern  
	  getPattern(img2(bb),pattern,mean,stdev);
      vector<int> isin;
      float dummy;
	  //估计置信度和有效性，调用getPattern()计算当前帧在跟踪结果区域的pattern，把pattern做为输入调用NNConf计算它与在线模型的保守相似度tconf，
	  //如果tconf>thr_nn_valid，则置tvalid为true，也即表示当前跟踪结果有效，否则tvalid仍为上一帧的值。
	   //计算图像片pattern到在线模型M的保守相似度  
      classifier.NNConf(pattern,isin,dummy,tconf); //1.tconf是用Conservative Similarity
      tvalid = lastvalid;	  
	   //保守相似度大于阈值，则评估跟踪有效 
      if (tconf>classifier.thr_nn_valid){//thr_nn_valid
          tvalid =true;//2.判定轨迹是否有效，从而决定是否要增加正样本，标志位tvalid【5.6.2 P-Expert】
      }
  }
  else
    printf("No points tracked\n");
}
//将bb切成10*10的网格，将网格交点存在points
//网格均匀撒点，box共10*10=100个特征点   
void TLD::bbPoints(vector<cv::Point2f>& points,const BoundingBox& bb){
  int max_pts=10;
  int margin_h=0;//留白没有用到 //采样边界  
  int margin_v=0;
  int stepx = ceil((bb.width-2.0*margin_h)/max_pts);//向上取整
  int stepy = ceil((bb.height-2.0*margin_v)/max_pts);
   //网格均匀撒点，box共10*10=100个特征点   
  for (int y=bb.y+margin_v;y<bb.y+bb.height-margin_v;y+=stepy){
      for (int x=bb.x+margin_h;x<bb.x+bb.width-margin_h;x+=stepx){
          points.push_back(Point2f(x,y));//最多有11*11=121个点
      }
  }
}
//依据points1，points2估计bb1的位移和尺度变化，这两个信息都有了，自然可以决定其范围bb2
//利用剩下的这不到一半的跟踪点输入来预测bounding box在当前帧的位置和大小
//它先按x维度和y维度分别计算所有点在上一帧和当前帧的跟踪差距，并计算出这些差距的中值dx，dy。
//接下来计算points1中所有点两两之间的距离d1和points2中所有点两两之间的距离d1，将d2/d1都放到d中，
//计算所有的d的中值s，这个s就表征了进行光流跟踪之后特征点变化相对于上一帧位置的偏移比例，
//从而用s去乘以上一帧窗口的宽高度以得到当前帧跟踪结果窗口的宽高度；
//另外，跟踪窗口的左上角坐标也要更新，坐标的偏移不能只考虑位移的绝对值，还要考虑窗口本身宽高度，也即这个位移相对于窗口本身的比例，
//所以用0.5*(s-1)分别去乘以上一帧窗口的宽度和高度，得到偏移量s1和s2，
//再结合表征了上一帧和当前帧跟踪差距的dx和dy，得到新跟踪结果tbb(track bounding box)的左上角坐标(lastbox.x + dx -s1, lastbox.y + dy -s2)。
void TLD::bbPredict(const vector<cv::Point2f>& points1,const vector<cv::Point2f>& points2,
                    const BoundingBox& bb1,BoundingBox& bb2)    {
  int npoints = (int)points1.size();
  vector<float> xoff(npoints);//位移
  vector<float> yoff(npoints);
  printf("tracked points : %d\n",npoints);
  // 用位移的中值，作为目标位移的估计
  for (int i=0;i<npoints;i++){//计算每个特征点在两帧之间的位移 
      xoff[i]=points2[i].x-points1[i].x;
      yoff[i]=points2[i].y-points1[i].y;
  }
  float dx = median(xoff);////计算位移的中值  
  float dy = median(yoff);
  float s;
  //计算bounding box尺度scale的变化：通过计算 当前特征点相互间的距离 与 先前（上一帧）特征点相互间的距离 的  
  //比值，以比值的中值作为尺度的变化因子   
  // 用点对之间的距离的伸缩比例的中值，作为目标尺度变化的估计
  if (npoints>1){
      vector<float> d;
      d.reserve(npoints*(npoints-1)/2);//等差数列求和：1+2+...+(npoints-1)  
      for (int i=0;i<npoints;i++){
          for (int j=i+1;j<npoints;j++){
			  //之前比较的都是对应点之间的相似性，现在计算的是任意两点的相似性，所以更能反映拓扑结构的变化
			  //问题：假设比值是s,那么加个min(s，1/s)不是更好吗？？？？
			  //呃，好吧，亲你又YY了，这一步不是干这个的好吗？，so,这三行都忽略
			   //计算 当前特征点相互间的距离 与 先前（上一帧）特征点相互间的距离 的比值（位移用绝对值）  
              d.push_back(norm(points2[i]-points2[j])/norm(points1[i]-points1[j]));
          }
      }
      s = median(d);//
  }
  else {
      s = 1.0;
  }
  float s1 = 0.5*(s-1)*bb1.width;// top-left 坐标的偏移(s1,s2)
  float s2 = 0.5*(s-1)*bb1.height;
  printf("s= %f s1= %f s2= %f \n",s,s1,s2);
  //得到当前bounding box的位置与大小信息   
  //当前box的x坐标 = 前一帧box的x坐标 + 全部特征点位移的中值（可理解为box移动近似的位移） - 当前box宽的一半  
  bb2.x = cvRound( bb1.x + dx -s1);
  bb2.y = cvRound( bb1.y + dy -s2);
  bb2.width = cvRound(bb1.width*s);
  bb2.height = cvRound(bb1.height*s);
  printf("predicted bb: %d %d %d %d\n",bb2.x,bb2.y,bb2.br().x,bb2.br().y);
}
/*
*方差->随机深林->最近邻
*Input: 
	-bb:	grid ？？？？
*Ouput: 
	-bb:	dbb
	-conf:	dconf //Conservative Similarity (for integration with tracker)
*/
void TLD::detect(const cv::Mat& frame){
  //cleaning
	//计算积分图和积分平方图，对img2进行高斯平滑。之后对所有扫描窗口，依次进入以下级联着的分类器
  dbb.clear();
  dconf.clear();
  dt.bb.clear();//检测的结果，一个目标一个bounding box
  double t = (double)getTickCount();//GetTickCount返回从操作系统启动到现在所经过的时间  
  Mat img(frame.rows,frame.cols,CV_8U);
  integral(frame,iisum,iisqsum);////计算frame的积分图   
  GaussianBlur(frame,img,Size(9,9),1.5);//
  int numtrees = classifier.getNumStructs();// nstructs： 10
  float fern_th = classifier.getFernTh();//thr_fern：0.6
  vector <int> ferns(10);
  float conf;
  int a=0;
  Mat patch;//级联分类器模块一：方差检测模块，利用积分图计算每个待检测窗口的方差，方差大于var阈值（目标patch方差的50%）的，  
	  //则认为其含有前景目标   
  // 1. 方差->结果存在tmp -> 随机森林-> dt.bb
  for (int i=0;i<grid.size();i++){//FIXME: BottleNeck
	  //方差检测分类器，利用积分图调用getVar()计算每个扫描窗口的方差，如果方差大于var阈值（var初始初始是best_box标准差平方的一半，也即目标patch方差的一半），
	  //则认为其含有前景目标，(注意：虽说标准差是方差的平方根，但是程序中求方差和标准差是用两个不同的函数计算的)。
      if (getVar(grid[i],iisum,iisqsum)>=var){//第一关：方差//计算每一个扫描窗口的方差
          a++;
		  patch = img(grid[i]); //级联分类器模块二：集合分类器检测模块  
          //Fern分类器，对通过前面方差检测分类器的扫描窗口，调用getFeatures()计算10棵树对该扫描窗口的编码(长度为13的01序列)，
		  //利用该编码结合measure_forest()得到10棵树对该扫描窗口的10个后验概率累加和的平均值，如果平均值大于Fern分类器的阈值fern_th(初始从myParam.yml获取是0.65后面会不断更新)，
		  //则该扫描窗口也通过Fern分类器的检测，将窗口放到容器中以便后面最近邻分类器的检测
		  classifier.getFeatures(patch,grid[i].sidx,ferns);//sidx:scale index//得到该patch特征（13位的二进制代码）  
          conf = classifier.measure_forest(ferns);//第二关：随机森林 //计算该特征值对应的后验概率累加值  
          tmp.conf[i]=conf;
          tmp.patt[i]=ferns;//只要能通过第一关就会保存到tmp
          if (conf>numtrees*fern_th){ //如果集合分类器的后验概率的平均值大于阈值fern_th（由训练得到），就认为含有前景目标  
              dt.bb.push_back(i);//第二关/将通过以上两个检测模块的扫描窗口记录在detect structure中  
          }
      }
      else
        tmp.conf[i]=0.0;//第一关都没过
  }
  //通过前面两个分类器检测的扫描窗口按照后验概率累加和conf进行降序排列，如果窗口数目超过100个，则取前面100个后验概率较大的
  int detections = dt.bb.size();
  printf("%d Bounding boxes passed the variance filter\n",a);
  printf("%d Initial detection from Fern Classifier\n",detections);
  if (detections>100){// 第二关附加赛：100名以后的回家去 //如果通过以上两个检测模块的扫描窗口数大于100个，则只取后验概率大的前100个  
      nth_element(dt.bb.begin(),dt.bb.begin()+100,dt.bb.end(),CComparator(tmp.conf));
      dt.bb.resize(100);
      detections=100;
  }
//  for (int i=0;i<detections;i++){
//        drawBox(img,grid[dt.bb[i]]);
//    }
//  imshow("detections",img);
  if (detections==0){
        detected=false;
        return;//啥都没看到……
      }
  printf("Fern detector made %d detections ",detections);
  t=(double)getTickCount()-t;//两次使用getTickCount()，然后再除以getTickFrequency()，计算出来的是以秒s为单位的时间（opencv 2.0 以前是ms）  
  printf("in %gms\n", t*1000/getTickFrequency());
                                                                       //  Initialize detection structure
  dt.patt = vector<vector<int> >(detections,vector<int>(10,0));        //  Corresponding codes of the Ensemble Classifier
  dt.conf1 = vector<float>(detections);                                //  Relative Similarity (for final nearest neighbour classifier)
  dt.conf2 =vector<float>(detections);                                 //  Conservative Similarity (for integration with tracker)
  dt.isin = vector<vector<int> >(detections,vector<int>(3,-1));        //  Detected (isin=1) or rejected (isin=0) by nearest neighbour classifier
  dt.patch = vector<Mat>(detections,Mat(patch_size,patch_size,CV_32F));//  Corresponding patches,patch_size: 15
  int idx;
  Scalar mean, stdev;
  float nn_th = classifier.getNNTh();//thr_nn:0.65
  //对这些窗口，调用getPattern()获取pattern，然后调用NNConf()计算与在线模型的相关相似度conf1和保守相似度conf2，如果相关相似度conf1大于近邻分类器的阈值nn_th，
  //则保留该窗口及其对应的保守相似度的值，通过以上三个分类器检测的扫描窗口放到dbb容器中，对应的保守相似度放到dconf容器中
  //3. 第三关：最近邻分类器，用Relative Similarity分类，但是却用 Conservative Similarity作为分数->dconf
   //级联分类器模块三：最近邻分类器检测模块   
  for (int i=0;i<detections;i++){                                         //  for every remaining detection
      idx=dt.bb[i];                                                       //  Get the detected bounding box index
	  patch = frame(grid[idx]);
      getPattern(patch,dt.patch[i],mean,stdev);                //  Get pattern within bounding box
      //计算图像片pattern到在线模型M的相关相似度和保守相似度  
	  classifier.NNConf(dt.patch[i],dt.isin[i],dt.conf1[i],dt.conf2[i]);  //  Evaluate nearest neighbour classifier
      dt.patt[i]=tmp.patt[idx];//ferns
      //printf("Testing feature %d, conf:%f isin:(%d|%d|%d)\n",i,dt.conf1[i],dt.isin[i][0],dt.isin[i][1],dt.isin[i][2]);
      //相关相似度大于阈值，则认为含有前景目标   
	  if (dt.conf1[i]>nn_th){                                               //  idx = dt.conf1 > tld.model.thr_nn; % get all indexes that made it through the nearest neighbour
          dbb.push_back(grid[idx]);                                         //  BB    = dt.bb(:,idx); % bounding boxes
          dconf.push_back(dt.conf2[i]);                                     //  Conf  = dt.conf2(:,idx); % conservative confidences
      }
  }     
  //打印检测到的可能存在目标的扫描窗口数（可以通过三个级联检测器的）//  end
  if (dbb.size()>0){
      printf("Found %d NN matches\n",(int)dbb.size());
      detected=true;
  }
  else{
      printf("No NN matches found.\n");
      detected=false;
  }
}
//作者已经用python脚本../datasets/evaluate_vis.py来完成算法评估功能，具体见README 
void TLD::evaluate(){
}
/*
* Fun:
*	更新决策森林和NN
*
*/
void TLD::learn(const Mat& img){// current_gray
  printf("[Learning] ");
  ///Check consistency
  //首先，用上一帧的跟踪区域lastbox在当前帧上截取
  BoundingBox bb;
  bb.x = max(lastbox.x,0); //lastbox 
  bb.y = max(lastbox.y,0);
  bb.width = min(min(img.cols-lastbox.x,lastbox.width),min(lastbox.width,lastbox.br().x));
  bb.height = min(min(img.rows-lastbox.y,lastbox.height),min(lastbox.height,lastbox.br().y));
  //调用getPattern()得到其pattern和方差
  Scalar mean, stdev;
  Mat pattern; //归一化img(bb)对应的patch的size（放缩至patch_size = 15*15），存入pattern  
  getPattern(img(bb),pattern,mean,stdev);// pattern：resized Zero-Mean patch，为什么要弄成0均值呢？是算相关系数
  vector<int> isin;
  float dummy, conf;
  // 1. 再粗略地检测一遍，因为结果是加权的，如果偏差很大，那岂不是误导检测器
  //计算输入图像片（跟踪器的目标box）与在线模型之间的相关相似度conf   
  //，再调用NNConf()计算其与在线模型最近邻数据集的相关相似度conf
  classifier.NNConf(pattern,isin,conf,dummy);
  //如果相关相似度太小、或者方差太小、或者被识别为负样本，则不进行训练，lastvalid置为false表示上一帧失效然后直接return
  if (conf<0.5) {//Relative Similarity，注意:nn_thr >= 0.65，所以阈值降低了，因为我们要迎接新人//如果相似度太小了，就不训练  
      printf("Fast change..not training\n");//形变是缓慢的，你如此不同，应该不是同类
      lastvalid =false;
      return;
  }
  if (pow(stdev.val[0],2)<var){ //如果方差太小了，也不训练  
      printf("Low variance..not training\n");
      lastvalid=false;
      return;
  }
  if(isin[2]==1){//是否是负样本//如果被被识别为负样本，也不训练  
	  //问题：这和conf<0.5的区别？isin[2]==1-> nccN > ncc_thesame，所以这里只是单独和负样本比较的
	  //问题：NN认为是负样本，那么应该是小T错了，那么不应该更新下bbnext吗？？？
      printf("Patch in negative data..not traing");
      lastvalid=false;
      return;
  }
/// Data generation样本产生   
  for (int i=0;i<grid.size();i++){//为getOverlappingBoxes函数预先计算grid[i].overlap//计算所有的扫描窗口与目标box的重叠度  
      grid[i].overlap = bbOverlap(lastbox,grid[i]);
  }
   //集合分类器   
  vector<pair<vector<int>,int> > fern_examples;
  good_boxes.clear();
  bad_boxes.clear();
  // 2. 用lastbox，重新计算good,bad,best bb 还有 bbhull
  //此函数根据传入的lastbox，在整帧图像中的全部窗口中寻找与该lastbox距离最小（即最相似，  
  //重叠度最大）的num_closest_update个窗口，然后把这些窗口 归入good_boxes容器（只是把网格数组的索引存入）  
  //同时，把重叠度小于0.2的，归入 bad_boxes 容器 
//如果以上条件都不满足，则更新当前帧扫描窗口与上一帧窗口lastbox的重叠率，调用getOverlappingBoxes(lastbox,num_closest_update)更新best_box、good­_boxes、bad_boxes、bbhull
  //此时bad_boxes中的规模为num_closest_update(值为10，从myParam.yml中获取)，也即bad_boxes的规模一直不变
  getOverlappingBoxes(lastbox,num_closest_update);//num_closest_update ： 10
  //然后借助good_boxes结合generatePositiveData(img,num_warps_update)清空并重新生成pX(仍然存储仿射变换后所得的200个样本)和fern中每棵树对pX中200个正样本的编码。
  if (good_boxes.size()>0)// 问题：grid是在所有范围内产生的，会出现good_boxes.size<=0吗？？//用仿射模型产生正样本（类似于第一帧的方法，但只产生10*10=100个）  
  // 3. 更新这一帧的 pX pEx,【5.6.2 P-Expert】 
	  generatePositiveData(img,num_warps_update);//注意：是用best_box，而不是lastbox
  else{
    lastvalid = false;
    printf("No good boxes..Not training");
    return;
  }
  fern_examples.reserve(pX.size()+bad_boxes.size());
  fern_examples.assign(pX.begin(),pX.end());
  int idx;
  // 4. 从bad_boxes挑选hard negative作为新增的随机森林训练负样本集【5.6.3 N-Expert】
  for (int i=0;i<bad_boxes.size();i++){
      idx=bad_boxes[i];
	  //对bad_boxes中的样本进行判断，如果10棵树对bad_boxes中的样本得到的10个后验概率之和超过1，则将其作为Fern分类器fern_examples中的样本，用于后面对Fern分类器的阈值进行更新。
      if (tmp.conf[idx]>=1){//回忆一下 grid->方差-> 结果存在tmp，conf是随机森林的分数//加入负样本，相似度大于1？？相似度不是出于0和1之间吗？  
          fern_examples.push_back(make_pair(tmp.patt[idx],0));
      }
  }
  // 5. 从dt.bb中挑选hard negative作为新增的最近邻分类器的负样本集【5.6.3 N-Expert】
  // 问题：为什么 5的样本（只从dt里面挑）远远小于4（从真个grid里面挑），这个和分类器有关，决策森林需要很多样本才嫩保证正确率，而NN负担不起太多样本
  //最近邻分类器   
  //接下来更新近邻数据集nn_examples，将检测模块中通过方差分类器和Fern分类器的样本dt与lastbox进行重叠率计算看是否不超过bad_overlap(固定值0.2从myParam.yml中读取)，不超过则可以将其放入近邻数据集nn_examples中，用于后面近邻分类器的更新训练和阈值的更新。
  vector<Mat> nn_examples;
  nn_examples.reserve(dt.bb.size()+1);
  nn_examples.push_back(pEx);//唯一一个正样本
  for (int i=0;i<dt.bb.size();i++){
      idx = dt.bb[i];
      if (bbOverlap(lastbox,grid[idx]) < bad_overlap)
        nn_examples.push_back(dt.patch[i]);
  }
  /// 6. Classifiers update分类器训练   
  classifier.trainF(fern_examples,2);
  classifier.trainNN(nn_examples);
  //问题：fern_examples和nn_examples都是新的数据，完全没有用到之前的pX，pEx,nx,nEx？？？
  //原来，随机森林分类器只要保存直方图统计就可以了，所以不需要存储正负样本集
  //而最近邻分类器，并没有clear pEx,nEx,而是不断地追加正负样本
  classifier.show(); //把正样本库（在线模型）包含的所有正样本显示在窗口上
}
/*
* Fun：计算img中，所有位置，所有尺度的bb(长宽比与box相同)与box的overlap
* In
	-img：
	-box：		我们自己框定的bb
* Out
	-grid:		所有bb都在这里
*/
//检测器采用扫描窗口的策略   
//此函数根据传入的box（目标边界框）在传入的图像中构建全部的扫描窗口，并计算每个窗口与box的重叠度 
//这个函数输入是当前帧frame和初始跟踪区域box，主要功能是获取当前帧中的所有扫描窗口，这些窗口有21个尺度，
//缩放系数是1.2，也即以1为中间尺度，进行10次1.2倍的缩小和10次1.2倍的放大。
//在每个尺度下，扫描窗口的步长为宽高的10%(SHIFT=0.1)，从而得到所有的扫描窗口存放在容器grid中，
//这个grid的每个元素包含6个属性：当前扫描窗口左上角的x坐标、y坐标、宽度w、高度h、
//与初始跟踪区域box的重叠率overlap、当前扫描窗口所在的尺度sidx。
//其中重叠率的定义是：两个窗口的交集/两个窗口的并集。与此同时，
//如果当前窗口的宽度和高度不小于最小窗口的阈值min_win(固定值15，从myParam.yml中获取)，
//就将这个尺寸放到scales容器中，由于有min_win的限制，所以某些特别消的尺度下的扫描窗口尺寸就不会放到scales中，
//故该容器长度小于等于2
void TLD::buildGrid(const cv::Mat& img, const cv::Rect& box){
  const float SHIFT = 0.1;
  const float SCALES[] = {0.16151,0.19381,0.23257,0.27908,0.33490,0.40188,0.48225,
                          0.57870,0.69444,0.83333,1,1.20000,1.44000,1.72800,
                          2.07360,2.48832,2.98598,3.58318,4.29982,5.15978,6.19174};//scales step 1.2，最多21层
  int width, height, min_bb_side;
  //Rect bbox;
  BoundingBox bbox;
  Size scale;
  int sc=0;
  for (int s=0;s<21;s++){
    width = cvRound(box.width*SCALES[s]);
    height = cvRound(box.height*SCALES[s]);
    min_bb_side = min(height,width);//bounding box最短的边  
	//由于图像片（min_win 为15x15像素）是在bounding box中采样得到的，所以box必须比min_win要大  
	//另外，输入的图像肯定得比 bounding box 要大了  
    if (min_bb_side < min_win || width > img.cols || height > img.rows)
      continue;
	//问题：检测窗口的尺寸越大，那么当前尺度层都不能检测，下一层更加不用说了，直接break岂不快哉？
	//原来尺度不是一直增大的，下10层，上10层
    scale.width = width;
    scale.height = height;
    scales.push_back(scale);
	//push_back在vector类中作用为在vector尾部加入一个数据  
	//scales在类TLD中定义：std::vector<cv::Size> scales;  
	 //把该尺度的窗口存入scales容器，避免在扫描时计算，加快检测速度  
    for (int y=1;y<img.rows-height;y+=cvRound(SHIFT*min_bb_side)){//步长分别是以检测窗口width,height的0.1
      for (int x=1;x<img.cols-width;x+=cvRound(SHIFT*min_bb_side)){
        bbox.x = x;
        bbox.y = y;
        bbox.width = width;
        bbox.height = height;
		//判断传入的bounding box（目标边界框）与 传入图像中的此时窗口的 重叠度，  
		//以此来确定该图像窗口是否含有目标   
        bbox.overlap = bbOverlap(bbox,BoundingBox(box));//重合面积交/并
        bbox.sidx = sc;//尺度索引
		//grid在类TLD中定义：std::vector<BoundingBox> grid;  
		//把本位置和本尺度的扫描窗口存入grid容器 
        grid.push_back(bbox);//bbox的形状和box都是一样的
      }
    }
    sc++;
  }
}
// intersection/union
//此函数计算两个bounding box 的重叠度   
//重叠度定义为 两个box的交集 与 它们的并集 的比   
 //先判断坐标，假如它们都没有重叠的地方，就直接返回0   
float TLD::bbOverlap(const BoundingBox& box1,const BoundingBox& box2){
  if (box1.x > box2.x+box2.width) { return 0.0; }
  if (box1.y > box2.y+box2.height) { return 0.0; }
  if (box1.x+box1.width < box2.x) { return 0.0; }
  if (box1.y+box1.height < box2.y) { return 0.0; }

  float colInt =  min(box1.x+box1.width,box2.x+box2.width) - max(box1.x, box2.x);
  float rowInt =  min(box1.y+box1.height,box2.y+box2.height) - max(box1.y,box2.y);

  float intersection = colInt * rowInt;
  float area1 = box1.width*box1.height;
  float area2 = box2.width*box2.height;
  return intersection / (area1 + area2 - intersection);
}
/*
* Fun:以box1为匹配目标，将所有grid的bb进行匹配，按照overlap的大小，分为good_boxes，bad_boxes
* In
	-box1:			主要是小D认为目标的所在区域
	-num_closest：	初始化为10
* Out
	-good_boxes:	overlap > 0.6，最多只收num_closest个
	-bad_boxes:		overlap < bad_overlap
	-best_box:		并没有直接用box1，而是从已有的grid里面挑
* Note:				顺带还计算了good_boxes的bbhull
//此函数根据传入的box1（目标边界框），在整帧图像中的全部窗口中寻找与该box1距离最小（即最相似，  
//重叠度最大）的num_closest个窗口，然后把这些窗口 归入good_boxes容器（只是把网格数组的索引存入）  
//同时，把重叠度小于0.2的，归入 bad_boxes 容器   
*/
//主要功能
//A.找出与初始跟踪窗口box重叠率最高的扫描窗口best_box
//B.找出与初始跟踪窗口box重叠率超过0.6的，将其当成较好的扫描窗口，可以用于后面产生正样本，放到good_boxes中
//C.找出与初始跟踪窗口box重叠率小于0.2的，将其当成较差的扫描窗口，可以用于后面产生负样本，放到bad_boxes中
//D.调用getBBHull()函数，获取good_boxes中所有窗口并集的外接矩形
void TLD::getOverlappingBoxes(const cv::Rect& box1,int num_closest){
  float max_overlap = 0;
  for (int i=0;i<grid.size();i++){
      if (grid[i].overlap > max_overlap) {//找出重叠度最大的box  
          max_overlap = grid[i].overlap;
          best_box = grid[i];
      }
      if (grid[i].overlap > 0.6){ //重叠度大于0.6的，归入 good_boxes 
          good_boxes.push_back(i);
      }
      else if (grid[i].overlap < bad_overlap){ //重叠度小于0.2的，归入 bad_boxes  
          bad_boxes.push_back(i);
      }
  }
  //Get the best num_closest (10) boxes and puts them in good_boxes
  // 只保留重合面积前 num_closest 的bb
  //STL中的nth_element()方法找出一个数列中排名第n（下面为第num_closest）的那个数。这个函数运行后  
  //在good_boxes[num_closest]前面num_closest个数都比他大，也就是找到最好的num_closest个box了  
  if (good_boxes.size()>num_closest){
    std::nth_element(good_boxes.begin(),good_boxes.begin()+num_closest,good_boxes.end(),OComparator(grid));
    good_boxes.resize(num_closest); //重新压缩good_boxes为num_closest大小   
  }
  getBBHull();//获取good_boxes 的 Hull壳，也就是窗口的边框  
}
//Out：bbhull:能将所有good_boxes包起来的最小bb，
//此函数获取good_boxes 的 Hull壳，也就是窗口（图像）的边框 bounding box  
void TLD::getBBHull(){
  int x1=INT_MAX, x2=0;
  int y1=INT_MAX, y2=0;
  int idx;
  for (int i=0;i<good_boxes.size();i++){
      idx= good_boxes[i];
      x1=min(grid[idx].x,x1);
      y1=min(grid[idx].y,y1);
      x2=max(grid[idx].x+grid[idx].width,x2);
      y2=max(grid[idx].y+grid[idx].height,y2);
  }
  bbhull.x = x1;
  bbhull.y = y1;
  bbhull.width = x2-x1;
  bbhull.height = y2 -y1;
}
//阈值 0.5
//如果两个box的重叠度小于0.5，返回false，否则返回true  
bool bbcomp(const BoundingBox& b1,const BoundingBox& b2){
  TLD t;
    if (t.bbOverlap(b1,b2)<0.5)
      return false;
    else
      return true;
}
/*
* In：
	-dbb：检测器所检测到的bb
* Out:
	-indexes:
* Ret:
	-聚类之后的个数
* Detail:
距离度量：重合面积
方法：	分层聚类
终止：	min_d>0.5
*/
int TLD::clusterBB(const vector<BoundingBox>& dbb,vector<int>& indexes){
  //FIXME: Conditional jump or move depends on uninitialised value(s)
  const int c = dbb.size();
  //1. Build proximity matrix，两两之间的距离（使用重合面积计算）
  Mat D(c,c,CV_32F);
  float d;
  for (int i=0;i<c;i++){
      for (int j=i+1;j<c;j++){
		d = 1 - bbOverlap(dbb[i], dbb[j]);  //不相似性/距离
        D.at<float>(i,j) = d;
        D.at<float>(j,i) = d;
      }
  }
  //2. Initialize disjoint clustering
 //float L[c-1]; //Level
  float *L = new float[c - 1]; //Level
 //int nodes[c-1][2];
  int **nodes;//记录每一次合并，合并的是哪两个类
  nodes = new int * [c - 1];
  for (int i = 0; i < c - 1; i++)
  {
	  nodes[i]=new int[2];
  }
 //int belongs[c];
 int *belongs = new int[c];
 int m=c;
 for (int i=0;i<c;i++){
    belongs[i]=i;//各自划地为王
 }
 for (int it=0;it<c-1;it++){//c个，自然最多能合并c-1次
 //3. Find nearest neighbor
     float min_d = 1;
     int node_a, node_b;
     for (int i=0;i<D.rows;i++){
         for (int j=i+1;j<D.cols;j++){
             if (D.at<float>(i,j)<min_d && belongs[i]!=belongs[j]){
                 min_d = D.at<float>(i,j);
                 node_a = i;
                 node_b = j;
             }
         }
     }
     if (min_d>0.5){//最近的距离都这么大了，说明已经够分散了（先看第4步）
         int max_idx =0;
         bool visited;
         for (int j=0;j<c;j++){//计算有几类
             visited = false;
             for(int i=0;i<2*c-1;i++){//由于m是从c开始递增的，所以最大不会超过c+c-1
                 if (belongs[j]==i){
                     indexes[j]=max_idx;// 有问题吧！！！！
                     visited = true;
                 }
             }
             if (visited)
               max_idx++;
         }
         return max_idx;//
     }

 //4. Merge clusters and assign level
     L[m]=min_d;
     nodes[it][0] = belongs[node_a];
     nodes[it][1] = belongs[node_b];
     for (int k=0;k<c;k++){
         if (belongs[k]==belongs[node_a] || belongs[k]==belongs[node_b])
           belongs[k]=m;//呃，隐含的是belongs[node_a]=belongs[node_b]=m
						//当然还要附带的吧node_a和node_b的家属统统改成一类
     }
     m++;
 }
 // free
 delete[] L;
 delete[] belongs;
 for (int i = 0; i < c - 1; i++)
 {
	 delete [] nodes[i];
 }
 delete[] nodes;
 return 1;//最后合并成一个类了……

}

//对检测器检测到的目标bounding box进行聚类   
//聚类（Cluster）分析是由若干模式（Pattern）组成的，通常，模式是一个度量（Measurement）的向量，或者是多维空间中的  
//一个点。聚类分析以相似性为基础，在一个聚类中的模式之间比不在同一聚类中的模式之间具有更多的相似性。  
void TLD::clusterConf(const vector<BoundingBox>& dbb,const vector<float>& dconf,vector<BoundingBox>& cbb,vector<float>& cconf){
  int numbb =dbb.size();
  vector<int> T;
  float space_thr = 0.5;
  int c=1; //记录 聚类的类个数  
  switch (numbb){//检测到的含有目标的bounding box个数  
  case 1://这个函数针对dbb中窗口的数目采取不同策略，如果只有一个窗口，那么不用聚类，直接把dbb和dconf分别赋给cbb和cconf即可
    cbb=vector<BoundingBox>(1,dbb[0]); //如果只检测到一个，那么这个就是检测器检测到的目标  
    cconf=vector<float>(1,dconf[0]);
    return;
    break;
  case 2://如果有两个窗口，调用bbOverlap()计算两个窗口的重叠率，如果重叠率小于一定的阈值1-space_thr(固定值为0.5)，则分为两类
    T =vector<int>(2,0);//此函数计算两个bounding box 的重叠度 
    if (1-bbOverlap(dbb[0],dbb[1])>space_thr){//分歧很大//如果只检测到两个box，但他们的重叠度小于0.5  
      T[1]=1;
      c=2;//重叠度小于0.5的box，属于不同的类 
    }
    break;
  default://检测到的box数目大于2个，则筛选出重叠度大于0.5的  
	  //stable_partition()重新排列元素，使得满足指定条件的元素排在不满足条件的元素前面。它维持着两组元素的顺序关系。  
	  //STL partition就是把一个区间中的元素按照某个条件分成两类。返回第二类子集的起点  
	  //bbcomp()函数判断两个box的重叠度小于0.5，返回false，否则返回true （分界点是重叠度：0.5）  
	  //partition() 将dbb划分为两个子集，将满足两个box的重叠度小于0.5的元素移动到序列的前面，为一个子集，重叠度大于0.5的，  
	  //放在序列后面，为第二个子集，但两个子集的大小不知道，返回第二类子集的起点  
	  //如果窗口超过两个，那么调用partition()函数按照重叠率将窗口分为两类，一类是重叠率小于0.5的，另一类是超过0.5的。
    T = vector<int>(numbb,0);
	//把一个区间中的元素按照某个条件分成两类，并返回第二类子集的起点
	//bbcomp这个指针所对应的函数是比较两个窗口重叠率，小于0.5返回假，否则返回真
    c = partition(dbb,T,(*bbcomp));//重叠度小于0.5的box，属于不同的类，所以c是不同的类别个数  
    //c = clusterBB(dbb,T);
    break;
  }
  //分成多个等价类，而不一定是两个，函数返回的c是等价类的数目，T是每个元素所属的类别标签。
  //比如有ABCDEFG七个窗口，经过调用后ABC是一类，两两之间重叠率小于0.5，DE是一类，FG是一类，
  //DE与ABC的所有窗口两两之间重叠率超过0.5，FG与ABC的所有窗口重叠率超过0.5，
  //而DE和FG之间的窗口两两之间重叠率也超过0.5。在聚类结束之后，把每个类别中的窗口信息进行平均，
  //也即ABC所有窗口的x坐标求和再除以该类别数目3，以其作为聚类中心窗口的x坐标，其它信息包括y坐标、宽度、高度、保守相似度也按如此计算，
  //并保存到cbb和cconf中，也即cbb和cconf的规模是聚类的数目。
  cconf=vector<float>(c);
  cbb=vector<BoundingBox>(c);
  printf("Cluster indexes: ");
  BoundingBox bx;
  for (int i=0;i<c;i++){//类别个数  
      float cnf=0;
      int N=0,mx=0,my=0,mw=0,mh=0;
      for (int j=0;j<T.size();j++){ 
          if (T[j]==i){
              printf("%d ",i);
              cnf=cnf+dconf[j];
              mx=mx+dbb[j].x;
              my=my+dbb[j].y;
              mw=mw+dbb[j].width;
              mh=mh+dbb[j].height;
              N++;
          }
      }
      if (N>0){ //然后求该类的box的坐标和大小的平均值，将平均值作为该类的box的代表  
          cconf[i]=cnf/N;
          bx.x=cvRound(mx/N);
          bx.y=cvRound(my/N);
          bx.width=cvRound(mw/N);
          bx.height=cvRound(mh/N);
          cbb[i]=bx;//返回的是聚类，每一个类都有一个代表的bounding box  
      }
  }
  printf("\n");
}


