/*
 * FerNNClassifier.cpp
 *
 *  Created on: Jun 14, 2011
 *      Author: alantrrs
 */

#include "FerNNClassifier.h"

using namespace cv;
using namespace std;
//下面这些参数通过程序开始运行时读入parameters.yml文件进行初始化  
void FerNNClassifier::read(const FileNode& file){
  ///Classifier Parameters
  valid = (float)file["valid"];
  ncc_thesame = (float)file["ncc_thesame"];
  nstructs = (int)file["num_trees"];//树木（由一个特征组构建，每组特征代表图像块的不同视图表示）的个数  
  structSize = (int)file["num_features"];//每棵树的特征个数，也即每棵树的节点个数；树上每一个特征都作为一个决策节点  
  thr_fern = (float)file["thr_fern"];
  thr_nn = (float)file["thr_nn"];
  thr_nn_valid = (float)file["thr_nn_valid"];
}
/*
* In
*	-scales:		所有能检测的尺度层的尺度
*	-nstructs：	（10）
* Out
*	-features:   Ferns features (one std::vector for each scale) ????
*	-thrN:		 0.5*nstructs ???
* Init
*   -nCounter:	特征取值的分布区间 2^13个bin,见 FerNNClassifier::update
*   -pCounter:
*/
void FerNNClassifier::prepare(const vector<Size>& scales){
  acum = 0;
  // 1. Initialize test locations for features
  //   随机产生需要坐标对（x1f，y1f，x2f，y2f，注意范围[0,1)），
  //   即确定由每一个特征是由哪些点对进行而得到，这些位置一旦确定就不会改变，
  //   由于我们要进行多尺度检测，所以同一个点在不同尺度scales，实际对应的坐标要乘以实际的width和height。 
  int totalFeatures = nstructs*structSize;//nstructs 10 structSize 13
  //二维向量  包含全部尺度（scales）的扫描窗口，每个尺度包含totalFeatures个特征  
  features = vector<vector<Feature> >(scales.size(),vector<Feature> (totalFeatures));
  RNG& rng = theRNG();
  float x1f,x2f,y1f,y2f;
  int x1, x2, y1, y2;
  //随机蕨分类器基于n个基本分类器，每个分类器都是基于一个pixel comparisons（像素比较集）的；  
  //pixel comparisons的产生方法：先用一个归一化的patch去离散化像素空间，产生所有可能的垂直和水平的pixel comparisons  
  //然后我们把这些pixel comparisons随机分配给n个分类器，每个分类器得到完全不同的pixel comparisons（特征集合），  
  //这样，所有分类器的特征组统一起来就可以覆盖整个patch了   
  //用随机数去填充每一个尺度扫描窗口的特征   
  for (int i=0;i<totalFeatures;i++){
      x1f = (float)rng; //产生[0,1)直接的浮点数
      y1f = (float)rng;
      x2f = (float)rng;
      y2f = (float)rng;
      for (int s=0;s<scales.size();s++){
          x1 = x1f * scales[s].width;
          y1 = y1f * scales[s].height;
          x2 = x2f * scales[s].width;
          y2 = y2f * scales[s].height;
          features[s][i] = Feature(x1, y1, x2, y2); //第s种尺度的第i个特征  两个随机分配的像素点坐标  
      }
  }
  // 2. Thresholds， 负样本的阈值
  thrN = 0.5*nstructs;
  // 3. Initialize Posteriors， 初始化后验概率   
  //后验概率指每一个分类器对传入的图像片进行像素对比，每一个像素对比得到0或者1，所有的特征13个comparison对比，  
  //连成一个13位的二进制代码x，然后索引到一个记录了后验概率的数组P(y|x)，y为0或者1（二分类），也就是出现x的  
  //基础上，该图像片为y的概率是多少对n个基本分类器的后验概率做平均，大于0.5则判定其含有目标 
  for (int i = 0; i<nstructs; i++) {
	  //每一个每类器维护一个后验概率的分布，这个分布有2^d个条目（entries），这里d是像素比较pixel comparisons  
	  //的个数，这里是structSize，即13个comparison，所以会产生2^13即8,192个可能的code，每一个code对应一个后验概率  
	  //后验概率P(y|x)= #p/(#p+#n) ,#p和#n分别是正和负图像片的数目，也就是下面的pCounter和nCounter  
	  //初始化时，每个后验概率都得初始化为0；运行时候以下面方式更新：已知类别标签的样本（训练样本）通过n个分类器  
	  //进行分类，如果分类结果错误，那么响应的#p和#n就会更新，这样P(y|x)也相应更新了
      posteriors.push_back(vector<float>(pow(2.0,structSize), 0));
      pCounter.push_back(vector<int>(pow(2.0,structSize), 0));
      nCounter.push_back(vector<int>(pow(2.0,structSize), 0));
  }
}
/*
* In
*	-image:		path
*	-scale_idx:	
*   -features   
* Out
*	-fern:	nstructs个整数组成的向量，每个整数都是structSize（13）位，怎么产生的呢？13对点的特征值的对比结果（0/1）
*			对比点都记录在features里面，是FerNNClassifier::prepare阶段随机产生
*/
//该函数得到输入的image的用于树的节点，也就是特征组的特征（13位的二进制代码）  
void FerNNClassifier::getFeatures(const cv::Mat& image,const int& scale_idx, vector<int>& fern){
  int leaf;
  //每一个每类器维护一个后验概率的分布，这个分布有2^d个条目（entries），这里d是像素比较pixel comparisons  
  //的个数，这里是structSize，即13个comparison，所以会产生2^13即8,192个可能的code，每一个code对应一个后验概率  
  for (int t=0;t<nstructs;t++){//nstructs 表示树的个数 10
      leaf=0;//叶子  树的最终节点
      for (int f=0; f<structSize; f++){
		  //依次得到每一位
		  //返回的patch图像片在(y1,x1)和(y2, x2)点的像素比较值，返回0或者1  
		  //然后leaf就记录了这13位的二进制代码，作为特征   
          leaf = (leaf << 1) + features[scale_idx][t*nstructs+f](image);//运算符重载，第一个点大于第二个点返回1，否则为0
      }
      fern[t]=leaf;
  }
}
// 概率和
float FerNNClassifier::measure_forest(vector<int> fern) {
  float votes = 0;
   // 后验概率posteriors[i][idx] = ((float)(pCounter[i][idx]))/(pCounter[i][idx] + nCounter[i][idx]);  
  for (int i = 0; i < nstructs; i++) {
      votes += posteriors[i][fern[i]];//每棵树的每个特征值对应的后验概率累加值 作投票
  }
  return votes;
}
// 跟新正负样本的直方图分布，注意：posteriors只算了正样本的概率
//更新正负样本数，同时更新后验概率   
void FerNNClassifier::update(const vector<int>& fern, int C, int N) {
  int idx;
  for (int i = 0; i < nstructs; i++) {//10
      idx = fern[i];//13位的特征
	  //直接用idx作为一个bin，sizeof(int)好奢侈呀
	  // 问题： 如果样本不是特别大，那么特征分布就会特别稀疏，如此，怎能保证统计的有效性？？
	  // 问题： 酱紫的直方图统计，不平滑性也太大了吧？？？
	  //C=1，正样本，C=0，负样本
      (C==1) ? pCounter[i][idx] += N : nCounter[i][idx] += N;
      if (pCounter[i][idx]==0) {//既然是正概率，如果正样本的数目为0，正样本的概率自然也为0
          posteriors[i][idx] = 0;
      } else {
          posteriors[i][idx] = ((float)(pCounter[i][idx]))/(pCounter[i][idx] + nCounter[i][idx]);
      }
  }
}
/*
* Fun
*	bootstrap训练随机森林分类器,更新样本分布和概率：容易分错的正样本：支持率（measure_forest）<=thrP（六成)；容易分错的负样本 支持率>=五成
*	-nCounter ->posteriors
*   -pCounter ->posteriors
* In
*	- ferns：	打乱过顺序的正负样本的特征
*	- resample：	bootstrap次数，实际上程序里面没有bootstrap……
*/
//问题：对于能正确分类的样本，就不更新分布，哪么何来的原始分布呀？难倒posteriors已经初始化过了吗？
//其实，对于正样本，如果开始没有正样本的话，那么posteriors应该都是0，所以满足更新条件
//不过，measure_forest(ferns[i].first)>thrP就不更新
// 那么，训练时样本的顺序就不能前面都是正样本，后面都是负样本，反之亦然，必须打乱顺序
//训练集合分类器（n个基本分类器集合）
//主要功能是：对每一个样本ferns_data[i] ，如果样本是正样本标签，先用measure_forest函数，
//找出该样本box中所有树的所有特征值，对应的后验概率累加值，该累加值如果小于正样本阈值（0.6* nstructs，
//0.6这个经验值是Fern分类器的阈值，，初始化时从myParam.yml中读取，后面会用测试集来评估修改，
//找到最优），也就是输入的是正样本，却被分类成负样本了，出现了分类错误，所以就把该样本添加到正样本库使pNum=pNum+1，
//同时用update函数更新后验概率。对于负样本，同样，如果出现负样本分类错误，就添加到负样本库使nNum=nNum+1。
//update函数有三个参数，第一个参数是该box对应的10棵树fern[]，第二个参数要进行更新的是正样本库还是负样本库，1表示更新正样本库的数目，
//0表示更新负样本库的数目，第三个参数表示要更新的数目，在整个程序中所有的调用该值都是取1，也即每次都是对样本库数目增加1(为何另一边的不用相应地减小1？)，
//这也跟上面提到的分错的样本放到对应的库是同样的意思，因为每次只能判断一个样本是否有分错，所以更新的数目也只能是1。
//而在更新数目的同时，也更新了后验概率值，按照post=pNum/nNum的式子来更新
void FerNNClassifier::trainF(const vector<std::pair<vector<int>,int> >& ferns,int resample){
  thrP = thr_fern*nstructs; //0.6*10                           // int step = numX / 10;
  //for (int j = 0; j < resample; j++) {                      // for (int j = 0; j < bootstrap; j++) {
      for (int i = 0; i < ferns.size(); i++){               //   for (int i = 0; i < step; i++) {
                                                            //     for (int k = 0; k < 10; k++) {
                                                            //       int I = k*step + i;//box index
                                                            //       double *x = X+nTREES*I; //tree index
          if(ferns[i].second==1){ //正样本  //为1表示正样本   //       if (Y[I] == 1) {
			  //measure_forest函数返回所有树的所有特征值对应的后验概率累加值  
			  //该累加值如果小于正样本阈值，也就是是输入的是正样本，却被分类成负样本了  
			  //出现分类错误，所以就把该样本添加到正样本库，同时更新后验概率  
			  if(measure_forest(ferns[i].first)<=thrP)      //         if (measure_forest(x) <= thrP)
                update(ferns[i].first,1,1);                 //             update(x,1,1);
			  ////更新正样本数，同时更新后验概率
          }else{//负样本                                            //        }else{
              if (measure_forest(ferns[i].first) >= thrN)   //         if (measure_forest(x) >= thrN)
                update(ferns[i].first,0,1);                 //             update(x,0,1);
          }
      }
	
  //}
}
/*
* Fun: 训练最费时的，NN分类器
* In:
*	- nn_examples:[+][-][-].... 只有第一个是正样本
* Out:
*	nEx <-- nn_examples的负样本中满足 Relative similarity  >0.5
*	pEx <-- nn_examples的正样本中满足 Relative similarity  <thr_nn(0.65)
* 注意： 是不断地增加 nEx 和 pEx
*/
//问题：正负样本的出场顺序会影响到最终的分类器，不喜欢随机！！！！
//训练最近邻分类器。对每一个样本nn_data，如果标签是正样本，通过NNConf(nn_examples[i], isin, conf, dummy)计算输入图像片与在线模型之间的相关相似度conf，
//如果相关相似度小于0.65 ，则认为其不含有前景目标，也就是分类错误了，这时候就把它加到正样本库。
//然后就通过pEx.push_back(nn_examples[i]);将该样本添加到pEx正样本库中；
//同样，如果出现负样本分类错误，就添加到负样本库
void FerNNClassifier::trainNN(const vector<cv::Mat>& nn_examples){
  float conf,dummy;
  vector<int> y(nn_examples.size(),0);
  y[0]=1;//只有第一个是正样本，并不是原始的目标区域，而是best_box，//上面说到调用trainNN这个函数传入的nn_data样本集，只有一个pEx，在nn_data[0]  
  vector<int> isin;
  for (int i=0;i<nn_examples.size();i++){//  For each example
      NNConf(nn_examples[i],isin,conf,dummy);//  Measure Relative similarity//计算输入图像片与在线模型之间的相关相似度conf  
	  //thr_nn: 0.65 阈值   
	  //标签是正样本，如果相关相似度小于0.65 ，则认为其不含有前景目标，也就是分类错误了；这时候就把它加到正样本库  
	  if (y[i]==1 && conf<=thr_nn){//    if y(i) == 1 && conf1 <= tld.model.thr_nn % 0.65
          if (isin[1]<0){ //注意：如果pEx为空，NNConf 直接返回 thr_nn=0，isin都为-1，                                         //      if isnan(isin(2))
              pEx = vector<Mat>(1,nn_examples[i]);                 //        tld.pex = x(:,i);
              continue;                                            //        continue;
          }                                                        //      end
          //pEx.insert(pEx.begin()+isin[1],nn_examples[i]);        //      tld.pex = [tld.pex(:,1:isin(2)) x(:,i) tld.pex(:,isin(2)+1:end)]; % add to model
          pEx.push_back(nn_examples[i]);//之前存在正样本，追加
      }                                                            //    end
      if(y[i]==0 && conf>0.5)                                      //  if y(i) == 0 && conf1 > 0.5
        nEx.push_back(nn_examples[i]);                             //    tld.nex = [tld.nex x(:,i)];

  }                                                            
  acum++;
  printf("%d. Trained NN examples: %d positive %d negative\n",acum,(int)pEx.size(),(int)nEx.size());
}                                                               

/*Inputs:
* -NN Patch
* Outputs:
* -Relative Similarity (rsconf), Conservative Similarity (csconf), In pos. set|Id pos set|In neg. set (isin)
* isin[3]：isin[0]:is pos ? isin[1]：是+的，那么记录最接近的正样本的Id；isin[2]：is neg?
//isin中存放三个int型值，初始化全为-1。第一个如果取值为1，则表示NNConf()在计算输入图像片patch与在线模型pEx中的box时发现在线模型中有一个与其相似度超过阈值ncc_thesame (固定值0.95，从myParam.yml中读取)，
//此时会把这个patch也放到在线模型的pEx中，所以第一个取值为1就表示已经把当前输入图像片patch放到pEx中。
//第二个的取值依赖于第1个的取值，如果第一个取值为-1，那么第二个的取值就是-1，如果第一个的取值是1，
//那么第二个的取值就是在遍历在线模型时找到的第一个与输入图像片patch相似度超过ncc_the same的box的索引。
//第三个意义与第一个接近，不同的地方只在于第一个是对应在线模型的正样本近邻数据集pEx，第三个是对应在线模型的负样本近邻数据集nEx。
*/
//对于输入的图像片patch，先遍历在线模型中的正样本近邻数据集pEx中的box(第一次其实就是best_box，后面会在线更新)，
//调用matchTemplate()计算匹配度ncc，再由ncc得到相似度nccP，并找出ncc中的最大者maxP；
//同样的方式也遍历在线模型中的负样本近邻数据集nEx中的box来找出maxN
void FerNNClassifier::NNConf(const Mat& example, vector<int>& isin,float& rsconf,float& csconf){
  isin=vector<int>(3,-1);
  if (pEx.empty()){ //if isempty(tld.pex) % IF positive examples in the model are not defined THEN everything is negative
      rsconf = 0; //    conf1 = zeros(1,size(x,2));
      csconf=0;
      return;
  }
  if (nEx.empty()){ //if isempty(tld.nex) % IF negative examples in the model are not defined THEN everything is positive
      rsconf = 1;   //    conf1 = ones(1,size(x,2));
      csconf=1;
      return;
  }
  Mat ncc(1,1,CV_32F);
  float nccP,csmaxP,maxP=0;
  bool anyP=false;
  int maxPidx,validatedPart = ceil(pEx.size()*valid);//正样本的前 50%，用于计算Conservative similarit【5.2 5】
  float nccN, maxN=0;
  bool anyN=false;
  //比较图像片p到在线模型M的距离（相似度），计算正样本最近邻相似度，也就是将输入的图像片与  
  //在线模型中所有的图像片进行匹配，找出最相似的那个图像片，也就是相似度的最大值  
  for (int i=0;i<pEx.size();i++){
      matchTemplate(pEx[i],example,ncc,CV_TM_CCORR_NORMED);// measure NCC to positive examples
	  //相关系数的取值范围是[-1,1]，加上1变成[0,2]，再将范围缩小为[0,1]
      nccP=(((float*)ncc.data)[0]+1)*0.5;//计算匹配相似度 
      if (nccP>ncc_thesame)//0.95
        anyP=true;
      if(nccP > maxP){
          maxP=nccP;//Relative similarity //记录最大的相似度以及对应的图像片index索引值 
          maxPidx = i;
          if(i<validatedPart)
            csmaxP=maxP;//Conservative similari
      }
  }
  //计算负样本最近邻相似度   
  for (int i=0;i<nEx.size();i++){
      matchTemplate(nEx[i],example,ncc,CV_TM_CCORR_NORMED);//measure NCC to negative examples
      nccN=(((float*)ncc.data)[0]+1)*0.5;
      if (nccN>ncc_thesame)
        anyN=true;
      if(nccN > maxN)
        maxN=nccN;
  }
  //set isin
  if (anyP) isin[0]=1;  //if he query patch is highly correlated with any positive patch in the model then it is considered to be one of them
  isin[1]=maxPidx;      //get the index of the maximall correlated positive patch
  if (anyN) isin[2]=1;  //if  the query patch is highly correlated with any negative patch in the model then it is considered to be one of them
  //Measure Relative Similarity
  //相关相似度 = 正样本最近邻相似度 / （正样本最近邻相似度 + 负样本最近邻相似度）  
  //接下来会计算1-maxP和1-maxN，个人理解为：正样本不相似的最小程度、负样本不相似的最小程度。
  //然后就计算相关相似度rsconf=(1-maxN)/[(1-maxP)+(1-maxN)]，也即正负样本不相似的最小情况下，负样本不相似所占比例就定义为相关相似度，
  //如果负样本不相似所占比重越大，那么该patchpatch与负样本不相似的可能性越大，从而相关相似度越高(好像有点拗口，个人是从对偶问题的角度理解的)
  float dN=1-maxN;
  float dP=1-maxP;
  // 关于保守相似度csconf是这样得到的，在pEx的前半部分数据中，如果有对maxP进行更新，那么把此时的maxP放到csmaxP中，
  //也即csmaxP记录正样本近邻数据集pEx的前半部分数据中与输入图像片的最大相似度，
  //然后csconf=(1-maxN)/[(1-maxN)+(1-csmaxP)]，由于csmaxP不超过maxP，所以csconf不超过rsconf，
  //也即在认为当前输入图像片patch与正样本近邻数据集数据相似的问题上，对同个patch，
  //rsconf的度量更为“保守”。在第一个进行训练的时候，是否保守没有多大意义，
  //所以在init()中第一次进行trainNN()时，会将rsconf所得到的值丢弃
  rsconf = (float)dN/(dN+dP);//与原文【5.2】有出入，不过也是可以理解的
  //Measure Conservative Similarity
  dP = 1 - csmaxP;
  csconf =(float)dN / (dN + dP);
}
//  evaluateTh
//	输入：
// 	负样本随机蕨测试集：nXT
// 	负样本模板测试集：nExT
// 	输出：
// 	更新后的随机蕨阈值：thr_fern
// 	更新后的模板NCC阈值：thr_nn
// 	更新后的有效性判断(P/N学习)阈值：thr_nn_valid
// 	描述：
// 	对所有负样本随机蕨测试集，计算其随机蕨可信度，若最大值大于预设阈值
// 	thr_fern，则用该最大值替换更新；对所有负样本模板测试集，计算其NCC系数，
// 	若最大值大于预设阈值thr_nn，则用该最大值替换更新；若最后更新得到的thr_nn
// 	大于预设thr_nn_valid，则替换更新thr_nn_valid
void FerNNClassifier::evaluateTh(const vector<pair<vector<int>,int> >& nXT,const vector<cv::Mat>& nExT){
float fconf;
// 更新thr_fern：对nXT中的每个样本，用measure_forest()函数找出10棵树对这个box的01码，由01码找到对应的后验概率，将10个后验概率累加后求平均，将平均值与thr_fern(初始值0.6，从myParam.yml中获取)比较，
//如果超过thr_fern则将thr_fern更新为这个平均值。也即对于nXT中所有的box，10棵树都会对其投票得到一个后验概率，
//将所有后验概率取平均后，比较所有的box，取后验概率均值最大的那个box所对应的平均值放到thr_fern中
  for (int i=0;i<nXT.size();i++){
	  //所有基本分类器的后验概率的平均值如果大于thr_fern，则认为含有前景目标  
	  //measure_forest返回的是所有后验概率的累加和，nstructs 为树的个数，也就是基本分类器的数目 
    fconf = (float) measure_forest(nXT[i].first)/nstructs;//平均
    if (fconf>thr_fern)//0.6thrP定义为Positive thershold
      thr_fern=fconf;//修正初始值//取这个平均值作为 该集合分类器的 新的阈值，这就是训练？？ 
}
  //更新thr_nn：对于负样本近邻数据nExT测试集中的每个样本，用NNConf()函数计算它与在线模型pEx和nEx中数据的相似度来得到相关相似度conf
  //(与第一次训练NN分类器一样，这里得到的保守相似度也被丢弃了)，如果conf大于阈值thr_nn(初始值0.65，从myParam.yml中获取)，
  //则更新thr_nn为conf
  vector <int> isin;
  float conf,dummy;
  for (int i=0;i<nExT.size();i++){
      NNConf(nExT[i],isin,conf,dummy);
      if (conf>thr_nn)
        thr_nn=conf; //取这个最大相关相似度作为 该最近邻分类器的 新的阈值，这就是训练？？  
  }
  //更新thr_nn_valid：如果更新后的thr_nn大于thr_nn_valid(初始值0.7，从myParam.yml中获取)，那么更新thr_nn_valid值为thr_nn
  if (thr_nn>thr_nn_valid)//我们的初始值可能不够严格，于是检测是否要提高最近邻分类器的阈值
    thr_nn_valid = thr_nn;
}
//把正样本库（在线模型）包含的所有正样本显示在窗口上   
void FerNNClassifier::show(){
  Mat examples((int)pEx.size()*pEx[0].rows,pEx[0].cols,CV_8U);
  double minval;
  Mat ex(pEx[0].rows,pEx[0].cols,pEx[0].type());
  for (int i=0;i<pEx.size();i++){
    minMaxLoc(pEx[i],&minval);//寻找pEx[i]的最小值 
    pEx[i].copyTo(ex);
    ex = ex-minval; //把像素亮度最小的像素重设为0，其他像素按此重设  
	//Mat Mat::rowRange(int startrow, int endrow) const 为指定的行span创建一个新的矩阵头。  
	//Mat Mat::rowRange(const Range& r) const   //Range 结构包含着起始和终止的索引值。  
    Mat tmp = examples.rowRange(Range(i*pEx[i].rows,(i+1)*pEx[i].rows));
    ex.convertTo(tmp,CV_8U);
  }
  imshow("Examples",examples);
}
