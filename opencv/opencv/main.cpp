#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <fstream>
using namespace cv;
using namespace std;

void drawHist(Mat& hist, int type,string name)
{
	// 创建直方图绘制窗口
	int histWidth = 512;
	int histHeight = 400;
	cv::Mat histImage(histHeight, histWidth, CV_8UC3, cv::Scalar(0, 0, 0));
	// 对直方图进行归一化
	cv::normalize(hist, hist, 1, 0, type, -1, cv::Mat());
	// 绘制直方图
	int binWidth = 2;
	for (int i = 1; i <= hist.rows; i++)
	{
		cv::line(histImage, cv::Point(binWidth * (i-1), histHeight-1), cv::Point(binWidth * i-1, histHeight - cvRound(20*histHeight*hist.at<float>(i-1))-1), cv::Scalar(255, 255, 255));
	}
	// 显示直方图
	cv::imshow(name, histImage);
}

void addsalt(Mat& image, int n) {
	for (int k = 0; k < n; k++)
	{
		int j = rand() % image.rows;
		int i = rand() % image.cols;
		int color = rand() % 2;
		if (color == 0)  //waite
		{
			if (image.type() == CV_8U)
			{
				image.at<uchar>(j, i) = 255;
			}
			else if(image.type() == CV_8UC3)
			{
				image.at<Vec3b>(j, i) = Vec3b(255, 255, 255);
			}
		}
		else
		{
			if (image.type() == CV_8U)
			{
				image.at<uchar>(j, i) = 0;
			}
			else if (image.type() == CV_8UC3)
			{
				image.at<Vec3b>(j, i) = Vec3b(0, 0, 0);
			}
		}
	}
}

void ord_feature(Mat& img,vector<KeyPoint>& kps,Mat& des) {
	Ptr<ORB>orb = ORB::create();
	orb->detect(img, kps);
	orb->compute(img, kps, des);
}

void match_min(vector<DMatch>& dm, vector<DMatch>& gdm) {
	double maxdis = 0, mindis = 10000;
	for (int i = 0; i < dm.size(); i++)
	{
		double dis = dm[i].distance;
		if (dis > maxdis) maxdis = dis;
		if (dis < mindis) mindis = dis;
	}
	cout << maxdis << mindis << endl;
	for (int i = 0; i < dm.size(); i++)
	{
		double dis = dm[i].distance;
		if (dis <= max(2 * mindis, 20.0)) {
			gdm.push_back(dm[i]);
		}
	}
}

void ransac(vector<DMatch> gdm, vector<KeyPoint> kps1, vector<KeyPoint> kps2, vector<DMatch>& gsc) {
	vector<Point2f> srcp(gdm.size()), dstp(gdm.size());
	for (int i = 0; i < gdm.size(); i++)
	{
		srcp[i] = kps1[gdm[i].queryIdx].pt;
		dstp[i] = kps2[gdm[i].trainIdx].pt;
		vector<int> inliersMask(srcp.size());
		findHomography(srcp, dstp,RANSAC,5,inliersMask);
		for (int i = 0; i < inliersMask.size(); i++)
		{
			if (inliersMask[i])
				gsc.push_back(gdm[i]);
		}
	}
}

int main(int argc, char** agrv) {
	//Mat imgg = imread("chess0.jpg");
	Mat img = imread("C:/Users/wfs/Pictures/123456.png");
	Mat socer = imread("C:/Users/wfs/Pictures/socear.png");
	Mat socear = imread("C:\\Users\\wfs\\Desktop\\opencv\\opencv\\opencv\\socear.png",IMREAD_GRAYSCALE);
	//Mat book = imread("C:/Users/wfs/Pictures/更多.png");
	//Mat mask = imread("C:/Users/wfs/Desktop/opencv/opencv/opencv/OR_MASK.jpg");
	Mat lena = imread("lena.png");
	Mat lenahead = imread("lenahead.png");
	Mat lenagray = imread("lenagray.jpg");
	Mat lenagraysalt = imread("lenagraysalt.png");
	Mat lenagraynoise = imread("lenagraynoise.png");
	cvtColor(lenahead, lenahead, COLOR_BGR2GRAY);
	cvtColor(lenagray, lenagray, COLOR_BGR2GRAY);
	cvtColor(img, img, COLOR_BGR2GRAY);
	equalizeHist(socear, socear); //直方图均衡化 ，只能单通道

	//Mat socear = socer(Range(400,1200), Range(242,1035));
	//imwrite("socear.png", socear);

	//img.convertTo(img,CV_8U,(0,255));
	//convertScaleAbs(img, img, 1.0 / 256.0, 0.0);

	//数据类型

	//CV_8U: 8 位无符号整数，占用 1 个字节。
	//CV_8S: 8 位有符号整数，占用 1 个字节。
	//CV_16U : 16 位无符号整数，占用 2 个字节。
	//CV_16S : 16 位有符号整数，占用 2 个字节。
	//CV_32S : 32 位有符号整数，占用 4 个字节。
	//CV_32F : 32 位浮点数，占用 4 个字节。
	//CV_64F : 64 位浮点数，占用 8 个字节。

	//数据读取

	//Mat A = (Mat_<int>(3, 3) << 1, 2, 5, 4, 5, 6, 7, 8, 9);
	//Mat B1 = Mat(5, 5, CV_8UC1,Scalar(4,5,6));
	//Mat B2 = Mat(5, 5, CV_8UC2, Scalar(4, 5, 6));
	//Mat B3 = Mat(5, 5, CV_8UC3, Scalar(4, 5, 6));
	//cout << A.at<int>(0, 0) << endl;
	//Vec2b vc = B2.at<Vec2b>(0, 0);
	//cout << B2 << endl;
	//cout << vc << endl;
	//cout << (int)vc.val[0] << endl;
	//cout << B3 << endl;
	//返回的是每一行，列的字节数
	//cout <<  (int)(*(B3.data + B3.step[0] * 1 + B3.step[1] * 2 + 2)) << endl;
	//cout << (int)(*(B3.data + B3.step[0] * 1 + B3.step[1] * 2 + 1)) << endl;
	//cout << (int)(*(B3.data + B3.step[0] * 1 + B3.step[1] * 2 + 0)) << endl;

	//Mat img;
	//cout << A << endl;
	//picture read and show
	//img = imread("c:/users/wfs/pictures/screenshot 2022-08-07 225529.png");
	//if (img.empty()) {
	//	cout << "no img" << endl;
	//	return -1;
	//}
	//imshow("sdz", A);
	// im.write();
	//waitKey(0);

	//mat运算
	//*相乘  ，dot内积 ，mul对应位相乘

	//摄像头
	//Mat img;
	//VideoCapture vc(0);
	////vc >> img;
	////bool iscolor = (img.type() == CV_8UC3);
	////VideoWriter vw;
	////int codeff = VideoWriter::fourcc('m', 'p', '4', 'v');
	////vw.open("live.mp4", codeff, 30, img.size(), iscolor);
	//while (1)
	//{
	//	vc.read(img);
	//	imshow("live", img);
	//	int fps = vc.get(CAP_PROP_FPS);
	//	cout << fps << endl;
	//	int width = vc.get(CAP_PROP_FRAME_WIDTH);
	//	int hight = vc.get(CAP_PROP_FRAME_HEIGHT);
	//	putText(img, to_string(fps), Point(50, 50), FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 3);
	//	waitKey(50);
	//}

	////图像数据类型转换  RGB,HSV（色度，饱和度，亮度 ） GRAY=R*0.3+G*0.59+B*0.11
	
	////8U :0-255,32F:0-1,d64f:0-1      >1为白色 ，<1为黑色
	//Mat img32,HSV,HSV_32,gray,gray_32;
	//img.convertTo(img32, CV_32F, 1 / 255.0, 0);
	//cvtColor(img, HSV, COLOR_BGR2HSV);
	//cvtColor(img32, HSV_32, COLOR_BGR2HSV);
	//cvtColor(img, gray, COLOR_BGR2GRAY);
	//cvtColor(img32, gray_32, COLOR_BGR2GRAY);
	////split
	//Mat imgs[3],img1,img2,img3;
	//split(img, imgs);
	//img1 = imgs[0];
	//img2 = imgs[1];
	//img3 = imgs[2];
	////merge
	//Mat zero = Mat::zeros(img.size(), CV_8UC1);
	//Mat imgsb[3];
	//imgsb[0] = img1;
	//imgsb[1] = zero;
	//imgsb[2] = zero;
	//Mat blue;
	//merge(imgsb, 3,blue);
	//vector<Mat> imgsv;
	//imgsv.push_back(zero);
	//imgsv.push_back(img2);
	//imgsv.push_back(zero);
	//Mat green;
	//merge(imgsv, green);

	////图像像素比较 需要大小，通道数，数据类型一样
	
	////min max（src1，src2，out）
	//Mat Min, Max;
	//min(socer, book, Min);
	////minMaxLoc  返回最大最小像素值及其坐标，可以有掩码矩阵

	////与 或 非 异或  0-255化为二进制，对每位进行运算   0-11111111
	
	//Mat AND, OR, NOT, XOR;
	//bitwise_not(img, NOT, mask);
	//bitwise_and(img, NOT, AND, mask);
	//bitwise_or(img, NOT, OR, mask);
	//bitwise_xor(img, NOT, XOR, mask);
	////for (int i = 0; i < img.rows / 2; i++)
	////{
	////	for (int j = 0; j < 20; j++)
	////	{
	////		OR.at<Vec3b>(i, i+j) = Vec3b(0, 0, 0);
	////	}
	////}
	////for (int i = 0; i < img.rows / 2; i++)
	////{
	////	for (int j = img.cols; j >= img.cols - 20; j--) {
	////		if (j - i >= 0 && j - i < img.cols) {
	////			OR.at<cv::Vec3b>(i, j - i) = cv::Vec3b(0, 0, 0);
	////		}
	////	}
	////}
	////Mat ORG;
	////cvtColor(OR, ORG, COLOR_BGR2GRAY);
	////imwrite("OR_MASK.png", ORG);

	////阈值二值化 5种条件
	
	//Mat gimg;
	//threshold(img, gimg, 127, 255, THRESH_BINARY);
	//cvtColor(img, img, COLOR_BGR2GRAY);
	//adaptiveThreshold(img, img, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 51, 0);

//	//LUT查找表

//uchar Lutfirst[256];
//for (int i = 0; i < 256; i++)
//{
//	if (i < 100)
//		Lutfirst[i] = 0;
//	if (i >= 100 && i <= 200)
//		Lutfirst[i] = 100;
//	if (i > 200)
//		Lutfirst[i] = 255;
//}
//Mat lutone(1, 256, CV_8UC1, Lutfirst);
//LUT(img, lutone, img);

////图像缩放，翻转，拼接

//resize(img, img, Size(512, 512), 0, 0, INTER_AREA); //用size就不用轴参数了
//resize(img, img, Size(1682, 1392), 0, 0, INTER_NEAREST);
//resize(img, img, Size(1682, 1392), 0, 0, INTER_LINEAR);
//resize(img, img, Size(1682, 1392), 0, 0, INTER_CUBIC);
//flip(img, img, 0);
//hconcat(img, img, img);
//vconcat(img, img, img);

////图片旋转，仿射变换

//Mat rotation_matrix1 = getRotationMatrix2D(Point2f(800.0, 700.0), 60, 1);
//warpAffine(img, img, rotation_matrix1, img.size());
//Point2f src[3];
//Point2f dst[3];
//src[0] = Point2f(0, 0);
//src[1] = Point2f(0, img.cols);
//src[2] = Point2f(img.rows, 0);
//dst[0] = Point2f((img.rows)*0.1, (img.cols)*0.2);
//dst[1] = Point2f((img.rows)*0.3, (img.cols)*0.8);
//dst[2] = Point2f((img.rows)*0.9, (img.cols)*0.1);
//Mat rotation_matrix2 = getAffineTransform(src,dst);
//warpAffine(img, img, rotation_matrix2, img.size());

//透视变换（四点变换）
//Mat persp_matrix = getPerspectiveTransform(Point2f src, Point2f dst);
//warpPerspective(img, img, persp_matrix, img.size());

//作图
//circle line rectangle fillPolv putText

////ROI截取

//Mat imgr = img(Range(100, 1000), Range(200, 1500));
//Rect rec(100,100,500,500);
//Mat imgrec = img(rec);
//Mat copy;
//img.copyTo(copy,mask);

////高斯金字塔 下采样（缩小尺寸）；缩放不变性（特征值）

////pyrDown(img, img);
//vector<Mat> Guass;
//Guass.push_back(img);
//for (int i = 0; i < 3; i++)
//{
//	Mat guass;
//	pyrDown(Guass[i], guass);
//	Guass.push_back(guass);
//}
//for (int i = 0; i < 3; i++)
//{
//	imshow(to_string(i), Guass[i]);
//}
//waitKey(0);

////拉普拉斯金字塔 上下采样结合

//vector<Mat> Guass;
//Guass.push_back(socer);
//for (int i = 0; i < 3; i++)
//{
//	Mat guass;
//	pyrDown(Guass[i], guass);
//	Guass.push_back(guass);
//}
//vector<Mat> Lap;
//for (int i = Guass.size()-1; i > 0 ; i--)
//{
//	Mat lap, upsample;
//	if (i==Guass.size()-1)
//	{
//		Mat down;
//		pyrDown(Guass[i], down);
//		pyrUp(down, upsample);
//		cout<<Guass[i].size()<<endl;
//		cout << upsample.size() << endl;
//		lap = Guass[i] - upsample;
//		Lap.push_back(lap);
//	}
//	pyrUp(Guass[i], upsample);
//	lap = Guass[i-1] - upsample;
//	Lap.push_back(lap);
//}
//for (int i = 0; i < Lap.size(); i++)
//{
//	imshow(to_string(i), Lap[i]);
//}
//waitKey(0);

////统计直方图

//// 定义直方图参数
//int histSize = 256;    // 直方图的bin数目
//float range[] = { 0, 256 };    // 像素值范围
//const float* histRange = { range };
//// 计算直方图
//cv::Mat hist;
//cv::calcHist(&socear, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
//// 创建直方图绘制窗口
//int histWidth = 512;
//int histHeight = 400;
//cv::Mat histImage(histHeight, histWidth, CV_8UC3, cv::Scalar(0, 0, 0));
//// 对直方图进行归一化
//cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
//// 绘制直方图
//int binWidth = cvRound((double)histWidth / histSize);
//for (int i = 0; i < histSize; i++)
//{
//	cv::line(histImage, cv::Point(binWidth * i, histHeight), cv::Point(binWidth * i, histHeight - cvRound(hist.at<float>(i))), cv::Scalar(255, 255, 255));
//}
//// 显示直方图
//imshow("scoear", socear);
//cv::imshow("Histogram", histImage);
//cv::waitKey(0);

////直方图匹配     通过原与目标直方图之间各像素之间的累积概率之差的最小值来确定像素映射关系

////构建差值矩阵，找每一行最小值确定Lut映射关系，Lut匹配直方图。
//int histSize = 256;    // 直方图的bin数目
//float range[] = { 0, 256 };    // 像素值范围
//const float* histRange = { range };
//// 计算直方图
//cv::Mat hist1,hist2;
//cv::calcHist(&img, 1, 0, cv::Mat(), hist1, 1, &histSize, &histRange);
//cv::calcHist(&socear, 1, 0, cv::Mat(), hist2, 1, &histSize, &histRange);
//drawHist(hist1, NORM_L1, "img");
//drawHist(hist2, NORM_L1, "soc");
//float hist1_cdf[256] = { hist1.at<float>(0) };
//float hist2_cdf[256] = { hist2.at<float>(0) };
//for (int i = 1; i < 256; i++)
//{
//	hist1_cdf[i] = hist1_cdf[i - 1] + hist1.at<float>(i);
//	hist2_cdf[i] = hist2_cdf[i - 1] + hist2.at<float>(i);
//}
//float diffcdfmatrix[256][256];
//for (int i = 0; i < 256; i++)
//{
//	for (int j = 0; j < 256; j++) {
//		diffcdfmatrix[i][j] = fabs(hist1_cdf[i] - hist2_cdf[j]);
//	}
//}
//Mat lut(1, 256, CV_8U);
//for (int i = 0; i < 256; i++)
//{
//	float min = diffcdfmatrix[i][0];
//	int index = 0;
//	for (int j = 1; j < 256; j++)
//	{
//		if (min> diffcdfmatrix[i][j])
//		{
//			min = diffcdfmatrix[i][j];
//			index = j;
//		}
//	}
//	lut.at<uchar>(i) = (uchar)index;
//}
//Mat res;
//LUT(socear, lut, res);
//imshow("socear", res);
//waitKey(0);

//图像模板匹配

//Mat result;
//matchTemplate(lena,lenahead,result, TM_CCOEFF_NORMED);
//double maxVal, minVal;
//Point minLoc, maxLoc;
////寻找匹配结果中的最大值和最小值以及坐标位置
//minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
////绘制最佳匹配区域
//rectangle(lena, cv::Rect(maxLoc.x, maxLoc.y, lenahead.cols, lenahead.rows), Scalar(0, 0, 255), 2);
//imshow("lena", lena);
//imshow("res", result);
//waitKey(0);

//图像卷积

//Mat kernel = (Mat_<float>(3, 3) << 1, 2, 1, 2, 0, 2, 1, 2, 1);
//Mat kernel_norm = kernel / 12;
//Mat filter;
//filter2D(lena, filter, -1,kernel_norm);
//imshow("fil",filter);

//图像噪声

//addsalt(lena, 10000);
//addsalt(lenagray, 10000);
//imshow("lena", lena);
//imshow("lenagray", lenagray);
//imwrite("lenasalt.png", lena);
//imwrite("lenagray.png",lenagray);
//Mat lena_noise = Mat::zeros(lena.rows, lena.cols, lena.type());
//Mat equalLena_noise = Mat::zeros(lena.rows, lena.cols, lenagray.type());
//imshow("lena原图", lena);
//imshow("equalLena原图", lenagray);
//RNG rng; //创建一个RNG类
//rng.fill(lena_noise, RNG::NORMAL, 10, 20); //生成三通道的高斯分布随机数
//rng.fill(equalLena_noise, RNG::NORMAL, 15, 30); //生成三通道的高斯分布随机数
//imshow("三通道高斯噪声", lena_noise);
//imshow("单通道高斯噪声", equalLena_noise);
//lena = lena + lena_noise; //在彩色图像中添加高斯噪声
//lenagray = lenagray + equalLena_noise; //在灰度图像中添加高斯噪声
////显示添加高斯噪声后的图像
//imshow("lena添加噪声", lena);
//imshow("equalLena添加噪声", lenagray);
//imwrite("lenanoise.png", lena);
//imwrite("lenagraynoise.png", lenagray);


//线性滤波    高斯噪声用高斯滤波更号

////均值滤波
//Mat res,res1;
//blur(lenagraysalt, res, Size(3, 3));
//blur(lenagraysalt, res1, Size(9, 9));
//imshow("res", res);
//imshow("res1", res1);
////方框滤波
//Mat equalLena_32F;
//lenagraysalt.convertTo(equalLena_32F, CV_32F, 1.0 / 255);
//Mat resultNorm, result, dataSqrNorm, dataSqr, equalLena_32FSqr;
////方框滤波boxFilter()和sqrBoxFilter()
//boxFilter(lenagraysalt, resultNorm, -1, Size(3, 3), Point(-1, -1), true);  //进行归一化
//boxFilter(lenagraysalt, result, -1, Size(3, 3), Point(-1, -1), false);  //不进行归一化
//sqrBoxFilter(equalLena_32F, equalLena_32FSqr, -1, Size(3, 3), Point(-1, -1),
//	true, BORDER_CONSTANT);
////显示处理结果
//imshow("resultNorm", resultNorm);
//imshow("result", result);
//imshow("equalLena_32FSqr", equalLena_32FSqr);
////高斯滤波
//Mat result_5gauss, result_9gauss;  //存放含有高斯噪声滤波结果，后面数字代表滤波器尺寸
//Mat result_5salt, result_9salt;  ////存放含有椒盐噪声滤波结果，后面数字代表滤波器尺寸
////调用均值滤波函数blur()进行滤波
//GaussianBlur(lenagraynoise, result_5gauss, Size(5, 5), 10, 20);
//GaussianBlur(lenagraynoise, result_9gauss, Size(9, 9), 10, 20);
//GaussianBlur(lenagraysalt, result_5salt, Size(5, 5), 10, 20);
//GaussianBlur(lenagraysalt, result_9salt, Size(9, 9), 10, 20);
////显示含有高斯噪声图像
//imshow("equalLena_gauss", lenagraynoise);
//imshow("result_5gauss", result_5gauss);
//imshow("result_9gauss", result_9gauss);
////显示含有椒盐噪声图像
//imshow("equalLena_salt", lenagraysalt);
//imshow("result_5salt", result_5salt);
//imshow("result_9salt", result_9salt);

//非线性滤波  中值滤波处理椒盐噪声

//Mat imgResult3, grayResult3, imgResult9, grayResult9;
////分别对含有椒盐噪声的彩色和灰度图像进行滤波，滤波模板为3×3
//medianBlur(lenasalt, imgResult3, 3);
//medianBlur(lenagraysalt, grayResult3, 3);
////加大滤波模板，图像滤波结果会变模糊
//medianBlur(lenasalt, imgResult9, 9);
//medianBlur(lenagraysalt, grayResult9, 9);
////显示滤波处理结果
//imshow("img", lenasalt);
//imshow("gray", lenagraysalt);
//imshow("imgResult3", imgResult3);
//imshow("grayResult3", grayResult3);
//imshow("imgResult9", imgResult9);
//imshow("grayResult9", grayResult9);

//可分离滤波
//X,Y两个方向分别

//边缘检测 sobel scharr算子

//Mat sobelX,sobelY,sobelXY;
//Sobel(lenagray, sobelX, CV_16S,1,0,3);
//Sobel(lenagray, sobelY, CV_16S, 0,1,3);
//convertScaleAbs(sobelX, sobelX);
//convertScaleAbs(sobelY, sobelY);
//sobelXY = sobelX + sobelY;
//Mat scX, scY, scXY;
//Scharr(lenagray, scX, CV_16S, 1, 0);
//Scharr(lenagray, scY, CV_16S, 0, 1);
//convertScaleAbs(scX, scX);
//convertScaleAbs(scY, scY);
//scXY = scX + scY;
//Mat BX, BY, CX, CY;
//getDerivKernels(BX,BY,1,0,3);
//BX = BX.reshape(CV_8U, 1);
//Mat BXY = BY * BX;
//getDerivKernels(CX, CY, 1, 0, FILTER_SCHARR);
//CX = CX.reshape(CV_8U, 1);
//Mat CXY = CY * CX;
//cout << BXY << endl;
//cout << CXY << endl;

//先进行高斯模糊去除噪声   拉普拉斯算子(方向无关，易受噪声影响)   canny算子

//GaussianBlur(lenagray, lenagray,Size(3,3),5);
//Mat lap, can;
//Laplacian(lenagray, lap, CV_16S, 3, 1);
//convertScaleAbs(lap, lap);
//Canny(lenagray, can, 80, 120,3);

//连通域分析


//图像距离变换     欧式，街区，棋盘距离

//图像形态学  腐蚀（去除微小，分离）结构元素判断保留还是去除      膨胀

//////生成用于腐蚀的原图像
//Mat src = (Mat_<uchar>(6, 6) << 0, 0, 0, 0, 255, 0,
//0, 255, 255, 255, 255, 255,
//0, 255, 255, 255, 255, 0,
//0, 255, 255, 255, 255, 0,
//0, 255, 255, 255, 255, 0,
// 0, 0, 0, 0, 0, 0);
//Mat struct1 = getStructuringElement(0, Size(3, 3));
//Mat struct2 = getStructuringElement(1, Size(3, 3));
//Mat src_erode;
//erode(src, src_erode, struct2);
////膨胀  通过结构元素添加新元素
//Mat src_dilate;
//dilate(src, src_dilate, struct2);

//形态学应用  开（先腐蚀再膨胀，去除图像中的噪声,消除较小连通域,保留较大连通域），闭（先膨胀再腐蚀，去除连通域内的小型空洞,平滑物体轮廓,连接两个临近的连通域）
//形态学梯度   顶帽  黑帽   集中击不中运算

////用于验证形态学应用的二值化矩阵
//Mat src = (Mat_<uchar>(9, 12) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 255, 0,
//0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0,
//0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0,
//0, 255, 255, 255, 0, 255, 255, 255, 0, 0, 0, 0,
//0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0,
//0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 255, 0,
//0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0,
//0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
////namedWindow("src", WINDOW_NORMAL);  //可以自由调节显示图像的尺寸
//Mat kernel = getStructuringElement(0, Size(3, 3));
//Mat open, close, gradient,tophat, blackhat, hitmiss,lenaopen;
//morphologyEx(src, open,MORPH_OPEN,kernel);					//先腐蚀再膨胀
//morphologyEx(src, close, MORPH_CLOSE, kernel);				//先膨胀再腐蚀
//morphologyEx(src, gradient, MORPH_GRADIENT, kernel);	//膨胀减去腐蚀
//morphologyEx(src, tophat, MORPH_TOPHAT, kernel);			//原图减开运算
//morphologyEx(src, blackhat, MORPH_BLACKHAT, kernel);	//闭运算减原图
//morphologyEx(src, hitmiss, MORPH_HITMISS, kernel);			//完全相同保留
//morphologyEx(lenagray, lenaopen, MORPH_OPEN, kernel);

//图像细化 骨架化 include ximgproc


//轮廓检测


//轮廓的面积，长度


//凸包检测



//直线检测 霍夫变换（图形空间向参数空间中变换，用极坐标表示）



//点集拟合 fitline   minencodingcircle/triangle


//QR二维码识别  QRcodedetector


//积分图像（左上方像素求和）防止重复计算

//Mat lenaint, lenaintsqr, lenainttr;
//integral(lenagray,lenaint);
//normalize(lenaint, lenaint, 0, 255, NORM_MINMAX);
//integral(lenagray, lenaint, lenaintsqr);
//normalize(lenaintsqr, lenaintsqr, 0, 255, NORM_MINMAX);
//integral(lenagray, lenaint, lenaintsqr,lenainttr);
//normalize(lenainttr, lenainttr, 0, 255, NORM_MINMAX);
//normalize(lenaint, lenaint, 0, 255, NORM_MINMAX);

//图像分割 漫水法（泼水淹空）

//RNG rng(10086);
//int connect = 4;
//int maskval = 255;
//int flag = connect | (maskval << 8) | FLOODFILL_FIXED_RANGE;
//Scalar updiff = Scalar(20, 20, 20);
//Scalar lowdiff = Scalar(20, 20, 20);
//Mat mask = Mat::zeros(lena.rows+2, lena.cols+2, CV_8UC1);
//while (true)
//{
//	int py = rng.uniform(0, lena.rows - 1);
//	int px = rng.uniform(0, lena.cols - 1);
//	Point point = Point(px, py);
//	Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
//	int area = floodFill(lena, mask,point, color,nullptr, lowdiff, updiff, flag);
//	imshow("lena", lena);
//	imshow("mask", mask);
//	int k = waitKey(0);
//	if ((k & 255)==27)
//	{
//		break;
//	}
//}

//分水岭法（排序找注水点，注水找到分水岭）

//Mat lenashed;
//watershed(lena,lenashed);

//Harris角点检测

////图像梯度计算：首先，对输入的图像应用 Sobel 算子或其他梯度算子，计算图像在 x 和 y 方向上的梯度。这将产生两个梯度图像（dx 和 dy），表示图像中每个像素点的梯度大小和方向。
////计算协方差矩阵：对于每个像素点，通过计算在其周围窗口中的梯度值的协方差矩阵来评估角点的可能性。协方差矩阵包括对应像素点的 x 和 y 方向梯度的平方和以及它们的乘积。
////计算角点响应函数：利用协方差矩阵的特征值来计算角点响应函数。通常，使用以下响应函数 R 来评估角点的重要性：
////R = λ1 * λ2 - k * (λ1 + λ2) ^ 2
////其中，λ1 和 λ2 是协方差矩阵的特征值，k 是一个经验常数。
////非极大值抑制：在计算角点响应函数后，根据响应函数的值，对每个像素点进行非极大值抑制。这意味着只有当当前像素点的响应值是其周围像素点中最大的时候，才将该像素点作为角点。
////设置阈值：可以根据应用的需求设置一个阈值，去除响应函数低于阈值的角点。
//Mat harris;
//cornerHarris(lena, harris, 2, 3,0.04);
//Mat harris_norm;
//normalize(harris, harris_norm,0,255,NORM_MINMAX);
//convertScaleAbs(harris_norm, harris_norm);
//vector<KeyPoint>points;
//for (int row = 0; row < harris_norm.rows; row++)
//{
//	for (int col = 0; col < harris_norm.cols; col++)
//	{
//		int R = harris_norm.at<uchar>(row, col);
//		if (R>125)
//		{
//			KeyPoint kp;
//			kp.pt.x = col;
//			kp.pt.y = row;
//			points.push_back(kp);
//		}
//	}
//}
//drawKeypoints(lena, points, lena);
//imshow("cornermap", harris_norm);
//imshow("corner", lena);

//hsi-Tomas角点检测 判断是否为角点不同

//vector<Point2f> corners;
//goodFeaturesToTrack(lenagray, corners,100,0.01,0.04,Mat());
//vector<KeyPoint>kp;
//for (int i = 0; i < corners.size(); i++)
//{
//	KeyPoint Kp ;
//	Kp.pt = corners[i];
//	kp.push_back(Kp);
//}
//drawKeypoints(lena, kp, lena);

//角点检测亚像素优化

//vector<Point2f> corners;
//goodFeaturesToTrack(lenagray, corners, 100, 0.01, 0.04, Mat());
//vector<Point2f>kpba = corners;
//TermCriteria cri = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 40, 0.01);
//cornerSubPix(lenagray, corners,Size(3,3),Size(-1,-1),cri);
//for (int i = 0; i < corners.size(); i++)
//{
//	cout << kpba[i]<< endl;
//	cout << corners[i] << endl;
//}
//vector<KeyPoint>kp;
//for (int i = 0; i < corners.size(); i++)
//{
//	KeyPoint Kp;
//	Kp.pt = corners[i];
//	kp.push_back(Kp);
//}
//drawKeypoints(lena, kp, lena);

//ORB特征点

//Ptr<ORB> orb =  ORB::create();
//vector<KeyPoint> kps;
//orb->detect(lena, kps);
//Mat desc;
//orb->compute(lena, kps, desc);
//drawKeypoints(lena, kps, lena);
//drawKeypoints(lena, kps, lena,Scalar(0,0,0),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//namedWindow("lena", WINDOW_FREERATIO);
//imshow("lena", lena);

//特征点匹配

//vector<KeyPoint>kps1, kps2;
//Mat des1, des2;
//ord_feature(lenagray, kps1, des1);
//ord_feature(lenahead, kps2, des2);
//vector<DMatch>dm;
//BFMatcher matcher(NORM_HAMMING);
//matcher.match(des1, des2, dm);
//double maxdis = 0, mindis = 10000;
//for (int i = 0; i < dm.size(); i++)
//{
//	double dis = dm[i].distance;
//	if (dis > maxdis) maxdis = dis;
//	if (dis < mindis) mindis = dis;
//}
//cout << maxdis << mindis << endl;
//vector<DMatch>gdm;
//for (int i = 0; i < dm.size(); i++)
//{
//	double dis = dm[i].distance;
//	if (dis <= max(2 * mindis, 20.0)) {
//		gdm.push_back(dm[i]);
//	}
//}
//Mat out;
//drawMatches(lenagray, kps1, lenahead, kps2, gdm, out);
//namedWindow("out", WINDOW_FREERATIO);
//imshow("out", out);

//RANSAC特征点优化

//vector<KeyPoint>kps1, kps2;
//Mat des1, des2;
//ord_feature(lenagray, kps1, des1);
//ord_feature(lenahead, kps2, des2);
//BFMatcher matcher(NORM_HAMMING);
//vector<DMatch>dm, gdm,grs;
//matcher.match(des1, des2, dm);
//match_min(dm, gdm);
//ransac(gdm, kps1, kps2, grs);
//cout << gdm.size() << grs.size() << endl;
//Mat out, sc,org;
//drawMatches(lenagray, kps1, lenahead, kps2, dm, org);
//namedWindow("org", WINDOW_FREERATIO);
//imshow("org", org);
//drawMatches(lenagray, kps1, lenahead, kps2, gdm, out);
//namedWindow("out", WINDOW_FREERATIO);
//imshow("out", out);
//drawMatches(lenagray, kps1, lenahead, kps2, grs, sc);
//namedWindow("grc", WINDOW_FREERATIO);
//imshow("grc", sc);

//单目相机模型，标定

//vector<Mat>imgs;
//string imgname;
//ifstream fin("pic.txt");
//while (getline(fin,imgname))
//{
//	Mat pic = imread(imgname, IMREAD_GRAYSCALE);
//	imgs.push_back(pic);
//}
//vector<vector<Point2f>> imgpts;
//for (int i = 0; i < imgs.size(); i++)
//{
//	Mat img1 = imgs[i];
//	vector<Point2f> cor;
//	findChessboardCorners(img1, Size(6, 8),cor);
//	find4QuadCornerSubpix(img1, cor,Size(11,11));
//	//drawChessboardCorners(img1, Size(6, 8), cor,true);
//	//imshow("img1", img1);
//	//waitKey(0);
//	imgpts.push_back(cor);
//}
//Size rect = Size(1, 1);
//vector<vector<Point3f>> objpts;
//for (int i = 0; i < imgpts.size(); i++)
//{
//	vector<Point3f>pt;
//	for (int x = 1; x < 7; x++)
//	{
//		for (int y = 1; y < 9;y++)
//		{
//			Point3f realpt;
//			realpt.x = rect.width * x;
//			realpt.y = rect.height * y;
//			realpt.z = 0;
//			pt.push_back(realpt);
//		}
//	}
//	objpts.push_back(pt);
//}
//Size picsize;
//picsize.width = imgs[0].cols;
//picsize.height = imgs[0].rows;
//Mat cameramatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));
//Mat diffmatrix = Mat(1, 5, CV_32FC1, Scalar::all(0));
//vector<Mat> rotation;
//vector<Mat> mov;
//calibrateCamera(objpts, imgpts, picsize, cameramatrix, diffmatrix, rotation, mov);
//cout << cameramatrix << endl;
//cout << diffmatrix << endl;
//for (int i = 0; i < rotation.size(); i++)
//{
//	cout << rotation[i] << endl;
//	cout << mov[i] << endl;
//}



waitKey(0);
return 0;
}
