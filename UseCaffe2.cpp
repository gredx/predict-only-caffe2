

#include "caffe2/core/flags.h"
#include "caffe2/core/init.h"
#include "caffe2/core/predictor.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/utils/math.h"

#include <memory>
#include <ctime>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
// DEBUG
//#pragma comment(lib,"opencv_world401d.lib")
//#pragma comment(lib,"openblasd035.lib")
//#pragma comment(lib,"libprotobufd.lib")
//#pragma comment(lib,"libprotobuf-lited.lib")
//#pragma comment(lib,"libprotocd.lib")

//RELAESE
#pragma comment(lib,"opencv_world401.lib")
#pragma comment(lib,"libprotoc.lib")
#pragma comment(lib,"libprotobuf.lib")
#pragma comment(lib,"libprotobuf-lite.lib")
#pragma comment(lib,"openblas035.lib")

using namespace cv;
using namespace std;
void readFromBinaryFile(string fileName, caffe2::NetDef& net)
{
	fstream input(fileName, ios::in | ios::binary);
	net.ParseFromIstream(&input);
}
namespace caffe2 {


	shared_ptr<Predictor> load_net(string init_netName,string predict_netName) {

		// 定义初始化网络结构与权重值
		caffe2::NetDef init_net, predict_net;
		DeviceOption op;
		op.set_random_seed(1701);

		std::unique_ptr<CPUContext> ctx_;
		ctx_ = caffe2::make_unique<CPUContext>(op);

		// 读入网络结构文件
		//readFromBinaryFile(init_netName, init_net);
		//readFromBinaryFile(predict_netName, predict_net);
		string initName = "init_net.pb";
		//ReadProtoFromBinaryFile(initName, &init_net);
		//ReadProtoFromBinaryFile("predict_net.pb", &predict_net);
		ReadProtoFromFile(initName, &init_net);
		ReadProtoFromFile("predict_net.pb",&predict_net);

		//ReadProtoFromTextFile("init_net.cpp", &init_net);
		//ReadProtoFromTextFile("predict_net.cpp", &predict_net);
		//WriteProtoToTextFile( init_net, "init_net");
		//WriteProtoToTextFile(predict_net, "predict_net");

		auto predictor = caffe2::make_unique<Predictor>(init_net, predict_net);
		return predictor;
	}
	TensorCPU& preProcessImg(string imgName, TensorCPU& input)
	{
		cv::Mat bgr_img = cv::imread(imgName);
		int height = bgr_img.rows;
		int width = bgr_img.cols;

		// 输入图像大小
		const int predHeight = 256;
		const int predWidth = 256;
		const int crops = 1;      // crops等于1表示batch的数量为1
		const int channels = 3;   // 通道数为3，表示BGR，为1表示灰度图
		const int size = predHeight * predWidth;
		const float hscale = ((float)height) / predHeight; // 计算缩放比例
		const float wscale = ((float)width) / predWidth;
		const float scale = std::min(hscale, wscale);
		// 初始化网络的输入，因为可能要做batch操作，所以分配一段连续的存储空间
		std::vector<float> inputPlanar(crops * channels * predHeight * predWidth);

		std::cout << "before resizing, bgr_img.cols=" << bgr_img.cols << ", bgr_img.rows=" << bgr_img.rows << std::endl;
		// resize成想要的输入大小
		cv::Size dsize = cv::Size(bgr_img.cols / wscale, bgr_img.rows / hscale);
		cv::resize(bgr_img, bgr_img, dsize);
		std::cout << "after resizing, bgr_img.cols=" << bgr_img.cols << ", bgr_img.rows=" << bgr_img.rows << std::endl;
		// Scale down the input to a reasonable predictor size.
		// 这里是将图像复制到连续的存储空间内，用于网络的输入，因为是BGR三通道，所以有三个赋值
		// 注意imread读入的图像格式是unsigned char，如果你的网络输入要求是float的话，下面的操作就不对了。
		for (auto i = 0; i < predHeight; i++) {
			//printf("+\n");
			for (auto j = 0; j < predWidth; j++) {
				inputPlanar[i * predWidth + j + 0 * size] = (float)bgr_img.data[(i*predWidth + j) * 3 + 0];
				inputPlanar[i * predWidth + j + 1 * size] = (float)bgr_img.data[(i*predWidth + j) * 3 + 1];
				inputPlanar[i * predWidth + j + 2 * size] = (float)bgr_img.data[(i*predWidth + j) * 3 + 2];
			}
		}
		// input就是网络的输入，所以把之前准备好的数据赋值给input就可以了
		input.Resize(std::vector<int>({ crops, channels, predHeight, predWidth }));
		input.ShareExternalPointer(inputPlanar.data());
		return input;
	}

}

void loadClassfier(vector<string>& imagenet_classes, string fileName)
{
	fstream classesFile(fileName, ios::in);
	string s;
	for (int i = 0; getline(classesFile, s); i++) {
		//cout << i << ":  " << s.substr(s.find(" ")+1) << endl;
		imagenet_classes.push_back(s.substr(s.find(" ") + 1));
	}
}


int main() {
	caffe2::GlobalInit();

	auto predictor = caffe2::load_net("init_net","predict_net");	// 加载参数文件
	vector<string> imagenet_classes;
	loadClassfier(imagenet_classes, "imagenet_classes.txt");		//加载分类结果文件

	// 输入图像,处理图像,预测
	string imgName;
	while (cin >> imgName)
	{
		
		caffe2::TensorCPU input;
		cv::Mat bgr_img = cv::imread(imgName);
		int height = bgr_img.rows;
		int width = bgr_img.cols;

		// 输入图像大小
		const int predHeight = 256;
		const int predWidth = 256;
		const int crops = 1;      // crops等于1表示batch的数量为1
		const int channels = 3;   // 通道数为3，表示BGR，为1表示灰度图
		const int size = predHeight * predWidth;
		const float hscale = ((float)height) / predHeight; // 计算缩放比例
		const float wscale = ((float)width) / predWidth;
		const float scale = std::min(hscale, wscale);
		// 初始化网络的输入，因为可能要做batch操作，所以分配一段连续的存储空间
		std::vector<float> inputPlanar(crops * channels * predHeight * predWidth);

		std::cout << "before resizing, bgr_img.cols=" << bgr_img.cols << ", bgr_img.rows=" << bgr_img.rows << std::endl;
		// resize成想要的输入大小
		cv::Size dsize = cv::Size(bgr_img.cols / wscale, bgr_img.rows / hscale);
		cv::resize(bgr_img, bgr_img, dsize);
		std::cout << "after resizing, bgr_img.cols=" << bgr_img.cols << ", bgr_img.rows=" << bgr_img.rows << std::endl;
		// Scale down the input to a reasonable predictor size.
		// 这里是将图像复制到连续的存储空间内，用于网络的输入，因为是BGR三通道，所以有三个赋值
		// 注意imread读入的图像格式是unsigned char，如果你的网络输入要求是float的话，下面的操作就不对了。
		for (auto i = 0; i < predHeight; i++) {
			//printf("+\n");
			for (auto j = 0; j < predWidth; j++) {
				inputPlanar[i * predWidth + j + 0 * size] = (float)bgr_img.data[(i*predWidth + j) * 3 + 0];
				inputPlanar[i * predWidth + j + 1 * size] = (float)bgr_img.data[(i*predWidth + j) * 3 + 1];
				inputPlanar[i * predWidth + j + 2 * size] = (float)bgr_img.data[(i*predWidth + j) * 3 + 2];
			}
		}
		// input就是网络的输入，所以把之前准备好的数据赋值给input就可以了
		input.Resize(std::vector<int>({ crops, channels, predHeight, predWidth }));
		input.ShareExternalPointer(inputPlanar.data());

		caffe2::Predictor::TensorVector inputVec{ &input }, outputVec;
		predictor->run(inputVec, &outputVec);

		// 输出预测结果
		float max_value = 0;
		int best_match_index = -1;
		for (auto output : outputVec)
		{
			for (auto i = 0; i < output->size(); i++)
			{
				float val = output-> data<float>()[i];
				if (val > 0.01) {
					cout << i << ":\t" << imagenet_classes[i] << " :\t" << val << endl;
				}
				if (val > max_value) {
					max_value = val;
					best_match_index = i;
				}
			}
		}
		cout << "predicted result is:" << imagenet_classes[best_match_index] << ", with confidence of " << max_value << endl;

	}

	// This is to allow us to use memory leak checks.
	google::protobuf::ShutdownProtobufLibrary();

	return 0;
}


