# predict-only-caffe2
extract necessary code in caffe2 to predict using trained model in CPU
## 从caffe2 开源的代码中抽取 用于加载已训练神经网络参数,使用CPU进行预测的 部分代码,并运行成功一个预测模型


#### 配置过程使用的资源来源及版本信息:
- pytorch  0.4.1
- protobuf 2.7.0
- openblas 0.3.5
- eigen 最新版
- opencv  4.0.1
- squeezenet  & imagenet_classes

#### 各个资源的用途:
- github 上开源了pytorch 的源码,其中的**caffe2文件夹 是框架的核心**.
    - 目前,官方github上只有v1.01和其他不知名的branch.
    - 当时选择v0.4.1的原因是:新版本的caffe2代码引用了caffe2文件夹以外的代码,并且代码组织方式混乱,头文件内经常找不到类的定义. 此外,v1.01版caffe2 甚至修改了部分变量的名字,与caffe2官网提供的api不匹配. 

- protobuf是用于**加载网络参数文件的接口**.
    - protobuf 主要用来定义结构化数据,将数据序列化.
    - 可以用protobuf的语法快速定义数据格式并生成对应代码,支持c++,Javampython三种语言的api
    - protobuf提供的序列化和反序列化接口使得 它可以作为数据通信和数据存储的工具
    - protobuf序列化的速度和空间利用率很高,相比xml和json 具有性能上绝对的优势
    - protobuf 定义数据结构的语法很简单,一目了然,易于学习

- openblas和eigen 是用于高效数学计算的库,caffe2用到这两个库做底层数学计算
- opencv 主要用来读取和预处理图片,是边缘依赖
- 项目中运行的是squeezenet,使用了imagenet的分类结果



#### 项目文件组织结构
    -caffe2
        -core           /  最核心,最基础的部分.基本数据类型的定义,框架级的操作定义
        -operators      /   基本操作的定义,可按需增减.卷积,池化,交叉熵,dropout等
        -proto          / 定义一些重要的数据类型,如设备信息,网络结构,operator结构
        -utils          / 一些工具类.线程,数学计算,wrapper等
    -Eigen          / 也可放在include文件夹内
    -include
        -google
            -protobuf   / protobuf提供的接口
        -opencv2
    -lib            / 链接库
    init_net        /网络参数文件
    predict_net     /网络结构定义文件
    imagenet_classes.txt    /分类结果
    imag.jpg        
    demo.cpp        /用户代码
    
#### 如果想使用不同版本的包,可按照包的功能更换文件夹和链接文件

### 对各个部分详细描述

#### protobuf和.proto文件
- caffe2使用了protobuf作为数据交互的媒介,caffe2/proto文件夹内的.proto文件定义了用户能接触到的最重要的数据定义.
- caffe2.proto文件的重要内容  [参考博客](http://www.cnblogs.com/dkblog/archive/2012/03/27/2419010.html)
    ``` 
    syntax = "proto2";
    
    package = caffe2;
    // message相当于class 内部的定义和C++/Java类似
    // required 必须含这个值
    // optional 表示该值可有0或1个,repeated 表示该属性是可重复(数组/vector)
    // 后面的 = num 表示该属性的标识 编码时[1,15]占1个字节,[16,2047]占2字节
    enum DeviceType {
        CPU = 0;                    // In default, we will use CPU.
        CUDA = 1;                   // CUDA.
        MKLDNN = 2;                 // Reserved for explicit MKLDNN
        OPENGL = 3;                 // OpenGL
        OPENCL = 4;                 // OpenCL
        IDEEP = 5;                  // IDEEP.
        HIP = 6;                    // AMD HIP
        // Change the following number if you add more devices in the code.
        COMPILE_TIME_MAX_DEVICE_TYPES = 7;
        ONLY_FOR_TEST = 20901701;   // This device type is only for test.
    }
    message DeviceOption {
        optional int32 device_type = 1 [ default = 0 ]; // 0 is CPU.
        optional int32 cuda_gpu_id = 2;
        optional uint32 random_seed = 3;
        optional string node_name = 4;
        optional int32 numa_node_id = 5 [default = -1];
        repeated string extra_info = 6;
        optional int32 hip_gpu_id = 7;
    }
    
    message OperatorDef {
        repeated string input = 1; // the name of the input blobs
        repeated string output = 2; // the name of output top blobs
        optional string name = 3; // the operator name. This is optional.
        optional string type = 4;
        repeated Argument arg = 5;
        optional DeviceOption device_option = 6;
        optional string engine = 7;
        repeated string control_input = 8;
        optional bool is_gradient_op = 9 [default = false];
        optional string debug_info = 10;
    }
    // NetDef 是caffe2定义网络的完整类,一个参数文件就是一个NetDef对象序列化输出的结果
    message NetDef {
        optional string name = 1; // the network's name
        // Operators that the network contains.
        repeated OperatorDef op = 2;    // 可以从参数文件中查看该网络用到的所有Operator
        optional string type = 3;
        optional int32 num_workers = 4 [deprecated=true];
        optional DeviceOption device_option = 5;    //使用CPU device_type = 0
        repeated Argument arg = 6;
        repeated string external_input = 7;
        repeated string external_output = 8;
    }
    
    ```
- 参数文件有二进制和text两种,使用二进制的参数文件读写速度很快,二进制和text可以使用protobuf的接口进行转换.caffe2 model zool提供的是二进制的参数文件,将其转换为text可以看到网络的结构
- protobuf可以将proto语言定义的数据类型转换为代码.将protobuf安装包编译成可执行文件,以此进行转换.二进制文件的名字为protoc.exe
- 转化为C++的指令: protoc --cpp_out=./ caffe2.proto    第二个参数:输出目录,第三个参数: 输入文件
- **编译protobuf**:在protobuf下载目录下的cmake文件夹内的README.md 详细描述了编译C++版本的方法,不同版本的protobuf有细微差别. 
    -  大致过程:
    -  打开Visual Studio 本机命令提示符
    -  cd cmake
    -  mkdir build & cd build
    -  mkdir release (debug)
    -  cd release
    -  cmake -G "NMake Makefiles"  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../../../../install  ../..    生成makefile
    -  nmake    生成可执行文件和lib文件
    -  nmake install 会在install文件夹内生成include , bin,lib文件夹
    -  详情看README.md

#### caffe2抽取文件的方法
- 首先认识到core,operator,util,proto文件夹是必要的
- core和util内的文件有很多的头文件引用,需要什么就找什么
- 不要加入不必要的文件 ,会造成额外的引用
- 看文件名的意义,推断文件的功能
- 编译出错误看源码,追踪需要的类 等
- operator中的方法是可拆卸的,首先选择最常见的操作直接加入,之后可以根据参数文件定义的操作名字选择添加

#### 依赖的库文件汇总
- Release 版本如下,Debug版本将其替换为debug版库文件
- libprotobuf.lib
- libprotobuf-lite.lib
- libprotoc.lib             // 三个都是编译protobuf获得
- openblas.lib              // 编译openblas获得
- opencv_world401.lib       //opencv 下载包内获得

#### 编译openblas
- [参考博客](https://blog.csdn.net/weixin_35776029/article/details/52719079)
- 去openblas官网下载源文件
- [Perl](http://www.activestate.com/activeperl/downloads) 编译过程需要,可能需要翻墙
- 在openblas 文件夹内新建build文件夹
- 使用cmake-gui 选择openblas和输出文件夹build
- 点击configure,完成后点击generate就会在build文件夹内生成VS项目sln
- 使用VS打开sln文件,点击生成,VS会编译出对应版本的库文件和dll文件
- 不同版本的openblas可能有所不同,有的会生成dll而有的不会.生成了dll需要将dll文件放到exe文件夹内或者添加到环境变量.


#### 配置eigen
- eigen是C++实现的矩阵运算库,没有任何依赖,只要包含头文件就可使用其功能
- 从eigen官网下载文件,将文件夹里的Eigen文件夹复制到项目的目录下,加入项目即可

#### 配置opencv
- 下载opencv,将目录build/include内的opencv2文件夹复制到项目内
- 将build/x64/vc15/lib内的lib文件拷贝到项目内的lib目录下
- 这么做是为了使需要的资源都包括在项目内,当然可以通过配置项目包含目录和链接目录减少项目大小,但是,终归是要设置依赖,移植很麻烦


**至此,项目中必须的文件获得方法都描述完成了,接下来介绍项目的配置要点**


- 使用VS新建一个控制台项目
- 按照上述说明,将文件复制进项目内
- 在VS解决方案中,选择显示所有文件,右键这些文件夹,选择 包括在项目中
- 在项目属性页面,不管是配置release还是debug,32位或是64位,统一的配置是:
    - C/C++目录->常规->附加包含目录  添加./;./include;
    - 链接器->常规->链接库依赖项  添加./lib;
    - C/C++目录->常规->预处理器->预处理器定义 编辑,添加一行_CRT_SECURE_NO_WARNINGS
    - C/C++目录->常规->预编译头->设为不适用预编译头
- debug版本和release版本的不同配置:
    - C/C++ ->代码生成->运行库  release版本设为多线程(/MT),debug版本设为多线程调试(/MTD)
    - 注意链接器引入lib的版本,release版本链接release版本的lib库,如果引入错误会发生无法解析的字符或者运行时错误等. 
    - 一定留意 链接器->输入->附加依赖项中的lib , 点击右面的下拉框->编辑查看其引用的所有lib文件
    - 先引入的lib文件会屏蔽后面同样作用的lib文件

#### 参考示例代码[C++ 预测(predict)Demo](http://www.cnblogs.com/zhonghuasong/p/7297696.html)
- 代码中经测试无法正常工作的部分: ReadProtoFromFile ,从二进制文件读取参数失败,原因未知
- 修改方法:弃用ReadProtoFromFile(string , NetDef*),这个函数整合了ReadProtoFromBinaryFile 和 ReadProtoFromTextFile , 分开使用即可.
- 读取二进制参数文件可使用ReadProtoFromBinaryFile(string,NetDef*)
- 读取文本参数文件使用ReadProtoFromTextFile(string,NetDef*)

- **protobuf 提供的原始读取参数接口 :**
- 使用bool message.ParseFromIstream(istream* input)读取参数文件
- 使用bool SerializeToOstream(ostream* output)const 写入文件流
