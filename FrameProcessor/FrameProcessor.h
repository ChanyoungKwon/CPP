#ifndef FRAMEPROCESSOR_H
#define FRAMEPROCESSOR_H

// STD
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <thread>
#include <future>
#include <atomic>
#include <mutex>
#include <chrono>
#include <bitset>

// QT
#include <QThread>
#include <QPainter>

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/version.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui/highgui.hpp>

// Yolo
#include <yolo_v2_class.hpp>

// ENDOAI
#include <MainWindow.h>

#define HASH_ROW_NUMS 64
#define HASH_COLUMN_NUMS 64
#define HASH_FRAME_SIZE (HASH_ROW_NUMS * HASH_COLUMN_NUMS)

#if defined(_WIN32)
#define CHRONO_TIME_POINT std::chrono::steady_clock::time_point
#elif defined(__linux__)
#define CHRONO_TIME_POINT std::chrono::system_clock::time_point
#endif

#if defined(_WIN32)
#include <direct.h>   //Diagnosis
#include <io.h>    // to check file existence using POSIX function access(). On Linux include <unistd.h>.
#elif defined(__linux__)
#include <unistd.h>
#include <decklink/LiveVideoWithOpenCV.h>
#endif

namespace frmproc
{
	extern int modelLevel;

	enum class FrameLabel {
		SKIP,
		BYPASS,					// bypass before inference
		CHECK,					// send to some checking thread
		INFERENCE,				// after inference, polyp not found
		DETECTION,				// after inference, polyp found
		DETECTION_WITH_BLUR,	// after inference, polyp found, blur is high
		BLUR
	};

	// for insertion recognition
	struct HsvData {
		float hueSum;
		float saturationSum;
		float valueSum;
		float decision;
		std::list<float> hue;
		std::list<float> saturation;
		std::list<float> value;
	};

	// 불필요한 테두리 부분을 제거한 중심 영역 
	struct FrameContour {
		cv::Mat maskFrame;
		cv::Rect coordinate;
		bool isValid;
	};

	struct TrackingBox {
		int frame;
		int id;
		cv::Rect_<float> box;
	};

	struct FrameData {
		cv::Mat frame;
		std::shared_ptr<image_t> detImage;
		std::vector<bbox_t> resultVec;
		bool exitFlag;
		unsigned int frameID;
		CHRONO_TIME_POINT captureTime;
		FrameLabel frameLabel;
		CHRONO_TIME_POINT endTime;
		FrameData() : frameID(0), exitFlag(false), frameLabel(FrameLabel::SKIP) {}
	};

	template<typename T>
	class InterThreadDataQueueT {
	private:
		static const size_t FRAME_QUEUE_SIZE_LIMIT = 100;
		std::mutex mutex;
		std::condition_variable condition;
		std::queue<T> data_queue;
		std::atomic<size_t> data_queue_size;

	public:
		void put(T const& data) {
			std::unique_lock<std::mutex> lock(mutex);
			condition.wait(lock, [&]() { return data_queue.size() < FRAME_QUEUE_SIZE_LIMIT; });
			data_queue.push(data);
			data_queue_size.store(data_queue.size());
			condition.notify_one();
		}

		T get(void) {
			std::unique_lock<std::mutex> lock(mutex);
			condition.wait(lock, [&]() { return !data_queue.empty(); });
			T data = data_queue.front();
			data_queue.pop();
			data_queue_size.store(data_queue.size());
			condition.notify_one();
			return data;
		}

		size_t size(void) {
			return data_queue_size.load();
		}
	};


	template<typename T>
	class InterThreadDataObjectT {
	private:
		std::atomic<T*> aPtr;
		std::mutex mutex;
		std::condition_variable condition;

	public:
		void send(T const& inputValue) {
			T* newPtr = new T;
			*newPtr = inputValue;
			std::unique_lock<std::mutex> lock(mutex);
			condition.wait(lock, [&]() { return aPtr.load() == NULL; });
			std::unique_ptr<T> oldPtr(aPtr.exchange(newPtr));
			condition.notify_one();
		}

		T receive(void) {
			std::unique_ptr<T> ptr;
			std::unique_lock<std::mutex> lock(mutex);
			condition.wait(lock, [&]() { return aPtr.load() != NULL; });
			ptr.reset(aPtr.exchange(NULL));
			condition.notify_one();
			return *ptr;
		}

		bool isObjectPresent(void) {
			return (aPtr.load() != NULL);
		}

		InterThreadDataObjectT(void) : aPtr(NULL) {
		}
	};

#if defined(_WIN32)
	class Capture {
	};
#endif

	class FrameProcessor : public QThread {

	public:
		FrameProcessor(QObject* parent = nullptr);

		static void setDetector(int level);
		void setDisplayLabel(QLabel* label);

		QPixmap cvMatToQPixmap(const cv::Mat& inMat);
		void reCropPixmap(void);
		bool resizeDisplay(cv::Rect& displaySize);
		void resizeFrame(cv::InputArray src, cv::OutputArray dst, double fx = 0, double fy = 0);
		bool isStateInside(void);
		bool isCapturingReady(void);
		void captureProcess(void);
		void displayFPS(void);
		void prepareProcess(void);
		void checkBlurProcess(void);
		void detectProcess(void);
		void drawProcess(void);
		void soundProcess(void);
		void recordProcess(void);

	protected:
		void inference(frmproc::FrameData& fd);
		void tracking(frmproc::FrameData& fd);
		bool checkBlurUsingCV(frmproc::FrameData& fd);
		bool checkFrozenFrame(frmproc::FrameData& fd);
		void saveSnapshot(const cv::Mat& plainScreen, frmproc::FrameData& detectionData);

	public:
		std::string saveSnapshotTime(void);
		cv::Mat letterbox(const cv::Mat& src, unsigned char pad, bool isBiased);
		void restoreResultVector(std::vector<bbox_t>& results, int width, int height, bool isBiased);
		void drawBoxes(cv::Mat& matImg, std::vector<bbox_t> resultVec);
		std::vector<std::string> objectsNamesFromFile(std::string const fileName);
		double getIOU(cv::Rect_<float> predictionBox, cv::Rect_<float> groundTruthBox);
		void calcHash(cv::Mat_ <double> image, const int& height, const int& width, std::bitset <HASH_FRAME_SIZE>& distanceHash);
		double hammingDistance(const std::bitset <HASH_FRAME_SIZE>& distanceHash1, const std::bitset <HASH_FRAME_SIZE>& distanceHash2);

#if defined(SINGLE_FRAMETIMER)
		void setFrameTimeLogFileName(void);
		void putFrameTimeLog(FrameTime& log);
#endif

	protected:
		void run(void) override;

		std::thread tFps;
		std::thread tCap;
		std::thread tPrepare;
		std::thread tCheckBlur;
		std::thread tDetect;
		std::thread tDraw;
		std::thread tSound;
		std::thread tRecord;

	private:
		QImage cvMatToQImage(const cv::Mat& inMat);
		QImage setDrawText(QImage& img, std::string str, int numFrame);
		frmproc::FrameContour getMainFrameFromVideoCapture(cv::VideoCapture* cvCap, frmproc::Capture* deckCap);
		frmproc::FrameContour getMainFrameFromMat(cv::Mat& img);
		bool checkInOut(cv::Mat& frame, int windowSize, float threshold);
		void resetHsvData(void);
		void initializeFrame(void);


		////////////////////////////////////////
		// QT
		////////////////////////////////////////
	public:
		Q_OBJECT

	signals:
		void newPixmapCaptured(void);
		void clearPixmap(void);
		void newPolypDetect(void);      //polyp detect
		void timerAction(bool timerAction);
		void newSnapshot(QImage img);
		void stopExamination(void);


	public slots:
		void stopVideo(void);
		bool isBlackFrame(cv::Mat& frame);
		////////////////////////////////////////
		// MEMBER
		////////////////////////////////////////
	public:
		bool stopped;
		bool isRunning;
		int frameCount;
		QPixmap pixmap;
		frmproc::HsvData hsv = {};
		bool camOpend;

	private:
		static Detector* detector;
		static float thresh;
		static std::thread threadCreateDetector;

	private:
		QLabel* displayLabel;
		bool needCrop;
		bool currentState;
		bool needNewRecord;
		std::atomic<size_t> numOfCaptureFrames;
		std::atomic<size_t> numOfInferenceFrames;
		std::atomic<size_t> numOfDisplayFrames;
		bool needResize;
		frmproc::FrameContour frameCrop;
		cv::VideoCapture* cap;
		cv::Mat* framePool;
		frmproc::Capture* capture;
	};
};

#endif // FRAMEPROCESSOR_H
