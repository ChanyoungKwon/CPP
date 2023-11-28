#include <FrameProcessor.h>

#include <algorithm/Hungarian.h>
#include <algorithm/KalmanTracker.h>
#include <utility/SoundEffector.h>
#include <utility/SenaConfigParser.h>
#include <utility/SummaryInfoManager.h>

#include <time.h>
#include <stdio.h>

#define DECISION_INSIDE true
#define DECISION_OUTSIDE false
#define TIMER_START true
#define TIMER_PAUSE false

using namespace cv;
using namespace std;

extern int boxShape;
extern int boxWidth;
extern BoxColor bColor;
extern SoundEffector soundEffector;
extern SenaConfigParser* config;
extern SummaryInfoManager* summary;
extern MainWindow* mainWindow;

namespace frmproc
{
	/*
	AI 민감도 레벨
		초기값 : -1
		Standard : 0
		Performance : 1
	*/
	int modelLevel = -1;

	InterThreadDataObjectT<FrameData> bypassDetectToDrawObject; 
	InterThreadDataObjectT<FrameData> fromDetectToBlurCheckObject;
	InterThreadDataObjectT<FrameData> fromBlurCheckToDetectObject;
	InterThreadDataQueueT<FrameData> fromCaptureToPrepareQueue;
	InterThreadDataQueueT<FrameData> fromPrepareToDetectQueue;
	InterThreadDataQueueT<FrameData> fromDetectToDrawQueue, fromDetectToSoundQueue;
	InterThreadDataQueueT<FrameData> fromDrawToShowQueue, fromDrawToRecordQueue;
}

using namespace frmproc;

Detector* FrameProcessor::detector = NULL;
float FrameProcessor::thresh = 0.0f;
thread FrameProcessor::threadCreateDetector;

// snapshot 순간의 시간 정보를 반환
string FrameProcessor::saveSnapshotTime(void) {
	std::time_t tm_now = chrono::system_clock::to_time_t(chrono::system_clock::now());
	struct tm* t = localtime(&tm_now);
	std::string saveTime = cv::format("%04d-%02d-%02d %02d:%02d:%02d", t->tm_year + 1900, t->tm_mon + 1, t->tm_mday, t->tm_hour, t->tm_min, t->tm_sec);

	return saveTime;
}

cv::Mat FrameProcessor::letterbox(const cv::Mat& src, unsigned char pad, bool isBiased) {
	int maxSize = std::max(src.cols, src.rows);
	cv::Mat dst = cv::Mat::zeros(maxSize, maxSize, CV_8UC(src.channels())) + cv::Scalar(pad, pad, pad, 0);

	if (isBiased) {
		src.copyTo(dst(cv::Rect(0, 0, src.cols, src.rows)));
	} else {
		int posX = (maxSize - src.cols) / 2;
		int posY = (maxSize - src.rows) / 2;
		src.copyTo(dst(cv::Rect(posX, posY, src.cols, src.rows)));
	}

	return dst;
}

void FrameProcessor::restoreResultVector(std::vector<bbox_t>& results, int width, int height, bool isBiased) {
	int maxSize = std::max(width, height);
	int posX = 0;
	int posY = 0;

	if (isBiased == false) {
		posX = (maxSize - width) / 2;
		posY = (maxSize - height) / 2;
	}

	for (bbox_t& box : results) {
		box.x -= posX;
		box.y -= posY;
	}
}

void FrameProcessor::drawBoxes(cv::Mat& matImg, std::vector<bbox_t> resultVec) {
	for (auto& i : resultVec) {
		cv::Scalar color = obj_id_to_color(i.obj_id);
		// int condition이 더 빠름 (objNames[i.obj_id] == "polyp")

		// 중심좌표 구하기
		int centerX = i.x + int(i.w * 0.5000);
		int centerY = i.y + int(i.h * 0.5000);

		// ratio에 따른 x, y, w, h 좌표 재설정
		int leftX = centerX - int(config->boundingBoxMagnificationRatio * 0.5000 * i.w);
		int rightX = centerX + int(config->boundingBoxMagnificationRatio * 0.5000 * i.w);
		int topY = centerY - int(config->boundingBoxMagnificationRatio * 0.5000 * i.h);
		int botY = centerY + int(config->boundingBoxMagnificationRatio * 0.5000 * i.h);
		int newW = rightX - leftX;
		int newH = botY - topY;

		//// 진단을 위한 크기 선별
		//int frameSize = mat_img.size().width * mat_img.size().height;    //SCAI check input whole frame size to calculate optimal box size ratio for diagnosis
		//int box_size = i.w * i.h;                //SCAI check result polyp box size to calculate optimal box size ratio for diagnosis
		//int b_box = (255 * box_size) / 90000;    //90000 is test value that means polyp box size should be larger than 90000 for optimal diagnosis 

		// 범위를 벗어나는 좌표 수정
		// if (b_box > 255) b_box = 255;
		if (leftX < 0) {
			leftX = 0;
		}

		if (rightX > matImg.cols - 1) {
			rightX = matImg.cols - 1;
		}

		if (topY < 0) {
			topY = 0;
		}

		if (botY > matImg.rows - 1) {
			botY = matImg.rows - 1;
		}

		float corner = 0.2f;

		switch (boxShape) {
		case 0:  // rectangle
		{
			rectangle(matImg, cv::Rect(leftX, topY, newW, newH), Scalar(bColor.b, bColor.g, bColor.r), boxWidth, 16, 0); //Green Box SCAI 
			break;
		}
		case 1: // aim
		{
			Point aim[4][3];

			aim[0][0] = Point(int(leftX + corner * newW), topY);
			aim[0][1] = Point(leftX, topY);
			aim[0][2] = Point(leftX, int(topY + corner * newH));

			aim[1][0] = Point(int(rightX - (corner * newW)), topY);
			aim[1][1] = Point(rightX, topY);
			aim[1][2] = Point(rightX, int(topY + (corner * newH)));

			aim[2][0] = Point(rightX, int(botY - (corner * newH)));
			aim[2][1] = Point(rightX, botY);
			aim[2][2] = Point(int(rightX - (corner * newW)), botY);

			aim[3][0] = Point(int(leftX + (corner * newW)), botY);
			aim[3][1] = Point(leftX, botY);
			aim[3][2] = Point(leftX, int(botY - (corner * newH)));

			const Point* pAim[4] = { aim[0],aim[1],aim[2],aim[3] };
			int npts[] = { 3,3,3,3 };

			polylines(matImg, pAim, npts, 4, false, Scalar(bColor.b, bColor.g, bColor.r), boxWidth, 16, 0);
			break;
		}
		case 2: // ellipse
		{
			ellipse(matImg, Point(centerX, centerY), Size(0.5000 * newW, 0.5000 * newH), 0, 0, 360, Scalar(bColor.b, bColor.g, bColor.r), boxWidth, 16, 0);
			break;
		}
		default:
			break;
		}
	}
}

vector<string> FrameProcessor::objectsNamesFromFile(string const fileName) {
	ifstream file(fileName);
	vector<string> fileLines;
	if (!file.is_open()) {
		return fileLines;
	}
	for (string line; getline(file, line);) {
		fileLines.push_back(line);
	}

	return fileLines;
}

// Computes IOU between two bounding boxes
double FrameProcessor::getIOU(Rect_<float> predictionBox, Rect_<float> groundTruthBox) {
	float in = (predictionBox & groundTruthBox).area();
	float un = predictionBox.area() + groundTruthBox.area() - in;

	if (un < DBL_EPSILON)
		return 0;

	return (double)(in / un);
}

void FrameProcessor::calcHash(Mat_ <double> image, const int& height, const int& width, bitset <HASH_FRAME_SIZE>& distanceHash) {
	for (int rowIndex = 0; rowIndex < height; rowIndex++) {
		for (int columnIndex = 0; columnIndex < width - 1; columnIndex++) {
			distanceHash[rowIndex * height + columnIndex] = (image(rowIndex, columnIndex) > image(rowIndex, columnIndex + 1)) ? 1 : 0;
		}
	}
}

double FrameProcessor::hammingDistance(const bitset <HASH_FRAME_SIZE>& distanceHash1, const bitset <HASH_FRAME_SIZE>& distanceHash2) {
	double distance = 0;
	for (int i = 0; i < 4096; i++) {
		distance += (distanceHash1[i] == distanceHash2[i] ? 0 : 1);
	}
	return distance;
}

FrameProcessor::FrameProcessor(QObject* parent) :QThread { parent } {
	stopped = true;
	isRunning = false;
	displayLabel = NULL;
	needCrop = false;
	currentState = DECISION_OUTSIDE;
}

void FrameProcessor::setDisplayLabel(QLabel* label) {
	displayLabel = label;
}

bool FrameProcessor::isBlackFrame(cv::Mat& frame) {
	Mat gray;
	double minVal, maxVal;
	cvtColor(frame, gray, COLOR_BGR2GRAY);
	minMaxIdx(gray, &minVal, &maxVal);
	return minVal == maxVal;
}

FrameContour FrameProcessor::getMainFrameFromMat(Mat& img) {
	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);
	FrameContour frame = { Mat(gray.size().height, gray.size().width, CV_8UC1, Scalar(0)), { }, { } };

	//remove letter
	Mat threshImg;
	threshold(gray, threshImg, config->cropColorWhite, 255, THRESH_TOZERO_INV);
	threshold(threshImg, threshImg, config->cropColorBlack, 255, THRESH_BINARY);

	//fill the dot 
	Mat closing;
	dilate(threshImg, closing, Mat::ones(Size(5, 5), CV_8UC1), Point(-1, -1), 1);

	// find contour
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(closing, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);

	if (contours.empty()) { // contours를 찾을 부분이 없다면
		frame.isValid = false;
		return frame; // 빈화면, 빈좌표
	} else {
		frame.isValid = true;
	}

	vector<int> xSmallList;
	vector<int> xList;
	vector<int> xList2;
	vector<int> yList;
	vector<int> bigAreaIndex;

	for (unsigned int i = 0; i < contours.size(); i++) {
		if (contourArea(contours[i]) >= config->validContourValue) {
			for (unsigned int j = 0; j < contours[i].size(); j++) {
				xList2.push_back(contours[i][j].x);
			}
			int xMin = *min_element(xList2.begin(), xList2.end());
			xSmallList.push_back(xMin);
			bigAreaIndex.push_back(i);
		}
		xList2.clear();
	}

	if (xSmallList.size() <= 0) {
		frame.isValid = false;
		return frame;
	}

	int minIndex = min_element(xSmallList.begin(), xSmallList.end()) - xSmallList.begin();
	int areaIndex = bigAreaIndex[minIndex];

	for (unsigned int j = 0; j < contours[areaIndex].size(); j++) {
		xList.push_back(contours[areaIndex][j].x);
		yList.push_back(contours[areaIndex][j].y);
	}

	frame.coordinate.x = *min_element(xList.begin(), xList.end());
	frame.coordinate.y = *min_element(yList.begin(), yList.end());
	frame.coordinate.width = *max_element(xList.begin(), xList.end()) - frame.coordinate.x;
	frame.coordinate.height = *max_element(yList.begin(), yList.end()) - frame.coordinate.y;

	// draw contour
	drawContours(frame.maskFrame, contours, areaIndex, Scalar(255, 255, 255), -1);

	return frame;
}

FrameContour FrameProcessor::getMainFrameFromVideoCapture(VideoCapture* cvCap, Capture* deckCap) {
	Mat img;
	for (size_t cnt = 0; cnt < 10; cnt++) {
		if (cvCap && cvCap->read(img)) {
			return getMainFrameFromMat(img);
#if defined(__linux__)
		} else if (deckCap && deckCap->read(img)) {
			return getMainFrameFromMat(img);
#endif
		} else {
			this_thread::sleep_for(100ms);
		}
	}

	return { {}, {}, true };
}

bool FrameProcessor::isCapturingReady(void) {

#if defined(_WIN32)
	cap = config->videoTestMode ? new VideoCapture(config->videoTestFilePath) : new VideoCapture(0, CAP_DSHOW);
#elif defined(__linux__)
	return true;
	cv::VideoCapture cap = config->videoTestMode ? cv::VideoCapture(config->videoTestFilePath) : cv::VideoCapture("/dev/video0");
#endif
	MainWindow* mainUI = mainWindow;
	cv::Mat frame;

	// step 1. check camera is opened
	if (cap->isOpened() == false) {
		cap->release();
		return false;
	}

	// step 2. take first frame from cap
	// step 2.1. wait for frame input
	bool isRead = false;
	for (size_t cnt = 0; cnt < 10; cnt++) {
		if (isRead) {
			break;
		}

		this_thread::sleep_for(chrono::milliseconds(1000 / config->fps));
		isRead = cap->read(frame);
	}

	// step 2.2. read frame from video capture
	if (isRead == false) {
		cap->release();
		return false;
	}

	// step 3. check frame is empty
	if (frame.empty() == true) {
		cap->release();
		return false;
	}

	// step 4. check frame is black mat
	cv::Mat gray;
	double minVal, maxVal;
	cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
	minMaxIdx(gray, &minVal, &maxVal);
	if (minVal == maxVal) {
		cap->release();
		return false;
	}

	return true;
}

void FrameProcessor::initializeFrame(void) {
	if (isCapturingReady()) {
		MainWindow* mainUI = mainWindow;
		framePool = new Mat[config->framePoolSize];
#if defined(_WIN32)
		capture = NULL;

		if (cap && cap->isOpened() == false) {
			delete[] framePool;
			delete(cap);
			stopped = true;
			isRunning = false;
			// TODO:: testValid = examination end time - examiantion start time > examinaiton valid time 으로 변경 예정
			bool testValid = true;
			if (testValid) {
				mainUI->turnPageSingal(testValid);
			} else {
				mainUI->turnPageSingal(testValid);
			}
			mainWindow->ui->pageExamination->hideFullscreenModal();
			return;
		}
		cap->set(cv::CAP_PROP_FPS, config->fps);

		if (config->videoInputWidth > 0) {
			cap->set(cv::CAP_PROP_FRAME_WIDTH, config->videoInputWidth);
		}
		if (config->videoInputHeight > 0) {
			cap->set(cv::CAP_PROP_FRAME_HEIGHT, config->videoInputHeight);
		}
#elif defined(__linux__)
		VideoCapture* cap = config->videoTestMode ? new VideoCapture(config->videoTestFilePath) : NULL;
		DeviceManager deviceManager;
		Capture* capture = NULL;

		if (config->videoTestMode == false) {
			cout << "Start with DeckLink device" << endl;
			com_ptr<IDeckLinkInput> deckLinkInput = deviceManager.getDeckLinkInput();
			capture = Capture::CreateInstance(deckLinkInput, bmdModeHD1080p6000);
			capture->StartDeckLinkCapture();
		} else if (cap && cap->isOpened() == false) {
			cout << "Start with VideoCapture device" << endl;
			delete[] framePool;
			// TODO:: testValid = examination end time - examiantion start time > examinaiton valid time 으로 변경 예정
			bool testValid = true;
			if (testValid) {
				mainUI->turnPageSingal(testValid);
			} else {
				mainUI->turnPageSingal(testValid);
			}
			delete(cap);
			stopped = true;
			isRunning = false;
			return;
		} else {
			cout << "Start with VideoFile" << endl;
			cap->set(cv::CAP_PROP_FRAME_WIDTH, config->frameWidth);
			cap->set(cv::CAP_PROP_FRAME_HEIGHT, config->frameHeight);
			cap->set(cv::CAP_PROP_FPS, config->fps);
		}
#endif
		needResize = false;

		if (config->frameCropping) {
			frameCrop = getMainFrameFromVideoCapture(cap, capture);
		} else {
			frameCrop.coordinate.x = 0;
			frameCrop.coordinate.y = 0;
			frameCrop.coordinate.width = cap->get(cv::CAP_PROP_FRAME_WIDTH);
			frameCrop.coordinate.height = cap->get(cv::CAP_PROP_FRAME_HEIGHT);
			frameCrop.isValid = true;
		}

		if (frameCrop.isValid == false) { // mask crop를 실행했으나 비어있다면 탈출
			displayLabel->clear();
			delete[] framePool;
			if (cap) {
				cap->release();
				delete cap;
			}
			if (capture) {
#if defined(__linux__)
				capture->StopDeckLinkCapture();
#endif
			}

			mainUI->ui->pageExamination->showSmallModal("Invalid screen area");
			while (mainUI->ui->pageExamination->frameSmallModal->isVisible()) {
				this_thread::sleep_for(100ms);
			}

			emit stopExamination();
			isRunning = false;
			return;
		}
		needResize = resizeDisplay(frameCrop.coordinate);
	}
}

void FrameProcessor::inference(FrameData& fd) {
	if (fd.detImage) {
		if (config->framePadding == 0) {
			// case 1. frame padding none (without letterbox)
			fd.resultVec = detector->detect_resized(*fd.detImage, fd.frame.cols, fd.frame.rows, FrameProcessor::thresh, true);  // true, 이미 추론하면서 절대좌표를 반환
		} else if (config->framePadding == 1) {
			// case 2. frame padding biased (with biased letterbox)
			int maxSize = std::max(fd.frame.rows, fd.frame.cols);
			fd.resultVec = detector->detect_resized(*fd.detImage, maxSize, maxSize, FrameProcessor::thresh, true);  // true, 이미 추론하면서 절대좌표를 반환
		} else if (config->framePadding == 2) {
			// caes 3. frame padding balanced (with balanced letterbox)
			int maxSize = std::max(fd.frame.rows, fd.frame.cols);
			fd.resultVec = detector->detect_resized(*fd.detImage, maxSize, maxSize, FrameProcessor::thresh, true);  // true, 이미 추론하면서 절대좌표를 반환
			restoreResultVector(fd.resultVec, fd.frame.cols, fd.frame.rows, false);
		} else {
			// error
			cout << "Frame Padding Configuration Error : " << config->framePadding << endl;
		}
	}
	numOfInferenceFrames++;
}

void FrameProcessor::tracking(FrameData& fd) {
	static vector<TrackingBox> detectedBoxes;
	static vector<KalmanTracker> trackers;
	static vector<Rect_<float>> predictedBoxes;
	static vector<vector<double>> iouMatrix;
	static vector<int> assignment;

	static set<int> unmatchedDetections;
	static set<int> unmatchedTrajectories;
	static set<int> allItems;
	static set<int> matchedItems;
	static vector<cv::Point> matchedPairs;

	static unsigned int predictedNum = 0;
	static unsigned int detectedNum = 0;
	static int maxAge = 1;	// TODO: set by config file

	static size_t invalidHitCount = 0;

	if (fd.frameLabel == FrameLabel::SKIP) {
		invalidHitCount++;
	} else if (fd.frameLabel == FrameLabel::DETECTION_WITH_BLUR) {
		cout << "BLUR has been detected ------------------------------" << endl;
		invalidHitCount = 0;
	} else {
		invalidHitCount = 0;
	}

	//Sort algorithm
	if (fd.resultVec.size() != 0) {
		detectedBoxes.clear();
		TrackingBox tb;
		for (auto& i : fd.resultVec) {
			float tpx = i.x;
			tb.box = Rect_<float>(Point_<float>(i.x, i.y), Point_<float>(i.x + i.w, i.y + i.h));
			detectedBoxes.push_back(tb);
		}

		// the first frame met
		if (trackers.size() == 0) {
			// initialize kalman trackers using first detections.
			for (unsigned int i = 0; i < detectedBoxes.size(); i++) {
				KalmanTracker trk = KalmanTracker(detectedBoxes[i].box);
				trackers.push_back(trk);
			}
			return;
		}

		//3.1. get predicted locations from existing trackers.
		predictedBoxes.clear();

		for (auto it = trackers.begin(); it != trackers.end();) {
			Rect_<float> pBox = (*it).predict();
			if (pBox.x >= 0 && pBox.y >= 0) {
				predictedBoxes.push_back(pBox);
				it++;
			} else {
				it = trackers.erase(it);
			}
		}

		// 3.2. associate detections to tracked object (both represented as bounding boxes)
		predictedNum = predictedBoxes.size();
		detectedNum = detectedBoxes.size();

		iouMatrix.clear();
		iouMatrix.resize(predictedNum, vector<double>(detectedNum, 0));

		// compute iou matrix as a distance matrix
		for (unsigned int i = 0; i < predictedNum; i++) {
			for (unsigned int j = 0; j < detectedNum; j++) {
				// use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
				iouMatrix[i][j] = 1 - getIOU(predictedBoxes[i], detectedBoxes[j].box);
			}
		}

		// solve the assignment problem using hungarian algorithm.
		// the resulting assignment is [track(prediction) : detection], with len=preNum
		HungarianAlgorithm HungAlgo;
		assignment.clear();
		HungAlgo.Solve(iouMatrix, assignment);

		// find matches, unmatched_detections and unmatched_predictions
		unmatchedTrajectories.clear();
		unmatchedDetections.clear();
		allItems.clear();
		matchedItems.clear();

		// there are unmatched detections 
		if (detectedNum > predictedNum) {
			for (unsigned int n = 0; n < detectedNum; n++) {
				allItems.insert(n);
			}

			for (unsigned int i = 0; i < predictedNum; ++i) {
				matchedItems.insert(assignment[i]);
			}

			set_difference(allItems.begin(), allItems.end(), matchedItems.begin(), matchedItems.end(), insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
		} else if (detectedNum < predictedNum) { // there are unmatched trajectory/predictions
			for (unsigned int i = 0; i < predictedNum; ++i) {
				if (assignment[i] == -1) { // unassigned label will be set as -1 in the assignment algorithm
					unmatchedTrajectories.insert(i);
				}
			}
		} else {
			// do nothing
		}

		// filter out matched with low IOU
		matchedPairs.clear();
		for (unsigned int i = 0; i < predictedNum; ++i) {
			if (assignment[i] == -1) { // pass over invalid values
				continue;
			}

			if (1 - iouMatrix[i][assignment[i]] < config->iouThreshold) {
				unmatchedTrajectories.insert(i);
				unmatchedDetections.insert(assignment[i]);
			} else {
				matchedPairs.push_back(cv::Point(i, assignment[i]));
			}
		}

		///////////////////////////////////////
		// 3.3. updating trackers
		// update matched trackers with assigned detections.
		// each prediction is corresponding to a tracker
		int detIdx, trkIdx;
		for (unsigned int i = 0; i < matchedPairs.size(); i++) {
			trkIdx = matchedPairs[i].x;
			detIdx = matchedPairs[i].y;
			trackers[trkIdx].update(detectedBoxes[detIdx].box);
		}

		// create and initialise new trackers for unmatched detections
		for (auto umd : unmatchedDetections) {
			KalmanTracker tracker = KalmanTracker(detectedBoxes[umd].box);
			trackers.push_back(tracker);
		}

		// get trackers' output
		fd.resultVec.clear();

		for (auto it = trackers.begin(); it != trackers.end();) {
			if ((it->m_time_since_update < 1) && (it->m_hit_streak - invalidHitCount >= config->minHitCount)) {
				bbox_t res = {};
				StateType box = it->get_smoothed_box();
				res.x = box.x;
				res.y = box.y;
				res.w = box.width;
				res.h = box.height;

				res.frames_counter = it->m_hit_streak;
				res.track_id = it->m_id + 1;
				fd.resultVec.push_back(res);
				it++;
			} else {
				it++;
			}
			// remove dead tracklet
			if (it != trackers.end() && (*it).m_time_since_update > maxAge) {
				it = trackers.erase(it);
			}
		}
	}
}

bool FrameProcessor::checkBlurUsingCV(FrameData& fd) {
	const int BLOCK = 60;
	Mat resizeFrame, grayFrame;

	resize(fd.frame, resizeFrame, Size(256, 256));
	cvtColor(resizeFrame, grayFrame, COLOR_BGR2GRAY);
	int cx = grayFrame.cols / 2;
	int cy = grayFrame.rows / 2;

	Mat floatImage;
	grayFrame.convertTo(floatImage, CV_32F);

	// FFT
	Mat fourierTransform;
	dft(floatImage, fourierTransform, DFT_SCALE | DFT_COMPLEX_OUTPUT);
	//center low frequencies in the middle
	//by shuffling the quadrants.
	Mat q0(fourierTransform, Rect(0, 0, cx, cy));       // Top-Left - Create a ROI per quadrant
	Mat q1(fourierTransform, Rect(cx, 0, cx, cy));      // Top-Right
	Mat q2(fourierTransform, Rect(0, cy, cx, cy));      // Bottom-Left
	Mat q3(fourierTransform, Rect(cx, cy, cx, cy));     // Bottom-Right

	Mat tmp;
	q0.copyTo(tmp);			// swap quadrants (Top-Left with Bottom-Right)
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);			// swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);

	// Block the low frequencies
	// #define BLOCK could also be a argument on the command line of course
	fourierTransform(Rect(cx - BLOCK, cy - BLOCK, 2 * BLOCK, 2 * BLOCK)).setTo(0);

	//shuffle the quadrants to their original position
	Mat orgFFT;
	fourierTransform.copyTo(orgFFT);
	Mat p0(orgFFT, Rect(0, 0, cx, cy));       // Top-Left - Create a ROI per quadrant
	Mat p1(orgFFT, Rect(cx, 0, cx, cy));      // Top-Right
	Mat p2(orgFFT, Rect(0, cy, cx, cy));      // Bottom-Left
	Mat p3(orgFFT, Rect(cx, cy, cx, cy));     // Bottom-Right

	p0.copyTo(tmp);			// swap quadrants (Top-Left with Bottom-Right)
	p3.copyTo(p0);
	tmp.copyTo(p3);

	p1.copyTo(tmp);			// swap quadrant (Top-Right with Bottom-Left)
	p2.copyTo(p1);
	tmp.copyTo(p2);

	// IFFT
	Mat invFFT;
	Mat logFFT;
	double minVal, maxVal;

	dft(orgFFT, invFFT, DFT_INVERSE | DFT_REAL_OUTPUT);

	invFFT = cv::abs(invFFT);
	cv::minMaxLoc(invFFT, &minVal, &maxVal, NULL, NULL);

	//check for impossible values
	if (maxVal <= 0.0) {
		cerr << "No information, complete black image!\n";
		return 1;
	}

	cv::log(invFFT, logFFT);
	logFFT *= 20;

	cv::Scalar result = cv::mean(logFFT);

	if (result.val[0] > config->blurCutOff) { // 10 for less blure. 20 for more blur.
		return false;	// not blur
	} else {
		return true;	// blur
	}
}

bool FrameProcessor::checkFrozenFrame(FrameData& fd) {
	static bitset<HASH_FRAME_SIZE> lastDistanceHash;
	static list<bool> isSimilarList(config->freezeHistoryDepth, false);

	Mat imgForHash;
	bitset <HASH_FRAME_SIZE> distanceHash;
	bool isFrozen = true;
	resize(fd.frame, imgForHash, Size(HASH_COLUMN_NUMS + 1, HASH_ROW_NUMS), cv::INTER_LINEAR);
	cvtColor(imgForHash, imgForHash, COLOR_BGR2GRAY);
	calcHash(imgForHash, HASH_ROW_NUMS, HASH_COLUMN_NUMS + 1, distanceHash);
	bool isSimilar = hammingDistance(distanceHash, lastDistanceHash) / HASH_FRAME_SIZE < config->freezeThreshold;
	lastDistanceHash = distanceHash;
	isSimilarList.pop_front();
	isSimilarList.push_back(isSimilar);

	for (list<bool>::iterator iter = isSimilarList.begin(); iter != isSimilarList.end(); iter++) {
		if (!isSimilar) {
			isFrozen = false;
			break;
		}
	}
	return isFrozen;
}

void FrameProcessor::saveSnapshot(const cv::Mat& plainScreen, FrameData& detectionData) {
	summary->lastestSnapshotIndex += 1;
	SnapshotInfoElement snapshotInfoElement = {};
	std::string snapshotSaveTime = "--:--:--";
	if (config->snapshotTime.compare("SYSTEM") == 0) {
		snapshotInfoElement.snapshotSaveDateTime = saveSnapshotTime();
		size_t timeOffset = snapshotInfoElement.snapshotSaveDateTime.find(" ") + 1;
		snapshotSaveTime = snapshotInfoElement.snapshotSaveDateTime.substr(timeOffset);
	} else {
		QString timeText = QString("%1:%2:%3")
			.arg(summary->progTimeLog->examinationTime / (60 * 60), 2, 10, QLatin1Char('0'))
			.arg(summary->progTimeLog->examinationTime / (60), 2, 10, QLatin1Char('0'))
			.arg(summary->progTimeLog->examinationTime % (60), 2, 10, QLatin1Char('0'));
		snapshotSaveTime = timeText.toStdString();
		snapshotInfoElement.snapshotSaveDateTime = snapshotSaveTime;
	}

	snapshotInfoElement.detectPolyp = !(detectionData.resultVec.empty());

	if (config->saveDetectionSnapshot) {
		snapshotInfoElement.detectionScreen = cvMatToQImage(detectionData.frame).copy();
	}
	if (config->savePlainScreenSnapshot) {
		snapshotInfoElement.plainScreen = cvMatToQImage(plainScreen).copy();
	}
	if (config->saveFullScreenSnapshot) {
		snapshotInfoElement.fullScreen = QGuiApplication::primaryScreen()->grabWindow(mainWindow->ui->pageExamination->winId()).toImage();
	}

	// show snapshot
	if (config->saveDetectionSnapshot) {
		// a snaposhot of detection frame
		emit newSnapshot(setDrawText(snapshotInfoElement.detectionScreen, snapshotSaveTime, summary->lastestSnapshotIndex));
	} else if (config->savePlainScreenSnapshot) {
		// // a snapshot of plain screen
		emit newSnapshot(setDrawText(snapshotInfoElement.plainScreen, snapshotSaveTime, summary->lastestSnapshotIndex));
	} else if (config->saveFullScreenSnapshot) {
		// a snapshot of SENA.Finder detection page
		emit newSnapshot(setDrawText(snapshotInfoElement.fullScreen, snapshotSaveTime, summary->lastestSnapshotIndex));
	}

	snapshotInfoElement.snapshotNumber = summary->lastestSnapshotIndex;
	summary->snapshotInfoList.push_back(snapshotInfoElement);
}

void FrameProcessor::captureProcess(void) {
	MainWindow* mainUI = mainWindow;
	FrameData capturedFrame;
	size_t poolIdx = 0;
	bool isBlack = false;
	unsigned int frameID = 0;
	int blackCount = 0;
	int frameCount = 0;
	cv::Mat frameInOut;
	int inOutFrameInterval = config->fps / config->inOutCheckFrequency; // 몇 프레임 마다 InOut check
	CHRONO_TIME_POINT capTime = {};

	do {
		// step 0. get a frame from frame pool
		poolIdx = (++poolIdx) % config->framePoolSize;
		Mat& newFrame = framePool[poolIdx];

		// step 1. insert delay when use video file
		if (config->videoTestMode) {
			this_thread::sleep_for(chrono::milliseconds(config->frameIntervalTuning));
		}

		// step 2. read a frame from VideoCapture
		if ((cap && cap->read(newFrame) == false) || stopped == true) {
			// condition 2.1. cannot read new frame from VideoCapture
			// step 2.1.1. re-asign FrameData
			capturedFrame = FrameData();
			capturedFrame.exitFlag = true;
			emit stopExamination();
#if defined(__linux__)
		} else if (capture && capture->read(newFrame) == false || stopped == true) {
			capturedFrame = FrameData();
			capturedFrame.exitFlag = true;
			emit stopExamination();
#endif
		} else {
			// A new frame has been captured.
			capTime = chrono::high_resolution_clock::now(); // frame timestamp
			numOfCaptureFrames++;

			// condition 2.2. read new frame from VideoCapture
			// step 2.2.1. re-asign FrameData
			capturedFrame = FrameData();
			capturedFrame.frameID = ++frameID;

			if (needCrop) {
				if (config->frameCropping) {
					frameCrop = getMainFrameFromMat(newFrame);
				} else {
					frameCrop.coordinate.x = 0;
					frameCrop.coordinate.y = 0;
					frameCrop.coordinate.width = cap->get(cv::CAP_PROP_FRAME_WIDTH);
					frameCrop.coordinate.height = cap->get(cv::CAP_PROP_FRAME_HEIGHT);
					frameCrop.isValid = true;
				}
				needResize = resizeDisplay(frameCrop.coordinate);
				needCrop = false;
				needNewRecord = true;

				if (frameCrop.isValid == false) {
					capturedFrame.exitFlag = true;
					emit stopExamination();
				}
			}

			frameCount += 1;

			// step 2.2.2 black 화면 체크
			if (isBlack || frameCount > config->fps) { // blackframe이면 매 프레임 확인 또는 1초마다 프레임 확인
				isBlack = isBlackFrame(newFrame);
				frameCount = 0;
			}
			// black 화면이면 blackCount & 조건 충족하면 stopvideo
			if (isBlack) {
				blackCount += 1;
				if (blackCount > config->fps) {
					capturedFrame.exitFlag = true;
					emit stopExamination();
				}
			} else {
				// not black 이면 정상 추론 시작
				blackCount = 0;
			}

			// step 2.2.4. crop valid frame area
			if (config->frameCropping == false) {
				capturedFrame.frame = newFrame;
			} else if (config->frameMasking) {
				newFrame = newFrame(frameCrop.coordinate);
				bitwise_and(newFrame, newFrame, capturedFrame.frame, frameCrop.maskFrame(frameCrop.coordinate)); // mask를 씌운 후지논 버전
			} else {
				capturedFrame.frame = newFrame(frameCrop.coordinate); // no mask 올림푸스 버전
			}
			// step 2.2.5 check In out
			if ((frameCount % inOutFrameInterval == 0) && (config->colonoscopyAutoAwareness)) {
				frameInOut = capturedFrame.frame.clone();
				bool updateState = checkInOut(frameInOut, config->inOutHistoryDepth, config->inOutThreshold);

				if (updateState == DECISION_INSIDE && mainUI->ui->pageExamination->isStopwatchOn() == false) {
					emit timerAction(TIMER_START);
				} else if (updateState == DECISION_OUTSIDE && mainUI->ui->pageExamination->isStopwatchOn() == true) {
					emit timerAction(TIMER_PAUSE);
				}
				currentState = updateState;
			}
		}

		capturedFrame.captureTime = capTime;

		fromCaptureToPrepareQueue.put(capturedFrame);
	} while (!capturedFrame.exitFlag);
	cout << "capture thread end" << endl;
}

void FrameProcessor::displayFPS(void) {
	MainWindow* mainUI = mainWindow;
	char dynamicFps[64];
	while (stopped == false) {
		this_thread::sleep_for(1s);
		snprintf(dynamicFps, 64,
			"C:%zu FPS, I:%zu FPS, D:%zu FPS.",
			numOfCaptureFrames.load(),
			numOfInferenceFrames.load(),
			numOfDisplayFrames.load());
		numOfCaptureFrames.store(0);
		numOfInferenceFrames.store(0);
		numOfDisplayFrames.store(0);
		mainUI->ui->pageExamination->labelDynamicFPS->setText(dynamicFps);
	}
	cout << "display thread end" << endl;
}

void FrameProcessor::checkBlurProcess(void) {
	shared_ptr<image_t> detImage;
	FrameData checkBlurDataFrame;

	do {
		checkBlurDataFrame = fromDetectToBlurCheckObject.receive();
		if (checkBlurDataFrame.frameLabel == FrameLabel::CHECK) {
			cout << "checking BLUR..." << endl;
			if (checkBlurUsingCV(checkBlurDataFrame)) {
				checkBlurDataFrame.frameLabel = FrameLabel::BLUR;
			}
			fromBlurCheckToDetectObject.send(checkBlurDataFrame);
		} else {
			// do nothing with frame data
		}
	} while (!checkBlurDataFrame.exitFlag);
	cout << "checkBlur thread end" << endl;
}

void FrameProcessor::prepareProcess(void) {
	shared_ptr<image_t> detImage;
	FrameData preparedData;
	do {
		preparedData = fromCaptureToPrepareQueue.get();

		if (preparedData.exitFlag == false) {
			if (config->framePadding == 0) {
				// case 1. frame padding none (without letterbox)
				preparedData.detImage = detector->mat_to_image_resize(preparedData.frame);
			} else if (config->framePadding == 1) {
				// case 2. frame padding biased (with biased letterbox)
				cv::Mat letterboxed = letterbox(preparedData.frame, 0, true);
				preparedData.detImage = detector->mat_to_image_resize(letterboxed);
			} else if (config->framePadding == 2) {
				// caes 3. frame padding balanced (with balanced letterbox)
				cv::Mat letterboxed = letterbox(preparedData.frame, 0, false);
				preparedData.detImage = detector->mat_to_image_resize(letterboxed);
			} else {
				// error
			}
		}
		fromPrepareToDetectQueue.put(preparedData);    // push to detection thread queue
	} while (!preparedData.exitFlag);
	cout << "prepare thread end" << endl;
}

void FrameProcessor::detectProcess(void) {
	FrameData lastCapturedFrameData;
	FrameData lastInferenceFrameData;
	FrameData currentFrameData;
	FrameData bypassFrameData;
	FrameData checkBlurDataFrame;
	std::vector<bbox_t> lastTrackingResultVec;

	do {
		currentFrameData = fromPrepareToDetectQueue.get();
		if (fromPrepareToDetectQueue.size() > 1) {
			cout << "Data frame queue size : " << fromPrepareToDetectQueue.size() << endl;
		}


		chrono::duration<double, std::milli> inferenceGapMS = currentFrameData.captureTime - lastInferenceFrameData.captureTime;

		if (inferenceGapMS.count() > config->inferenceInterval) {
			//-------------------------------------------------------
			// 1. Bypass current frame to drawing thread first
			//
			cout << "Bypass before Inference." << endl;
			bypassFrameData = currentFrameData;
			bypassFrameData.frameLabel = FrameLabel::BYPASS;
			// reuse last tracking result
			bypassFrameData.resultVec = lastTrackingResultVec;
			bypassFrameData.exitFlag = false; // bypass Frame should not stop the following threads
			bypassDetectToDrawObject.send(bypassFrameData);
			//
			//-------------------------------------------------------

			//-------------------------------------------------------
			// 2. Send frame data to blur checking thread
			//
			checkBlurDataFrame = currentFrameData;
			checkBlurDataFrame.frameLabel = FrameLabel::CHECK;
			fromDetectToBlurCheckObject.send(checkBlurDataFrame);
			//
			//-------------------------------------------------------

			//-------------------------------------------------------
			// 3. do inference
			//
			this->inference(currentFrameData);
			cout << "Inference done." << endl;
			//
			//-------------------------------------------------------

			//-------------------------------------------------------
			// 4. get blur check result
			//
			checkBlurDataFrame = fromBlurCheckToDetectObject.receive();
			//
			//-------------------------------------------------------

			if (currentFrameData.resultVec.size()) {
				if (checkBlurDataFrame.frameLabel == FrameLabel::BLUR) {
					currentFrameData.frameLabel = FrameLabel::DETECTION_WITH_BLUR;
				} else {
					currentFrameData.frameLabel = FrameLabel::DETECTION;
				}
			} else {
				currentFrameData.frameLabel = FrameLabel::INFERENCE;
			}

			if (currentFrameData.frameLabel == FrameLabel::DETECTION_WITH_BLUR) {
				// remove detection result
				currentFrameData.resultVec.clear();
			}

			lastCapturedFrameData = currentFrameData;
			lastInferenceFrameData = currentFrameData;

			this->tracking(currentFrameData);
		} else {
			cout << "Inference skipped." << endl;
			currentFrameData.frameLabel = FrameLabel::SKIP;
			lastCapturedFrameData = currentFrameData;

			//-------------------------------------------------------
			// Send Exit signal to blur checking thread
			// Do not get(receive) a data frame to this sending.
			// This inter-thread communication is not good and should be improved later.
			if (currentFrameData.exitFlag) {
				checkBlurDataFrame = currentFrameData;
				checkBlurDataFrame.frameLabel = FrameLabel::SKIP;
				fromDetectToBlurCheckObject.send(checkBlurDataFrame);
			}

			// reuse last inference result
			currentFrameData.resultVec = lastInferenceFrameData.resultVec;
			this->tracking(currentFrameData);
		}

		// save traking data for next bypass
		lastTrackingResultVec = currentFrameData.resultVec;

		fromDetectToDrawQueue.put(currentFrameData);
		fromDetectToSoundQueue.put(currentFrameData);

	} while (!currentFrameData.exitFlag);
	cout << "detect thread end" << endl;
}

void FrameProcessor::drawProcess(void) {
	MainWindow* mainUI = mainWindow;
	vector<bbox_t> resizedResult;
	cv::Mat resizedFrame;
	FrameData detectionData;
	unsigned int recentFrozenFrame = 0;
	int framePeriod = config->fps / config->freezeFrequencyPerSecond;

	do {
		if (bypassDetectToDrawObject.isObjectPresent()) {			// process BYPASS frame
			detectionData = bypassDetectToDrawObject.receive();
		} else if(fromDetectToDrawQueue.size() > 0) {				// process INFERENCE or DETECT or SKIP frame
			detectionData = fromDetectToDrawQueue.get();
		} else {													// busy-waiting
			this_thread::sleep_for(chrono::milliseconds(1));
			continue;
		}

		int frameWidth = detectionData.frame.size().width;
		int frameHeight = detectionData.frame.size().height;

		if (detectionData.exitFlag == false) {
			if (detectionData.frame.data) {
				if (config->frameResizing || needResize) {
					resizeFrame(detectionData.frame, detectionData.frame);
				}
			}

			// snapshot plain, detection, sena screen
			bool snapshotTrigger = false;
			cv::Mat plainScreen;

			// freeze trigger
			if (detectionData.frameID % framePeriod == 0) {

				bool isFrozen = checkFrozenFrame(detectionData);

				if (isFrozen && detectionData.frameID - recentFrozenFrame > framePeriod * config->freezeKeepingDistance) {
					snapshotTrigger = true;
					if (config->savePlainScreenSnapshot) {
						plainScreen = detectionData.frame.clone();
					}
				}
				if (isFrozen) {
					recentFrozenFrame = detectionData.frameID;
				}
			}

			// draw boxes
			if (config->isDetectable) {
				// resize result vec 
				resizedResult.clear();
				double ratioWidth = static_cast<double>(detectionData.frame.size().width) / frameWidth;
				double ratioHeight = static_cast<double>(detectionData.frame.size().height) / frameHeight;

				for (auto it = detectionData.resultVec.begin(); it != detectionData.resultVec.end();) {
					bbox_t res = {};
					res.x = (*it).x * ratioWidth;
					res.y = (*it).y * ratioHeight;
					res.w = (*it).w * ratioWidth;
					res.h = (*it).h * ratioHeight;
					res.frames_counter = (*it).frames_counter;
					res.track_id = (*it).track_id;
					resizedResult.push_back(res);
					it++;
				}
				drawBoxes(detectionData.frame, resizedResult);
			}

			// get snapshot 
			if (snapshotTrigger) {
				saveSnapshot(plainScreen, detectionData);
			}
		}

		fromDrawToShowQueue.put(detectionData);
		numOfDisplayFrames++;

		if (config->recordVideo) {
			fromDrawToRecordQueue.put(detectionData);
		}
	} while (!detectionData.exitFlag);
	cout << "draw thread end" << endl;
}

void FrameProcessor::soundProcess(void) {
	FrameData detectionData;
	typedef tuple<unsigned int, unsigned int, bbox_t> BboxTuple; //frameID, track_id, detectedBox
	typedef vector<BboxTuple> BboxTupleVec;
	BboxTupleVec trackingData, expiredData;

	unsigned int soundThres = config->soundMinimumHitCount;

	do {
		detectionData = fromDetectToSoundQueue.get();

		if (config->isSoundable == false) {
			trackingData.clear();
			expiredData.clear();
		}

		if (detectionData.exitFlag) {
			break;
		}

		if (detectionData.resultVec.empty()) {
			continue;
		}
		bool isNew = true;

		// step 1. determine each boxes in resultVec is newly detected or not.
		for (bbox_t& detectedBox : detectionData.resultVec) {
			if (detectedBox.frames_counter < soundThres) {
				isNew = false;
				continue;
			}

			isNew = true;
			BboxTupleVec::iterator IDCheckforTrackingData = find_if(trackingData.begin(), trackingData.end(), [=](BboxTuple bboxTuple) -> bool {return get<1>(bboxTuple) == detectedBox.track_id; });
			BboxTupleVec::iterator IDCheckforExpiredData = find_if(expiredData.begin(), expiredData.end(), [=](BboxTuple bboxTuple) -> bool {return get<1>(bboxTuple) == detectedBox.track_id; });

			// step 1.2. determine a detectedBox is new or not
			// 'if' case for first detection / 'else if' case for detection with same trackingID(so not a new one)
			if (trackingData.empty() && expiredData.empty()) {
				trackingData.push_back(make_tuple(detectionData.frameID, detectedBox.track_id, detectedBox));
			} else if (IDCheckforTrackingData != trackingData.end()) {
				trackingData.erase(IDCheckforTrackingData);
				isNew = false;
			} else if (IDCheckforExpiredData != expiredData.end()) {
				expiredData.erase(IDCheckforExpiredData);
				isNew = false;
			} else {
				TrackingBox detectedRect;
				detectedRect.box = Rect_<float>(Point_<float>(detectedBox.x, detectedBox.y), Point_<float>(detectedBox.x + detectedBox.w, detectedBox.y + detectedBox.h));
				for (BboxTupleVec::iterator iter = trackingData.begin(); iter != trackingData.end(); iter++) {
					bbox_t trackingBox = get<2>(*iter);
					TrackingBox trackingRect;
					trackingRect.box = Rect_<float>(Point_<float>(trackingBox.x, trackingBox.y), Point_<float>(trackingBox.x + trackingBox.w, trackingBox.y + trackingBox.h));
					double IOU = getIOU(detectedRect.box, trackingRect.box);
					if (IOU >= config->iouThreshold) {
						isNew = false;
					}
				}

				if (isNew) {
					for (BboxTupleVec::iterator iter = expiredData.begin(); iter != expiredData.end(); iter++) {
						bbox_t expiredBox = get<2>(*iter);
						TrackingBox expiredRect;
						expiredRect.box = Rect_<float>(Point_<float>(expiredBox.x, expiredBox.y), Point_<float>(expiredBox.x + expiredBox.w, expiredBox.y + expiredBox.h));
						double IOU = getIOU(detectedRect.box, expiredRect.box);
						if (IOU >= config->iouThreshold) {
							isNew = false;
						}
					}
				}
			}
			trackingData.push_back(make_tuple(detectionData.frameID, detectedBox.track_id, detectedBox));

			if (isNew) {
				soundEffector.play();
			}
		}

		// delete data which is not appearing for recent 'minimumHitCount' frames from the trackingData and send this data to expiredData.
		for (BboxTupleVec::iterator trackingIter = trackingData.begin(); trackingIter != trackingData.end();) {
			if (detectionData.frameID - get<0>(*trackingIter) > config->minimumHitCount) {
				bbox_t trackingBox = get<2>(*trackingIter);
				TrackingBox trackingRect;
				trackingRect.box = Rect_<float>(Point_<float>(trackingBox.x, trackingBox.y), Point_<float>(trackingBox.x + trackingBox.w, trackingBox.y + trackingBox.h));

				for (BboxTupleVec::iterator expiredIter = expiredData.begin(); expiredIter != expiredData.end();) {
					bbox_t expiredBox = get<2>(*expiredIter);
					TrackingBox expiredRect;
					expiredRect.box = Rect_<float>(Point_<float>(expiredBox.x, expiredBox.y), Point_<float>(expiredBox.x + expiredBox.w, expiredBox.y + expiredBox.h));
					double IOU = getIOU(trackingRect.box, expiredRect.box);
					// if the tracking data intended to be sent to expiredData has large IOU with data in expiredData, the old one in expiredData will be deleted and new data from trackingData will be added.
					if (get<1>(*trackingIter) == get<1>(*expiredIter) || IOU >= config->iouThreshold) {
						expiredIter = expiredData.erase(expiredIter);
					} else {
						expiredIter++;
					}
				}
				expiredData.push_back(*trackingIter);
				trackingIter = trackingData.erase(trackingIter);
			} else {
				trackingIter++;
			}
		}

		// remain just 'historyDepth' numbers of data in expiredData.
		if (expiredData.size() > config->soundHistoryDepth) {
			sort(expiredData.begin(), expiredData.end(), [](BboxTuple t1, BboxTuple t2) {return get<0>(t1) > get<0>(t2); });
			expiredData.erase(expiredData.begin() + config->soundHistoryDepth, expiredData.end());
		}

		// remove expired historical frame
		for (BboxTupleVec::iterator expiredIter = expiredData.begin(); expiredIter != expiredData.end();) {
			if (detectionData.frameID - get<0>(*expiredIter) > (config->fps * config->soundHistoryLifespan)) {
				expiredIter = expiredData.erase(expiredIter);
			} else {
				expiredIter++;
			}
		}
	} while (!detectionData.exitFlag);
	cout << "sound thread end" << endl;
}

void FrameProcessor::recordProcess(void) {
	map<unsigned int, tuple<unsigned int, string>> trackingMap;
	map<unsigned int, tuple<unsigned int, string, string>> bboxMap;
	struct tm* t = localtime(&summary->sysTimeLog->beginTime);
	char filename[256] = {};
	sprintf(filename, "%s/video/%04d-%02d-%02d_%02d-%02d-%02d.mp4", summary->examinationOutputDataPath.c_str(), t->tm_year + 1900, t->tm_mon + 1, t->tm_mday, t->tm_hour, t->tm_min, t->tm_sec);

	//save video
	VideoWriter outputVideo;
	int videoWidth = 0;
	int videoHeight = 0;
	int videoFps = 0;
	int lastVideoWidth = 0;
	int lastVideoHeight = 0;

	FrameData detectionData;
	do {
		if (needNewRecord || videoWidth == 0) {
			needNewRecord = false;
			if (needResize) {
				double rateWidth = (double)frameCrop.coordinate.width / (double)displayLabel->width();
				double rateHeight = (double)frameCrop.coordinate.height / (double)displayLabel->height();

				if (rateWidth > rateHeight) {
					// case 1. fit to width
					videoWidth = displayLabel->width();
					videoHeight = frameCrop.coordinate.height / rateWidth;
				} else {
					// case 2. fit to height
					videoHeight = displayLabel->height();
					videoWidth = frameCrop.coordinate.width / rateHeight;
				}
			} else if (config->frameCropping) {
				videoWidth = frameCrop.coordinate.width;
				videoHeight = frameCrop.coordinate.height;
			} else {
				videoWidth = cap->get(cv::CAP_PROP_FRAME_WIDTH);
				videoHeight = cap->get(cv::CAP_PROP_FRAME_HEIGHT);
			}
			videoFps = cap->get(cv::CAP_PROP_FPS);

			if (summary->recordingIdx == 0) {
				sprintf(filename, "%s/video/%04d-%02d-%02d_%02d-%02d-%02d.mp4", summary->examinationOutputDataPath.c_str(), t->tm_year + 1900, t->tm_mon + 1, t->tm_mday, t->tm_hour, t->tm_min, t->tm_sec);
			} else {
				outputVideo.release();
				sprintf(filename, "%s/video/%04d-%02d-%02d_%02d-%02d-%02d(%d).mp4", summary->examinationOutputDataPath.c_str(), t->tm_year + 1900, t->tm_mon + 1, t->tm_mday, t->tm_hour, t->tm_min, t->tm_sec, summary->recordingIdx);
			}
			bool isOpened = outputVideo.open(filename, VideoWriter::fourcc('X', 'V', 'I', 'D'), videoFps, Size(videoWidth, videoHeight), true);
			summary->recordingIdx++;
			lastVideoWidth = videoWidth;
			lastVideoHeight = videoHeight;
		}

		detectionData = fromDrawToRecordQueue.get();
		if (outputVideo.isOpened() && detectionData.exitFlag == false) {
			outputVideo << detectionData.frame;
		}

	} while (!detectionData.exitFlag);
	outputVideo.release();
	cout << "record thread end" << endl;
}

void FrameProcessor::run(void) {
	isRunning = true;
	stopped = false;
	numOfCaptureFrames = 0;
	numOfInferenceFrames = 0;
	numOfDisplayFrames = 0;
	MainWindow* mainUI = mainWindow;
	resetHsvData();

#if defined(SINGLE_FRAMETIMER)
	setFrameTimeLogFileName();
#endif
	mainWindow->ui->pageExamination->showFullscreenModal("Load AI model");
	// step 1. wait for detector creation
	while (stopped == false && detector == NULL) {
		std::this_thread::sleep_for(100ms);
	}
	if (stopped) {
		isRunning = false;
		mainWindow->ui->pageExamination->hideFullscreenModal();
		return;
	}
	mainWindow->ui->pageExamination->showFullscreenModal("Open video capturing device");

	initializeFrame();

	mainWindow->ui->pageExamination->hideFullscreenModal();

	if (config->showFps) {
		tFps = std::thread(&FrameProcessor::displayFPS, this);
	}

	// capture new video-frame
	tCap = std::thread(&FrameProcessor::captureProcess, this);

	// pre-processing video frame (resize, convertion)
	tPrepare = std::thread(&FrameProcessor::prepareProcess, this);

	// check blur
	tCheckBlur = std::thread(&FrameProcessor::checkBlurProcess, this);

	// detection by yolo and sort algorithm
	tDetect = std::thread(&FrameProcessor::detectProcess, this);

	// draw rectangles (and track objects)
	tDraw = std::thread(&FrameProcessor::drawProcess, this);

	// sound processing
	tSound = std::thread(&FrameProcessor::soundProcess, this);

	// write frame to videofile
	if (config->recordVideo) {
		cout << "Initialize video recording" << endl;
		tRecord = std::thread(&FrameProcessor::recordProcess, this);
	}

	// show detection
	FrameData detectionData;

#if defined(TIME_LOG)
	time_t timer = time(NULL);
	struct tm* logtimeBegin = localtime(&timer);
	char logtimeBeginStr[256];
	struct tm* logtimeCurrent;
	sprintf(logtimeBeginStr, "\nbegin: %04d-%02d-%02d %02d:%02d:%02d\nsize_t", logtimeBegin->tm_year + 1900, logtimeBegin->tm_mon + 1, logtimeBegin->tm_mday, logtimeBegin->tm_hour, logtimeBegin->tm_min, logtimeBegin->tm_sec);
#endif

	do {
		detectionData = fromDrawToShowQueue.get();

		pixmap = QPixmap::fromImage(cvMatToQImage(detectionData.frame));
		emit newPixmapCaptured();

		if (detectionData.exitFlag == false) {

#if defined(TIME_LOG)
			timer = time(NULL);
			struct tm* logtimeCurrent = localtime(&timer);
			printf("%scurrent: %04d-%02d-%02d %02d:%02d:%02d\n", logtimeBeginStr, logtimeCurrent->tm_year + 1900, logtimeCurrent->tm_mon + 1, logtimeCurrent->tm_mday, logtimeCurrent->tm_hour, logtimeCurrent->tm_min, logtimeCurrent->tm_sec);
#endif
		}
	} while (!detectionData.exitFlag);

	if (tFps.joinable()) {
		tFps.join();
		cout << "tFPS join!" << endl;
	}
	if (tCap.joinable()) {
		tCap.join();
		cout << "tCap join!" << endl;
	}
	if (tPrepare.joinable()) {
		tPrepare.join();
		cout << "tPrepare join!" << endl;
	}
	if (tCheckBlur.joinable()) {
		tCheckBlur.join();
		cout << "tCheckBlur join!" << endl;
	}
	if (tDetect.joinable()) {
		tDetect.join();
		cout << "tDetect join!" << endl;
	}
	if (tDraw.joinable()) {
		tDraw.join();
		cout << "tDraw join!" << endl;
	}
	if (tSound.joinable()) {
		tSound.join();
		cout << "tSound join!" << endl;
	}
	if (tRecord.joinable()) {
		tRecord.join();
		cout << "tRecord join!" << endl;
	}

	// clear display
	emit clearPixmap();

	delete[] framePool;
	if (cap) {
		cap->release();
		delete cap;
	}
	if (capture) {
#if defined(__linux__)
		capture->StopDeckLinkCapture();
#else
		delete capture;
#endif
	}
	isRunning = false;
	resetHsvData();
}

void FrameProcessor::stopVideo(void) {
	cout << "===================== STOP PROCESSING VIDEO =====================" << endl;
	stopped = true;
}

QImage FrameProcessor::cvMatToQImage(const cv::Mat & inMat) {
	switch (inMat.type()) {
	case CV_8UC4: // 8-bit, 4 channel
	{
		QImage image(inMat.data, inMat.cols, inMat.rows, static_cast<int>(inMat.step), QImage::Format_ARGB32);
		return image;
	}
	case CV_8UC3: // 8-bit, 3 channel
	{
		return QImage(inMat.data, inMat.cols, inMat.rows, static_cast<int>(inMat.step), QImage::Format_BGR888);
	}
	case CV_8UC1: // 8-bit, 1 channel
	{
#if QT_VERSION >= QT_VERSION_CHECK(5, 5, 0)
		QImage image(inMat.data, inMat.cols, inMat.rows, static_cast<int>(inMat.step), QImage::Format_Grayscale8);
#else
		static QVector<QRgb>  sColorTable;

		// only create our color table the first time
		if (sColorTable.isEmpty()) {
			sColorTable.resize(256);
			for (int i = 0; i < 256; ++i) {
				sColorTable[i] = qRgb(i, i, i);
			}
		}

		QImage image(inMat.data, inMat.cols, inMat.rows, static_cast<int>(inMat.step), QImage::Format_Indexed8);

		image.setColorTable(sColorTable);
#endif
		return image;
			}
	default:
		qWarning() << "ASM::cvMatToQImage() - cv::Mat image type not handled in switch:" << inMat.type();
		break;
		}

	return QImage();
}

QPixmap FrameProcessor::cvMatToQPixmap(const cv::Mat & inMat) {
	return QPixmap::fromImage(cvMatToQImage(inMat));
	}

void FrameProcessor::reCropPixmap(void) {
	needCrop = true;
}

bool FrameProcessor::resizeDisplay(cv::Rect& displaySize) {
	bool needResize = false;
	if ((double)(config->frameWidth) / (double)(config->frameHeight) < 1.7) {
		// TODO: Remove 4 to 3 Mode
#if defined(WITH_OVERLAY)
		// resize frame area with overlay
		displayLabel->setGeometry(QRect(0, 0, config->frameWidth, config->frameHeight));
#else
		// resize frame area without overlay
		if (displaySize.width > (config->frameWidth - (config->frameWidth * 0.1823 + config->frameHeight * 0.0865))) {
			needResize = true;
		}
		displayLabel->setGeometry(QRect(0, 0, config->frameWidth - (config->frameWidth * 0.1823 + config->frameHeight * 0.0865), config->frameHeight * 1.0000));
#endif
	} else if (config->frameResizing == false) {
		int posX = displaySize.width;
		int posY = displaySize.height;
		int width = displaySize.width;
		int height = displaySize.height;
#if defined(WITH_OVERLAY)
		// resize frame area with overlay
		if (posX <= config->frameWidth) {
#else
		// resize frame area without overlay
		if (posX < config->frameWidth * 0.1302) {
#endif
			posX = (config->frameWidth * 0.1302) + (((config->frameWidth * 0.6500) - posX) / 2);
		} else {
			posX = config->frameWidth * 0.1302;
			width = config->frameWidth * 0.6500;
			needResize = true;
		}
#if defined(WITH_OVERLAY)
		// resize frame area with overlay
		if (posX < 0) {
			posX = 0;
		}
#else
#endif
		if (posY < config->frameHeight) {
			posY = (config->frameHeight - posY) / 2;
		} else if (posY == config->frameHeight) {
			posY = 0;
			height = config->frameHeight;
		} else {
			posY = 0;
			height = config->frameHeight;
			needResize = true;
		}
		displayLabel->setGeometry(QRect(posX, posY, width, height));
	} else if (config->frameResizing) {
		displayLabel->setGeometry(QRect(config->frameWidth * 0.1302, config->frameHeight * 0.0000, config->frameWidth * 0.6500, config->frameHeight * 1.0000));
		needResize = true;
	}

	return needResize;
}

void FrameProcessor::resizeFrame(cv::InputArray src, cv::OutputArray dst, double fx, double fy) {
	int srcWidth = src.cols();
	int srcHeight = src.rows();
	int dstWidth = 0;
	int dstHeight = 0;
	double rateWidth = (double)srcWidth / (double)displayLabel->width();
	double rateHeight = (double)srcHeight / (double)displayLabel->height();

	if (rateWidth > rateHeight) {
		// case 1. fit to width
		dstWidth = displayLabel->width();
		dstHeight = srcHeight / rateWidth;
	} else {
		// case 2. fit to height
		dstHeight = displayLabel->height();
		dstWidth = srcWidth / rateHeight;
	}

	resize(src, dst, Size(dstWidth, dstHeight), fx, fy, CV_INTER_LINEAR);
}

void FrameProcessor::setDetector(int level) {
	// step 1. wait for existing thread
	if (threadCreateDetector.joinable()) {
		threadCreateDetector.join(); // why waiting?
	}
	// step 2. delete past detector
	if (detector) {
		delete detector;
		detector = NULL;
	}
	// step 3. select path and thresh
	static string weightsFile;
	switch (level) {
	case 0: // standard mode
		weightsFile = config->standardWeightFilePath;
		FrameProcessor::thresh = config->standardThreshold;
		break;
	case 1: // performance mode
		weightsFile = config->performanceWeightFilePath;
		FrameProcessor::thresh = config->performanceThreshold;
		break;
	}

	// step 4. create new detector
	threadCreateDetector = std::thread([&]() {
		FrameProcessor::detector = new Detector(config->configFilePath, weightsFile);
		});
}

bool FrameProcessor::checkInOut(cv::Mat & frame, int windowSize, float threshold) {
	cv::Mat dstImg;
	cv::resize(frame, dstImg, cv::Size(300, 300));
	cv::cvtColor(dstImg, dstImg, COLOR_BGR2HSV);
	vector<Mat> channels;
	cv::split(dstImg, channels);
	float hueChannel = cv::mean(channels[0])[0];
	float saturationChannel = cv::mean(channels[1])[0];
	float valueChannel = cv::mean(channels[2])[0];

	if (hsv.hue.size() != windowSize) {
		// queue 원소 추가
		hsv.hue.push_back(hueChannel);
		hsv.saturation.push_back(saturationChannel);
		hsv.value.push_back(valueChannel);

		hsv.hueSum += hueChannel;
		hsv.saturationSum += saturationChannel;
		hsv.valueSum += valueChannel;

		hsv.decision = -1;
		return DECISION_OUTSIDE;
	} else {
		// queue update : pop & push
		hsv.hueSum += (hueChannel - hsv.hue.front());
		hsv.hue.pop_front();
		hsv.hue.push_back(hueChannel);

		hsv.saturationSum += (saturationChannel - hsv.saturation.front());
		hsv.saturation.pop_front();
		hsv.saturation.push_back(saturationChannel);

		hsv.valueSum += (valueChannel - hsv.value.front());
		hsv.value.pop_front();
		hsv.value.push_back(valueChannel);

		if ((int)(hsv.hueSum) == 0) {
			return DECISION_OUTSIDE;
		} else {
			hsv.decision = ((hsv.saturationSum + hsv.valueSum) / hsv.hueSum);
			//  (S + V) / H
			if (hsv.decision > threshold) {
				return DECISION_INSIDE;
			} else {
				return DECISION_OUTSIDE;
			}
		}
	}
}

void FrameProcessor::resetHsvData(void) {
	//hsv struct initialize
	hsv.hueSum = 0;
	hsv.saturationSum = 0;
	hsv.valueSum = 0;
	hsv.decision = 0;
	hsv.hue.clear();
	hsv.saturation.clear();
	hsv.value.clear();
}

QImage FrameProcessor::setDrawText(QImage & img, string str, int numFrame) {
	QImage sourceImage = img.scaled(config->frameWidth * 0.1000, config->frameWidth * 0.1000, Qt::KeepAspectRatio);
	QPainter painter(&sourceImage);
	painter.setPen(QColor("#88ffff00"));
	painter.setFont(QFont("Consolas", config->frameWidth * 0.0130));
	painter.drawText(sourceImage.rect(), Qt::AlignHCenter | Qt::AlignBottom, QString::fromStdString(str));
	painter.setFont(QFont("Consolas", config->frameWidth * 0.0400));
	painter.drawText(sourceImage.rect(), Qt::AlignHCenter | Qt::AlignTop, QString::number(numFrame));

	return sourceImage;
}

bool FrameProcessor::isStateInside(void) {
	return currentState;
}

#if defined(SINGLE_FRAMETIMER)
void FrameProcessor::setFrameTimeLogFileName(void) {
	time_t timer = time(NULL);
	struct tm* t = localtime(&timer);

	// timer test code (log file name)
	sprintf(frametimeLogFileName, "frame_time_%04d-%02d-%02d_%02d-%02d.csv", t->tm_year + 1900, t->tm_mon + 1, t->tm_mday, t->tm_hour, t->tm_min);

	FILE* fp = fopen(frametimeLogFileName, "w");
	fprintf(fp, "begin,end,elapsed,cap,prepare,detection,draw,show\n");
	fclose(fp);
}

void  FrameProcessor::putFrameTimeLog(FrameTime & log) {
	FILE* fp = fopen(frametimeLogFileName, "a");
	fprintf(fp, "%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld\n", \
		log.begin, \
		log.end, \
		log.end - log.begin, \
		log.beginPrepare - log.begin, \
		log.beginDetection - log.beginPrepare, \
		log.beginDraw - log.beginDetection, \
		log.beginShow - log.beginDraw, \
		log.end - log.beginShow
	);
	fclose(fp);
}
#endif

