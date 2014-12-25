package com.example.textdetectadv;

import java.util.ArrayList;
import java.util.List;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.KeyPoint;
import org.opencv.imgproc.Imgproc;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;

public class MainActivity extends Activity implements CvCameraViewListener2 {
	private static final String TAG = "Textdecetion::Activity";
	private CameraBridgeViewBase mOpenCvCameraView;
	private Button mButton;
	private boolean isProcess = false;
	private Mat mRgba;
	private Mat mGray;
	private Mat mByte;
	private Scalar CONTOUR_COLOR;
	private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
		@Override
		public void onManagerConnected(int status) {
			switch (status) {
			case LoaderCallbackInterface.SUCCESS: {
				Log.i(TAG, "OpenCV loaded successfully");
				mOpenCvCameraView.enableView();
			}
				break;
			default: {
				super.onManagerConnected(status);
			}
				break;
			}
		}
	};

	public MainActivity() {
		Log.i(TAG, "Instantiated new " + this.getClass());
	}

	/** Called when the activity is first created. */
	@Override
	public void onCreate(Bundle savedInstanceState) {
		Log.i(TAG, "called onCreate");
		super.onCreate(savedInstanceState);
		getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

		setContentView(R.layout.activity_main);

		mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.textdetetion_view);
		mOpenCvCameraView.setCvCameraViewListener(this);
		mButton = (Button) findViewById(R.id.detetionbutton);
		mButton.setOnClickListener(new View.OnClickListener() {

			@Override
			public void onClick(View v) {
				// TODO Auto-generated method stub
				isProcess = !isProcess;
			}
		});
	}

	@Override
	public void onPause() {
		super.onPause();
		if (mOpenCvCameraView != null)
			mOpenCvCameraView.disableView();
	}

	@Override
	public void onResume() {
		super.onResume();
		OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_10, this,
				mLoaderCallback);
	}

	public void onDestroy() {
		super.onDestroy();
		if (mOpenCvCameraView != null)
			mOpenCvCameraView.disableView();
	}

	public void onCameraViewStarted(int width, int height) {
		mRgba = new Mat(height, width, CvType.CV_8UC3);
		mByte = new Mat(height, width, CvType.CV_8UC1);
	}

	public void onCameraViewStopped() {
		// Explicitly deallocate Mats
		mRgba.release();
	}

	public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

		//
		mRgba = inputFrame.rgba();
		mGray = inputFrame.gray();
		CONTOUR_COLOR = new Scalar(255);
		MatOfKeyPoint keypoint = new MatOfKeyPoint();
		List<KeyPoint> listpoint = new ArrayList<KeyPoint>();
		KeyPoint kpoint = new KeyPoint();
		Mat mask = Mat.zeros(mGray.size(), CvType.CV_8UC1);
		int rectanx1;
		int rectany1;
		int rectanx2;
		int rectany2;

		//
		Scalar zeos = new Scalar(0, 0, 0);
		List<MatOfPoint> contour1 = new ArrayList<MatOfPoint>();
		List<MatOfPoint> contour2 = new ArrayList<MatOfPoint>();
		Mat kernel = new Mat(1, 50, CvType.CV_8UC1, Scalar.all(255));
		Mat morbyte = new Mat();
		Mat hierarchy = new Mat();

		Rect rectan2 = new Rect();//
		Rect rectan3 = new Rect();//
		int imgsize = mRgba.height() * mRgba.width();
		//
		if (isProcess) {
			FeatureDetector detector = FeatureDetector
					.create(FeatureDetector.MSER);
			detector.detect(mGray, keypoint);
			listpoint = keypoint.toList();
			//
			for (int ind = 0; ind < listpoint.size(); ind++) {
				kpoint = listpoint.get(ind);
				rectanx1 = (int) (kpoint.pt.x - 0.5 * kpoint.size);
				rectany1 = (int) (kpoint.pt.y - 0.5 * kpoint.size);
				// rectanx2 = (int) (kpoint.pt.x + 0.5 * kpoint.size);
				// rectany2 = (int) (kpoint.pt.y + 0.5 * kpoint.size);
				rectanx2 = (int) (kpoint.size);
				rectany2 = (int) (kpoint.size);
				if (rectanx1 <= 0)
					rectanx1 = 1;
				if (rectany1 <= 0)
					rectany1 = 1;
				if ((rectanx1 + rectanx2) > mGray.width())
					rectanx2 = mGray.width() - rectanx1;
				if ((rectany1 + rectany2) > mGray.height())
					rectany2 = mGray.height() - rectany1;
				Rect rectant = new Rect(rectanx1, rectany1, rectanx2, rectany2);
				Mat roi = new Mat(mask, rectant);
				roi.setTo(CONTOUR_COLOR);

			}
			/*
			 * Imgproc.findContours(mask, contour1, hierarchy,
			 * Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
			 * 
			 * 
			 * for (int ind = 0; ind < contour1.size(); ind++) { rectan2 =
			 * Imgproc.boundingRect(contour1.get(ind)); if (rectan2.area() > 0.3
			 * * imgsize || rectan2.area() < 50 || (rectan2.width /
			 * rectan2.height) > 3 || (rectan2.width / rectan2.height) < 0.1) {
			 * Mat roi = new Mat(mask, rectan2); roi.setTo(zeos);
			 * 
			 * } }
			 */
			Imgproc.morphologyEx(mask, morbyte, Imgproc.MORPH_DILATE, kernel);
			Imgproc.findContours(morbyte, contour2, hierarchy,
					Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
			for (int ind = 0; ind < contour2.size(); ind++) {
				rectan3 = Imgproc.boundingRect(contour2.get(ind));
				if (rectan3.area() > 0.5 * imgsize || rectan3.area() < 100
						|| rectan3.width / rectan3.height < 2) {
					Mat roi = new Mat(morbyte, rectan3);
					roi.setTo(zeos);

				} else
					Core.rectangle(mRgba, rectan3.br(), rectan3.tl(),
							CONTOUR_COLOR);
			}

			return mRgba;
		}
		/*
		 * Features2d.drawKeypoints(mGray, keypoint, output, new Scalar(2, 254,
		 * 255), Features2d.DRAW_RICH_KEYPOINTS);
		 */
		// DescriptorExtractor
		// descriptor=DescriptorExtractor.create(DescriptorExtractor.SIFT);
		// descriptor.compute(mRgba, keypoint, mask);

		return mRgba;
	}
}
