package com.example.textdetectadv;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

public class CornerAnalysis {
	Scalar zeos = new Scalar(0, 0, 0);
	List<MatOfPoint> contour = new ArrayList<MatOfPoint>();
	Mat kernel = new Mat(1, 20, CvType.CV_8UC1, Scalar.all(255));
	Mat morbyte = new Mat();
	Mat hierarchy = new Mat();
	Rect rectan1 = new Rect();
	Rect rectan2 = new Rect();

	public Mat process(Mat mbyte) {
		int imgsize = mbyte.height() * mbyte.width();
		Imgproc.findContours(mbyte, contour, hierarchy, Imgproc.RETR_EXTERNAL,
				Imgproc.CHAIN_APPROX_NONE);
		// Mat filter_mbyte=new
		// Mat(mbyte.width(),mbyte.height(),mbyte.channels());
		for (int ind = 0; ind < contour.size(); ind++) {
			rectan1 = Imgproc.boundingRect(contour.get(ind));
			if (rectan1.area() > 0.3 * imgsize || rectan1.area() < 50
					|| (rectan1.width / rectan1.height) > 3
					|| (rectan1.width / rectan1.height) < 0.1) {
				Mat roi = new Mat(mbyte, rectan1);
				roi.setTo(zeos);

			}
		}
		Imgproc.morphologyEx(mbyte, morbyte, Imgproc.MORPH_DILATE, kernel);
		Imgproc.findContours(morbyte, contour, hierarchy,
				Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
		for (int ind = 0; ind < contour.size(); ind++) {
			rectan2 = Imgproc.boundingRect(contour.get(ind));
			if (rectan2.area() > 0.5 * imgsize || rectan2.area() < 200
					|| rectan2.width / rectan2.height > 3) {
				Mat roi = new Mat(mbyte, rectan1);
				roi.setTo(zeos);

			}
		}
		return mbyte;

	}
}
