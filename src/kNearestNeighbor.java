import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;

/**
 * Implementation of the K Nearest Neighbor Classifier, ties revolved randomly.
 * 
 * @author Wenlyu Ye
 */
public class kNearestNeighbor {

	static final int DIMENSION = 784;
	static List<FeatureVector> trainingData;

	/**
	 * The classifier of KNN with prediction rule trained on training data set.
	 * 
	 * @param k - parameter for the K-Nearest Neighbor Problem
	 * @param x - unseen input data point from training/validation/test data set
	 * @return - predicted label for x from the trained classifier
	 */
	private static int kNNClassifier(int k, FeatureVector x) {

		// this is the approach of finding smallest k elements using a max heap
		PriorityQueue<FeatureVector> knnMaxHeap = new PriorityQueue<>(k, new Comparator<FeatureVector>() {
			public int compare(FeatureVector v1, FeatureVector v2) {
				return (dist(v1, x) > dist(v2, x)) ? -1 : 1;
			}
		});

		// find the k nearest neighbors in the training data with respect to x
		int i = 0;
		for (FeatureVector trainingDataPt : trainingData) {
			if (i < k) {
				knnMaxHeap.offer(trainingDataPt);
				i++;
			} else {
				if (dist(trainingDataPt, x) < dist(knnMaxHeap.peek(), x)) {
					knnMaxHeap.poll();
					knnMaxHeap.offer(trainingDataPt);
					i++;
				}
			}
		}

		// retrieve the labels of the k nearest neighbors and compute their mode
		List<Integer> labels = new ArrayList<>();
		for (FeatureVector vector : knnMaxHeap) {
			labels.add(vector.getLabel());
		}

		return modeOf(labels);
	}

	/**
	 * Return the mode of input integer array, ties resolved in random
	 * 
	 * @param data
	 * @return
	 */
	private static int modeOf(List<Integer> data) {

		// compute the frequencies of each element
		Map<Integer, Integer> dataMap = new HashMap<>();
		for (Integer elem : data) {
			if (dataMap.containsKey(elem)) {
				dataMap.put(elem, dataMap.get(elem) + 1);
			} else {
				dataMap.put(elem, 0);
			}
		}
		Set<Map.Entry<Integer, Integer>> dataSet = dataMap.entrySet();

		// find the maximum frequency
		int maximum = Integer.MIN_VALUE;
		for (Map.Entry<Integer, Integer> entry : dataSet) {
			if (entry.getValue() > maximum) {
				maximum = entry.getValue();
			}
		}

		// filter labels with maximum frequency
		List<Integer> modes = new ArrayList<>();
		for (Map.Entry<Integer, Integer> entry : dataSet) {
			if (entry.getValue() == maximum) {
				modes.add(entry.getKey());
			}
		}

		// resolve ties by randomly choosing a mode
		Random rand = new Random(System.currentTimeMillis());
		return modes.get(rand.nextInt(modes.size()));
	}

	/**
	 * Return the square of Euclidean Distance of two vectors
	 * 
	 * @param x1
	 * @param x2
	 * @return $dist(x1, x2) = \sum_{i=0}^DIMENSION (x1_i, x2_i)^2$
	 */
	private static int dist(FeatureVector x1, FeatureVector x2) {
		int sumOfSquareDiff = 0;
		for (int i = 0; i < DIMENSION; i++) {
			int diff = x1.getCoordinate()[i] - x2.getCoordinate()[i];
			sumOfSquareDiff += diff * diff;
		}
		return sumOfSquareDiff;
	}

	/**
	 * This function parses the input data set file into a List of FeatureVectors.
	 * 
	 * @param dataFile
	 * @param emptyDataSet - container list for data
	 * @throws FileNotFoundException
	 */
	private static void parseDataFile(File dataFile, List<FeatureVector> emptyDataSet) throws FileNotFoundException {
		int entryCounter = 0;

		Scanner fScanner = new Scanner(dataFile);
		while (fScanner.hasNextLine()) {
			String[] dataStringArray = fScanner.nextLine().split(" ");
			emptyDataSet.add(new FeatureVector(dataStringArray));
			entryCounter++;
		}
		fScanner.close();

		System.out.println("Parsing \"" + dataFile.getName() + "\", " + entryCounter + " entries read.");
	}

	/**
	 * Compute the training/validation/test data against the trained classifier,
	 * then return the corresponding error
	 * 
	 * @param k            - parameter for the K-Nearest Neighbor Problem
	 * @param inputDataSet
	 * @return training/validation/test error
	 */
	private static double kNNcompute(int k, List<FeatureVector> inputDataSet) {
		int dataSetSize = inputDataSet.size();
		int processedData = 0;
		int errors = 0;

		for (FeatureVector dataPt : inputDataSet) {
			int outputLable = kNNClassifier(k, dataPt);
			processedData++;
			if (outputLable != dataPt.getLabel()) {
				errors++;
			}
			// printProgress(errors, processedData, dataSetSize);
		}
		System.out.print(processedData + " data points processed, " + errors + " prediction error reported.");

		return (double) errors / (double) dataSetSize;
	}

	/**
	 * Train the Classifier, compute Validation/Training/Test Error against the
	 * prediction rule.
	 * 
	 * @param args
	 */
	public static void main(String[] args) {

		// data files
		File trainingDataFile = new File("pa1train.txt");
		File validationDataFile = new File("pa1validate.txt");
		File testDataFile = new File("pa1test.txt");

		// parse training data set
		List<FeatureVector> trainingDataSet = new LinkedList<>();
		try {
			parseDataFile(trainingDataFile, trainingDataSet);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		trainingData = trainingDataSet;

		// parse validation data set
		List<FeatureVector> validationDataSet = new LinkedList<>();
		try {
			parseDataFile(validationDataFile, validationDataSet);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		// parse test data set
		List<FeatureVector> testDataSet = new LinkedList<>();
		try {
			parseDataFile(testDataFile, testDataSet);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		// test K-NN Classifier with k = 1, 5, 9 and 15.
		int[] params = new int[] { 1, 3, 5, 9, 15 };
		for (int i = 0; i < params.length; i++) {
			int k = params[i];

			System.out.println("\n>> K-NN Classifier with k = " + k);

			long clockStart = System.currentTimeMillis();
			System.out.println("\nComputing K-NN against Training data ...");
			double trainingError = kNNcompute(k, trainingDataSet);
			System.out.println("\nTraining Error (k = " + k + ") = " + trainingError);
			long clockStop = System.currentTimeMillis();
			double durationInSec = (double) (clockStop - clockStart) / 1000;
			System.out.println("Time Elapsed: " + durationInSec + "s");

			clockStart = System.currentTimeMillis();
			System.out.println("\nComputing K-NN against Validation data ...");
			double validationError = kNNcompute(k, validationDataSet);
			System.out.println("\nValidation Error (k = " + k + ") = " + validationError);
			clockStop = System.currentTimeMillis();
			durationInSec = (double) (clockStop - clockStart) / 1000;
			System.out.println("Time Elapsed: " + durationInSec + "s");

			clockStart = System.currentTimeMillis();
			System.out.println("\nComputing K-NN against Test data ...");
			double testError = kNNcompute(k, testDataSet);
			System.out.println("\nTest Error (k = " + k + ") = " + testError);
			clockStop = System.currentTimeMillis();
			durationInSec = (double) (clockStop - clockStart) / 1000;
			System.out.println("Time Elapsed: " + durationInSec + "s");
		}

		System.out.println("\nComputation completed.");
	}
}
