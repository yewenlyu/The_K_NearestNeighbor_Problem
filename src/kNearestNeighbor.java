import java.io.File;
import java.io.FileNotFoundException;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;

/**
 * The k Nearest Neighbor problem tester
 * @author Wenlyu Ye
 */
public class kNearestNeighbor {

	/**
	 * This function parses the input data set file into a List of FeatureVectors.
	 * @param dataFile 		- 
	 * @param emptyDataSet 	- container list
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
	 * The classifier of KNN with prediction rule trained on training data set
	 * @param k - parameter for the K-Nearest Neighbor Problem
	 * @param x - input data point from training/validation/test data set
	 * @return	- predicted label for x from the trained classifier
	 */
	private static int kNNClassifier(int k, FeatureVector x, List<FeatureVector> trainingData) {

		return 0;
	}

	/**
	 * Train the Classifier, calculate validation/train/test error
	 * against the prediction rule.
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

		// input parameter k
		System.out.print("Input parameter k for k-NN: ");
		Scanner inScanner = new Scanner(System.in);
		int k = Integer.parseInt(inScanner.nextLine());
		inScanner.close();
	}
}
