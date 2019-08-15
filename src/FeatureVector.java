/**
 * Feature vector of data set, consists of 784 dimension coordinate
 * and one integer label.
 * @author Wenlyu Ye
 */
public class FeatureVector {
	private int[] coordinate;
	private int label;

	FeatureVector(String[] inputData) {
		final int DIMENSION = 784;
		
		if (inputData.length != DIMENSION + 1) {
			System.out.println("WARNING: Data entry has invalid length.");
		}

		coordinate = new int[DIMENSION];
		for (int i = 0; i < DIMENSION; i++) {
			coordinate[i] = Integer.parseInt(inputData[i]);
		}

		label = Integer.parseInt(inputData[DIMENSION]);
	}

	public int[] getCoordinate() {
		return coordinate;
	}

	public int getLabel() {
		return label;
	}
}