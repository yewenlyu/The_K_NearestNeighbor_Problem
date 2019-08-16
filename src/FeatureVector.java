/**
 * Feature vector of data set, consists of 784 dimension coordinate and one
 * integer label.
 * 
 * @author Wenlyu Ye
 */
public class FeatureVector {
	private double[] coordinate;
	private int label;

	FeatureVector(String[] inputData) {
		final int DIMENSION = 784;

		coordinate = new double[DIMENSION];
		for (int i = 0; i < DIMENSION; i++) {
			coordinate[i] = Double.parseDouble(inputData[i]);
		}

		label = Integer.parseInt(inputData[DIMENSION]);
	}

	public double[] getCoordinate() {
		return coordinate;
	}

	public int getLabel() {
		return label;
	}

	/**
	 * Project the Feature Vector onto the vector space spanned by 20 orthonormal
	 * vectors
	 * 
	 * @param pMatrix - dimension 784x20, orthonormal columns
	 */
	public void project(double[][] pMatrix) {
		final int ORIG_DIM = 784;
		final int PROJ_DIM = 20;
		
		double[] projection = new double[PROJ_DIM]; 
		
		// matrix multiplication
		for (int n = 0; n < PROJ_DIM; n++) {
			projection[n] = 0;
			for (int m = 0; m < ORIG_DIM; m++) {
				projection[n] += coordinate[m] * pMatrix[m][n];
			}
		}
		
		coordinate = projection;
	}
}