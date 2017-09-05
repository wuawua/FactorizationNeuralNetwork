package com.wuawua.research.fnn.layer;

import java.util.Random;
import java.util.function.DoubleBinaryOperator;

import com.wuawua.research.fnn.data.DataRecord;
import com.wuawua.research.fnn.math.Matrix;
import com.wuawua.research.fnn.math.Vector;

public  class Layer<T> {
	
	protected static final DoubleBinaryOperator SUM = (x, y) -> x + y;

    protected int dim;
    protected final float learnRate;
    protected final Vector bias;
    protected final Vector regBias;
    protected final Matrix weights;
    protected final Vector regWeights;
    private float loss;

    /**
     * Constructor.
     *
     * @param bias initial bias
     * @param weights initial feature weight vector
     */
    public Layer(int dim, float learnRate, Vector bias, Vector regBias,  Matrix weights, Vector regWeights) {
        this.dim = dim;
        this.learnRate = learnRate;
        this.bias = bias;
        this.regBias = regBias;
        this.weights = weights;
        this.regWeights = regWeights;
    }

    public Layer(int dim, int numFeatures, float learnRate, Vector regBias, Vector regWeights, Random rnd, float sdev) {
        this.dim = dim;
        this.learnRate = learnRate;
        this.regBias = regBias;
        this.regWeights = regWeights;
        this.bias = new Vector(numFeatures);
        this.weights = new Matrix(numFeatures, dim);
        
        bias.zero();
        
        for (int ii = 0; ii < weights.getRows(); ii++) {
            for (int jj = 0; jj < weights.getColumns(); jj++) {
                weights.set(ii, jj, (float)rnd.nextGaussian() * sdev);
            }
        }
    }

    public Vector forward(DataRecord<T> x, Vector hidden) {
        return null;
    }
    
    public Vector backward(DataRecord<T> x, Vector output, Vector hidden, Vector gradient, double percent) {
        return null;
    }

    public int getDimension() {
    	return dim;
    }

    /**
     * Get feature weight vector.
     *
     * @return feature weight vector
     */
    public Matrix getWeight() {
        return weights;
    }

	public float getLoss() {
		return loss;
	}
}
