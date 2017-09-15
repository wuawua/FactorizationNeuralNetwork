package com.wuawua.research.nn.layer;

import java.util.Random;
import java.util.function.DoubleBinaryOperator;

import com.wuawua.research.nn.data.DataRecord;
import com.wuawua.research.nn.math.Matrix;
import com.wuawua.research.nn.math.Vector;
import com.wuawua.research.nn.optimizer.Optimizer;

/***
 * Layer of neural network
 * @author Huang Haiping
 *
 * @param <T>
 */
public abstract class Layer<T> {
	
	protected static final DoubleBinaryOperator SUM = (x, y) -> x + y;

    protected final int dim;
    protected final float learnRate;
    protected final Vector bias;
    protected final Vector regBias;
    protected final Matrix weights;
    protected final Vector regWeights;
    protected float loss;
    
    protected Optimizer<T> optimizer;

    /**
     * Constructor.
     *
     * @param bias initial bias
     * @param weights initial feature weight vector
     */
    public Layer(int dim, float learnRate, Vector bias, Vector regBias,  Matrix weights, Vector regWeights, Optimizer<T> optimizer) {
        this.dim = dim;
        this.learnRate = learnRate;
        this.bias = bias;
        this.regBias = regBias;
        this.weights = weights;
        this.regWeights = regWeights;
        this.optimizer = optimizer;
        if(this.optimizer != null) {
        	this.optimizer.register(this);
        }
    }

    /**
     * Constructor
     * 
     * @param dim            Dimension of this layer
     * @param numFeatures
     * @param learnRate
     * @param regBias
     * @param regWeights
     * @param rnd
     * @param sdev
     */
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

    /**
     * Forward propagate
     * @param record: data record of train data with features
     * @param hidden
     * @return
     */
    public abstract Vector forwardPropagate(DataRecord<T> record, Vector hidden);
    
    
    /**
     * Forward propagate
     * @param record: data record of train data with features
     * @param hidden
     * @return
     */
    public abstract Matrix forwardPropagate(DataRecord<T> record, Matrix hidden);
    
    /**
     * Backward propagate
     * @param record
     * @param output
     * @param hidden
     * @param gradient
     * @param percent
     * @return
     */
    public abstract Vector backwardPropagate(DataRecord<T> record, Vector output, Vector hidden, Vector gradient);

    
    /**
     * Backward propagate
     * @param record
     * @param output
     * @param hidden
     * @param gradient
     * @param percent
     * @return
     */
    public abstract Matrix backwardPropagate(DataRecord<T> record, Matrix output, Matrix hidden, Matrix gradient);

    
    /**
     * Accumulate gradient
     * @param grad
     */
    public abstract void accumulateGradient(Vector grad);
    
    /**
     * Dimension of this layer
     * @return
     */
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
    
    public Vector getRegWeights() {
    	return regWeights;
    }

	public float getLoss() {
		return loss;
	}
	
	public float getLearnRate() {
		return learnRate;
	}
}
