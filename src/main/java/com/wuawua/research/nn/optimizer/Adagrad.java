package com.wuawua.research.nn.optimizer;

import com.wuawua.research.nn.layer.Layer;
import com.wuawua.research.nn.math.Matrix;
import com.wuawua.research.nn.math.Vector;

public class Adagrad<T> implements Optimizer<T> {

	long numEpochs;
	long miniBatchSize;
	float learnRate;
	Layer<T> layer;
	Matrix historicalGradient;
	
	public static final float DEFAULT_ADAGRAD_LEARNING_RATE = 1e-1f;
    public static final float DEFAULT_ADAGRAD_EPSILON = 1e-6f;
    

	//long cnt;

	public Adagrad(long epochs, long miniBatchSize, float learnRate) {
        this.numEpochs = epochs;
        this.miniBatchSize = miniBatchSize;
        this.learnRate = learnRate;
    }

    public void register(Layer<T> layer) {
    	this.layer = layer;
    	if(this.layer != null && this.layer.getWeight() != null) {
    		this.historicalGradient = new Matrix(this.layer.getWeight().getRows(), this.layer.getWeight().getColumns());
    		this.learnRate = layer.getLearnRate();
    	}
    }

    /**
     * Update weights
     * @param layer
     * @param gradient
     * @param index
     * @param xi
     */
    public void update(Vector gradient, int index, float xi) {
    	if(layer == null || historicalGradient == null) {
    		return;
    	}
    	
    	Matrix weights = layer.getWeight();
    	Vector regWeights = layer.getRegWeights();
    	//Vector regWeights = layer.getRegWeights();
    	for(int jj = 0; jj < weights.getColumns(); jj++) {
    		
			float g1 = gradient.get(jj) * xi + regWeights.get(index) * weights.get(index, jj);
			historicalGradient.add(index, jj, g1 * g1);
			float sqrtHistory = (float)Math.sqrt(historicalGradient.get(index, jj)) + DEFAULT_ADAGRAD_EPSILON;
			
			float updateValue = - (learnRate * g1 /sqrtHistory ); // -  learnRate * regWeights.get(index) * weights.get(index, jj);
			weights.add(index, jj, updateValue);
			
        }
    }

    public Optimizer<T> dup() {
    	return null;
    }



    public String toString() {

        String fullname = this.getClass().getName();

        String classname = fullname.substring(0, fullname.lastIndexOf('.'));

        return "opt." + classname +

            "[epochs:" + numEpochs +

            ", mini_batch_sz:" + miniBatchSize +

            ", lr:" + learnRate +

            "]";

    }

	@Override
	public void update(Matrix gradient, int index, float xi, int gradienIndex) {
		// TODO Auto-generated method stub
		
	}
}
