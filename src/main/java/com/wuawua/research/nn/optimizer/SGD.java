package com.wuawua.research.nn.optimizer;

import com.wuawua.research.nn.layer.Layer;
import com.wuawua.research.nn.math.Matrix;
import com.wuawua.research.nn.math.Vector;

public class SGD<T> implements Optimizer<T> {

	long numEpochs;
	long miniBatchSize;
	float learnRate;
	Layer<T> layer;

	long cnt;

	public SGD(long epochs, long miniBatchSize, float learnRate) {
        this.numEpochs = epochs;
        this.miniBatchSize = miniBatchSize;
        this.learnRate = learnRate;
    }

    public void register(Layer<T> layer) {
    	this.layer = layer;
    }

    /**
     * Update weights
     * @param layer
     * @param gradient
     * @param index
     * @param xi
     */
    public void update(Vector gradient, int index, float xi) {
    	if(layer == null) {
    		return;
    	}
    	
    	Matrix weights = layer.getWeight();
    	Vector regWeights = layer.getRegWeights();
    	
    	for(int jj = 0; jj < weights.getColumns(); jj++) {
			float updateValue = - (learnRate * (gradient.get(jj) * xi + regWeights.get(index) * weights.get(index, jj)));
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
}
