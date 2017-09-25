package com.wuawua.research.nn.layer.impl.fm;

import java.util.Random;

import com.wuawua.research.nn.data.DataRecord;
import com.wuawua.research.nn.data.Feature;
import com.wuawua.research.nn.layer.Layer;
import com.wuawua.research.nn.math.Matrix;
import com.wuawua.research.nn.math.Vector;
import com.wuawua.research.nn.optimizer.Optimizer;

public class FmInputLayer extends Layer<Feature> {
	
    /**
     * Constructor.
     *
     */
    public FmInputLayer(int dim, float learnRate, Vector bias, Vector regBias, Matrix weights, Vector regWeights, Optimizer<Feature> optimizer) {
        super(dim, learnRate, bias, regBias, weights, regWeights, optimizer);
    }

    public FmInputLayer(int dim, int numFeatures, float learnRate, Vector regBias, Vector regWeights, Random rnd, float sdev) {
    	super(dim, numFeatures, learnRate, regBias, regWeights, rnd, sdev);
    }

    @Override
    public Vector forwardPropagate(DataRecord<Feature> x, Vector hidden) {
    	Vector nextHidden = new Vector(dim);
        for(Feature feature : x.getFeatures()) {
        	int ii = (int)feature.getID();
        	float xi = feature.getValue();
        	
        	for(int jj = 0; jj < dim; jj++) {
        		nextHidden.add(jj, xi * weights.get(ii, jj) );
        	}
        }
        
        int size = x.getFeatureSize();
        nextHidden.mul((float)(1/(float)size));
        return nextHidden;
    }
    
    /**
     * No implements
     */
    @Override
	public Matrix forwardPropagate(DataRecord<Feature> record, Matrix hidden) {
		return null;
	}

    
    @Override
    public Vector backwardPropagate(DataRecord<Feature> x, Vector output, Vector hidden, Vector gradient) {
    	for(Feature feature : x.getFeatures()) {
        	int ii = (int)feature.getID();
        	float xi = feature.getValue();
        	
        	optimizer.update(gradient, ii, xi);
    		
        	//for(int jj = 0; jj < dim; jj++) {
    		//	float updateValue = - (learnRate * (gradient.get(jj) * xi + regWeights.get(ii) * weights.get(ii, jj)));
    		//	weights.add(ii, jj, updateValue);
            //}
    	}
    	return null;
    }

	@Override
	public void accumulateGradient(Vector grad) {
	}

	@Override
	public Matrix backwardPropagate(DataRecord<Feature> record, Matrix output, Matrix hidden, Matrix gradient) {
		return null;
	}

	
}