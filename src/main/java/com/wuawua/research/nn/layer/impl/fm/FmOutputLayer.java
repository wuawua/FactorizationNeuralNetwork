package com.wuawua.research.nn.layer.impl.fm;

import static java.lang.Math.max;
import static java.lang.Math.min;

import java.util.List;
import java.util.Random;

import com.wuawua.research.fnn.utils.concurrent.ConcurenceRunner;
import com.wuawua.research.fnn.utils.concurrent.TaskManager;
import com.wuawua.research.nn.data.DataRecord;
import com.wuawua.research.nn.data.Feature;
import com.wuawua.research.nn.layer.Layer;
import com.wuawua.research.nn.math.Matrix;
import com.wuawua.research.nn.math.Vector;
import com.wuawua.research.nn.optimizer.Optimizer;


public class FmOutputLayer extends Layer<Feature> {
	
    private int numLabels;
    private final int SINGLE_LABEL_NUMBER = 1;
    private Vector featureWeights;

    /**
     * Constructor.
     *
     * @param b initial bias
     * @param w initial feature weight vector
     * @param m initial feature interaction matrix
     */
    public FmOutputLayer(int dim, int numLabels, float learnRate, Vector bias, Vector regBias, Matrix weights, Vector regWeights, Optimizer<Feature> optimizer) {
    	super(dim, learnRate, bias, regBias, weights, regWeights, optimizer);
    	this.numLabels = numLabels;
    	this.featureWeights = new Vector(weights.getRows());
    }

    public FmOutputLayer(int dim, int numLabels, int numFeatures, float learnRate, Vector regBias, Vector regWeights, Random rnd, float sdev) {  
    	super(dim, numFeatures * numLabels, learnRate, regBias, regWeights, rnd, sdev);
    	this.numLabels = numLabels;
    	this.featureWeights = new Vector(numFeatures);
    }
    
    @Override
    public Vector forwardPropagate(DataRecord<Feature> x, Vector hidden) {
    	Vector output = new Vector(numLabels);
    	output.zero();
    	for(Feature feature : x.getFeatures()) {
        	int ii = (int)feature.getID();
        	float xi = feature.getValue();
    		for(int index = 0; index < numLabels; index++) {
    			output.add(index, bias.get(ii));
    			for(int jj = 0; jj < dim; jj++) {
    				output.add(index, hidden.get(jj) * xi * weights.get(ii * numLabels + index, jj));
    			}
    		}
    	}

    	if(numLabels == SINGLE_LABEL_NUMBER) {
    		return output;
    	}
    	else {
    		return computeSoftmax(output);
    	}
    }
    
    @Override
	public Matrix forwardPropagate(DataRecord<Feature> record, Matrix hidden) {
		// TODO Auto-generated method stub
		return null;
	}
    
    public Vector computeSoftmax(Vector output) {		
		double max = output.get(0);
		Vector result = new Vector(output.getRows());
		double z = 0.0f;
		for (int i = 0; i < numLabels; i++) {
			max = Math.max(output.get(i), max);
		}
		for (int i = 0; i < numLabels; i++) {
			output.add(i, (float)Math.exp(output.get(i) - max));
			z += output.get(i);
		}
		for (int i = 0; i < numLabels; i++) {
			result.add(i, (float)(output.get(i) / z));
		}
		return result;
	}
        
    @Override
    public Vector backwardPropagate(DataRecord<Feature> x, Vector output, Vector hidden, Vector gradient) {
    	Vector nextGradient = new Vector(dim);
    	int target = (numLabels == SINGLE_LABEL_NUMBER)?(int)(x.getTarget()):(int)(x.getTarget() - 1);
    	
    	for(Feature feature : x.getFeatures()) {
        	int ii = (int)feature.getID();
        	float xi = feature.getValue();
    		for(int index = 0; index < numLabels; index++) {
    			float label = (index == target) ? 1.0f : 0.0f;
    			if(numLabels == 1) {
    				label = target;
    			}
    			else {
    				label = (index == target) ? 1.0f : 0.0f;
    			}
    			float lambda = predict(output.get(index)) - label;
    			
    			//float biasUpdate = -learnRate * (lambda + regBias.get(ii) * bias.get(ii));
    			bias.add(ii, 
    					-learnRate * (lambda + regBias.get(ii) * bias.get(ii)));
    			
    			//optimizer.update(hidden, ii, xi);
    			
    			int wIndex = ii * numLabels + index;
    			
	    		for(int jj = 0; jj < dim; jj++) {	
	    			nextGradient.add(jj, lambda * xi * weights.get(wIndex, jj));
	    			//float updateWeight = -(float)(learnRate * (lambda * hidden.get(jj) * xi + regWeights.get(ii) * weights.get(wIndex,jj)));
	    			weights.add(wIndex, jj, 
	    					-(float)(learnRate * (lambda * hidden.get(jj) * xi + regWeights.get(ii) * weights.get(wIndex,jj))));
	            }
    		}
    	}
    	
    	
    	int size = x.getFeatureSize();
    	nextGradient.mul((float)(1/(float)size));
    	return nextGradient;
    }
    
    @Override
	public Matrix backwardPropagate(DataRecord<Feature> record, Matrix output, Matrix hidden, Matrix gradient) {
		return null;
	}
    
    public float predict(float x) {
    	float min = 1.0f;
    	float max = 5.0f;
        return x; //min(max, max(min, x));
    }


	@Override
	public void accumulateGradient(Vector grad) {
	}

	

	
}
