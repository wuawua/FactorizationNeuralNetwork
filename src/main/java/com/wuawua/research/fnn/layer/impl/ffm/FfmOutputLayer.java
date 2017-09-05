package com.wuawua.research.fnn.layer.impl.ffm;

import java.util.Random;

import com.wuawua.research.fnn.data.DataRecord;
import com.wuawua.research.fnn.data.Feature;
import com.wuawua.research.fnn.layer.Layer;
import com.wuawua.research.fnn.math.Matrix;
import com.wuawua.research.fnn.math.Vector;


public class FfmOutputLayer extends Layer<Feature> {
	
    private int numLabels;
    private int numFields;
    private Vector output;
    private int featureIndex;
    
    private final int SINGLE_LABEL_NUMBER = 1;

    /**
     * Constructor.
     *
     * @param b initial bias
     * @param w initial feature weight vector
     * @param m initial feature interaction matrix
     */
    public FfmOutputLayer(int dim, int numLabels, int numFields, float learnRate, Vector bias, Vector regBias, Matrix weights, Vector regWeights) {
    	super(dim, learnRate, bias, regBias, weights, regWeights);
    	this.numLabels = numLabels;
    	this.numFields = numFields;
    }

    public FfmOutputLayer(int dim, int numLabels, int numFields, int numFeatures, float learnRate, Vector regBias, Vector regWeights, Random rnd, float sdev) {  
    	super(dim, numFeatures * numLabels, learnRate, regBias, regWeights, rnd, sdev);
    	this.numLabels = numLabels;
    	this.numFields = numFields;
        this.output = new Vector(numLabels);
    }
    
    @Override
    public Vector forward(DataRecord<Feature> x, Vector hidden) {
    	Vector output = new Vector(numLabels);
    	output.zero();
    	for(Feature feature : x.getFeatures()) {
        	int ii = (int)feature.getID();
        	float xi = feature.getValue();
        	int field = feature.getField();
    		for(int index = 0; index < numLabels; index++) {
    			int biasIndex = ii * numFields + field;
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
    public Vector backward(DataRecord<Feature> x, Vector output, Vector hidden, Vector gradient, double percent) {
    	Vector nextGradient = new Vector(dim);
    	int target = (numLabels == SINGLE_LABEL_NUMBER)?(int)(x.getTarget()):(int)(x.getTarget() - 1);

    	featureIndex = 0;
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
    			float lambda = output.get(index) - label;
    			
    			float biasUpdate = -learnRate * (lambda + regBias.get(ii) * bias.get(ii));
    			bias.add(ii, biasUpdate);
    			
	    		for(int jj = 0; jj < dim; jj++) {
	    			nextGradient.add(jj, lambda * weights.get(ii * numLabels + index, jj));
	    			float updateWeight = -(float)(learnRate * (lambda * hidden.get(jj) * xi + regWeights.get(ii) * weights.get(ii * numLabels + index,jj)));
	    			weights.add((ii * numLabels + index), jj, updateWeight);
	            }
    		}
    		featureIndex++;
    	}
    	
    	int size = x.getFeatureSize();
    	nextGradient.mul((float)(1/(float)size));
    	return nextGradient;
    }
}
