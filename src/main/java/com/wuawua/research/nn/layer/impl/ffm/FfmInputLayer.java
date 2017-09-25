package com.wuawua.research.nn.layer.impl.ffm;

import java.util.Random;

import com.wuawua.research.nn.data.DataRecord;
import com.wuawua.research.nn.data.Feature;
import com.wuawua.research.nn.layer.Layer;
import com.wuawua.research.nn.math.Matrix;
import com.wuawua.research.nn.math.Vector;
import com.wuawua.research.nn.optimizer.Optimizer;

public class FfmInputLayer extends Layer<Feature> {
	
	private int numFields;
	
    /**
     * Constructor
     * 
     * @param dim
     * @param learnRate
     * @param bias
     * @param regBias
     * @param weights rows = feature size * field size
     * @param regWeights
     * @param optimizer
     */
    public FfmInputLayer(int dim, int numFields, float learnRate, Vector bias, Vector regBias, Matrix weights, Vector regWeights, Optimizer<Feature> optimizer) {
        super(dim, learnRate, bias, regBias, weights, regWeights, optimizer);
        this.numFields = numFields;
    }

    /***
     * Constructor
     * 
     * Weights matrix rows equal feature size * field size
     * @param dim
     * @param numFeatures
     * @param numField
     * @param learnRate
     * @param regBias
     * @param regWeights
     * @param rnd
     * @param sdev
     */
    public FfmInputLayer(int dim, int numFeatures, int numFields, float learnRate, Vector regBias, Vector regWeights, Random rnd, float sdev) {
    	super(dim, numFeatures * numFields, learnRate, regBias, regWeights, rnd, sdev);
    	this.numFields = numFields;
    }

    public Vector forwardPropagate(DataRecord<Feature> x, Vector hidden) {
    	return null;
    }
    
    @Override
	public Matrix forwardPropagate(DataRecord<Feature> record, Matrix hidden) {
    	Matrix nextHidden = new Matrix(numFields, dim);
    	for(Feature feature : record.getFeatures()) {
        	int ii = (int)feature.getID();
        	int field = feature.getField();
        	float xi = feature.getValue();
        	
        	for(int jj = 0; jj < dim; jj++) {
        		float value =  xi * weights.get(ii, jj);
        		nextHidden.add(field, jj, value);
        	}
        }
    	return nextHidden;
	}

    /**
     * Not implements.
     */
    @Override
    public Vector backwardPropagate(DataRecord<Feature> x, Vector output, Vector hidden, Vector gradient) {
    	return null;
    }
    
    @Override
	public Matrix backwardPropagate(DataRecord<Feature> record, Matrix output, Matrix hidden, Matrix gradient) {
    	for(Feature feature : record.getFeatures()) {
        	int ii = (int)feature.getID();
        	int field = feature.getField();
        	float xi = feature.getValue();
        	
        	optimizer.update(gradient, ii, xi, field);
    		
        	//for(int jj = 0; jj < dim; jj++) {
    		//	float updateValue = - (learnRate * (gradient.get(field, jj) * xi + regWeights.get(ii) * weights.get(ii, jj)));
    		//	weights.add(ii, jj, updateValue);
            //}
    	}
		return null;
	}

	@Override
	public void accumulateGradient(Vector grad) {
	}

	

	
}