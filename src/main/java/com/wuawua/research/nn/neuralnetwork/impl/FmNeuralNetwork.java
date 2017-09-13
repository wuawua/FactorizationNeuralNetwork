package com.wuawua.research.nn.neuralnetwork.impl;

import java.util.logging.Logger;

import com.wuawua.research.nn.data.DataRecord;
import com.wuawua.research.nn.data.DataSet;
import com.wuawua.research.nn.data.Feature;
import com.wuawua.research.nn.layer.Layer;
import com.wuawua.research.nn.math.Vector;
import com.wuawua.research.nn.neuralnetwork.NeuralNetwork;


/***
 * 
 * @author Huang Haiping
 *
 */
public class FmNeuralNetwork extends NeuralNetwork<Feature> {

    private static final Logger LOG = Logger.getLogger(FmNeuralNetwork.class.getName());
    private int trainNum = 0;
    
    public FmNeuralNetwork(int numLabels, Layer<Feature> inputLayer, Layer<Feature> outputLayer) {
    	super(numLabels, inputLayer, outputLayer);
    }
    
    /**
     * Train neural network with record
     * @param record
     */
    @Override
    public void learn(DataRecord<Feature> record) {
    	double percent = 1.0;
    	
    	//Forward
    	Vector hidden = inputLayer.forwardPropagate(record, null);
    	Vector output = outputLayer.forwardPropagate(record, hidden);
    	
    	//Backward
    	Vector gradient = outputLayer.backwardPropagate(record, output, hidden, null);
    	inputLayer.backwardPropagate(record, null, null, gradient);
    }
    
    /**
     * Learn parameters from data record.
     */
    @Override
    public void learn(int numIter, DataSet<DataRecord<Feature>> train, DataSet<DataRecord<Feature>> test) {
        train.shuffle();
        train.stream().forEach(x -> {
        	double percent = 1.0;
        	
        	//Forward
        	Vector hidden = inputLayer.forwardPropagate(x, null);
        	Vector output = outputLayer.forwardPropagate(x, hidden);
        	
        	//Backward
        	Vector gradient = outputLayer.backwardPropagate(x, output, hidden, null);
        	inputLayer.backwardPropagate(x, null, null, gradient);
        	
        });
    }
}
