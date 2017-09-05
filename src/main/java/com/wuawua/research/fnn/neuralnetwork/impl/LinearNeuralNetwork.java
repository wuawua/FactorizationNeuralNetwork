package com.wuawua.research.fnn.neuralnetwork.impl;

import java.util.logging.Logger;

import com.wuawua.research.fnn.data.DataRecord;
import com.wuawua.research.fnn.data.DataSet;
import com.wuawua.research.fnn.data.Feature;
import com.wuawua.research.fnn.layer.Layer;
import com.wuawua.research.fnn.math.Vector;
import com.wuawua.research.fnn.neuralnetwork.NeuralNetwork;


/***
 * 
 * @author Huang Haiping
 *
 */
public class LinearNeuralNetwork extends NeuralNetwork<Feature> {

    private static final Logger LOG = Logger.getLogger(LinearNeuralNetwork.class.getName());
    private int trainNum = 0;
    
    public LinearNeuralNetwork(int numLabels, Layer<Feature> inputLayer, Layer<Feature> outputLayer) {
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
    	Vector hidden = inputLayer.forward(record, null);
    	Vector output = outputLayer.forward(record, hidden);
    	
    	//Backward
    	Vector gradient = outputLayer.backward(record, output, hidden, null, percent);
    	inputLayer.backward(record, null, null, gradient, percent);
    }
    
    /**
     * Learn parameters from data record.
     */
    @Override
    public void learn(int numIter, DataSet<DataRecord<Feature>> train, DataSet<DataRecord<Feature>> test) {
    	
    	int totalTrainNum = numIter * train.numRecords();
    	trainNum = 0;
    	System.out.println("totalTrainNum " + totalTrainNum);
        for (int t = 1; t <= numIter; t++) {
            long time0 = System.nanoTime();
            train.shuffle();
            train.stream().forEach(x -> {
            //for(DataRecord record : train.)
            	double percent = 1.0;
            	
            	//Forward
            	Vector hidden = inputLayer.forward(x, null);
            	Vector output = outputLayer.forward(x, hidden);
            	
            	//Backward
            	Vector gradient = outputLayer.backward(x, output, hidden, null, percent);
            	inputLayer.backward(x, null, null, gradient, percent);
            	
            	trainNum ++;
            });

            
            int iter = t;
            long time1 = System.nanoTime() - time0;

            //LOG.info(String.format("iteration n = %3d t = %.2fs", iter, time1 / 1_000_000_000.0));
            //LOG.info(() -> String.format("iteration n = %3d e = %.6f e = %.6f", iter, error(fm, train), error(fm, test)));
            System.out.println(iter + "," + error(train) + "," + error(test));
        }
        
    	
    	
    }
}
