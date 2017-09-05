
package com.wuawua.research.fnn;

import static java.lang.Double.parseDouble;
import static java.lang.Integer.parseInt;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.Well19937c;

import com.wuawua.research.fnn.data.DataRecord;
import com.wuawua.research.fnn.data.Feature;
import com.wuawua.research.fnn.data.SimpleDataSet;
import com.wuawua.research.fnn.layer.impl.fm.FmInputLayer;
import com.wuawua.research.fnn.layer.impl.fm.FmOutputLayer;
import com.wuawua.research.fnn.math.Matrix;
import com.wuawua.research.fnn.math.Vector;
import com.wuawua.research.fnn.model.Dictionary;
import com.wuawua.research.fnn.neuralnetwork.impl.FmNeuralNetwork;


public class ML100kRatingFm3 {

    private static final int NUM_USERS = 943;
    private static final int NUM_ITEMS = 1682;
    public static Args args = new Args();
	public static Dictionary dict = new Dictionary(args);

    public static void main(String[] args) throws Exception {
        //FMData train = getRecommendationDataset("data/u1.base");
        //FMData test = getRecommendationDataset("data/u1.test");
    	String trainFile = "data/u1.base";
    	String testFile = "data/u1.test";
    	
    	ArrayList<Integer> skips = new ArrayList<Integer>();
		skips.add(3);
    	dict.readFromFile(trainFile, 2, skips);
    	dict.readFromFile(testFile, 2, skips);
    	
    	System.out.println(dict.getWords());
    	
        SimpleDataSet<DataRecord<Feature>> train = getRecommendationDataset2(trainFile, skips);
        SimpleDataSet<DataRecord<Feature>> test = getRecommendationDataset2(testFile, skips);
        
        float learnRate = 0.01f;
        int numIter = 200;
        float sdev = 0.1f;
        float regB = 0.1f;
        int numFeatures = NUM_USERS + NUM_ITEMS + 7;
        int K = 50;
        int F = 100;
        Random rnd = new Random();
        
        Vector inputBias = new Vector(numFeatures);
        inputBias.zero();
        Matrix inputWeights = new Matrix(numFeatures, K);
        for (int ii = 0; ii < inputWeights.getRows(); ii++) {
            for (int jj = 0; jj < inputWeights.getColumns(); jj++) {
            	inputWeights.set(ii, jj, (float)rnd.nextGaussian() * sdev);
            }
        }
        Vector regW = new Vector(numFeatures);
        regW.fill(0.1f);
        int numLabels = 1;
        
        Vector outputBias = new Vector(numFeatures);
        outputBias.zero();
        Matrix outputWeights = new Matrix(numFeatures, K);
        for (int ii = 0; ii < outputWeights.getRows(); ii++) {
            for (int jj = 0; jj < outputWeights.getColumns(); jj++) {
            	outputWeights.set(ii, jj, (float)rnd.nextGaussian() * sdev);
            }
        }
        Vector regM = new Vector(numFeatures);
        regM.fill(0.1f);
        
        
        //BoundedFM fm = new BoundedFM(1.0, 5.0, train.numFeatures(), K, new Random(), sdev);
        
        FmInputLayer inputLayer = new FmInputLayer( K, learnRate, inputBias, regW, inputWeights,regW);
        FmOutputLayer outputLayer = new FmOutputLayer(K, numLabels, learnRate, outputBias, regM, outputWeights, regM);
        
        
        new FmNeuralNetwork(numLabels, inputLayer, outputLayer).learn(numIter, train, test);
    }

    
    
    public static SimpleDataSet<DataRecord<Feature>> getRecommendationDataset2(String file, List<Integer> skips) throws IOException {
    	SimpleDataSet<DataRecord<Feature>> dataset = new SimpleDataSet<DataRecord<Feature>>(NUM_USERS + NUM_ITEMS);


        InputStream is = new FileInputStream(file);

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(is))) {
        	String line = null;
        	while( (line = reader.readLine()) != null) {
            //reader.lines().forEach(line -> {
        		if(line.length() <= 0) {
        			continue;
        		}
        		
        		RandomGenerator rng = new Well19937c(12345);
        		UniformRealDistribution urd = new UniformRealDistribution(rng, 0, 1);
        		//System.out.println(line + "--");
        		DataRecord<Feature> record = dict.getDataRecord(line, 2, skips);
        		
        		//System.out.println(record.toString());
                dataset.add(record);
                
                /*
                String[] tokens = line.split("\t");
                int u = parseInt(tokens[0]) - 1;
                int i = parseInt(tokens[1]) - 1 + NUM_USERS;
                int u1Index = NUM_USERS + NUM_ITEMS;
                double u1 = Double.parseDouble(tokens[2]);
                int u2Index = NUM_USERS + NUM_ITEMS + 1;
                double u2 = Double.parseDouble(tokens[3]);
                int u3Index = NUM_USERS + NUM_ITEMS + 2;
                double u3 = Double.parseDouble(tokens[4]);
                int i1Index = NUM_USERS + NUM_ITEMS + 3;
                double i1 = Double.parseDouble(tokens[5]);
                int i2Index = NUM_USERS + NUM_ITEMS + 4;
                double i2 = Double.parseDouble(tokens[6]);
                int i3Index = NUM_USERS + NUM_ITEMS + 5;
                double i3 = Double.parseDouble(tokens[7]);
                int i4Index = NUM_USERS + NUM_ITEMS + 6;
                double i4 = Double.parseDouble(tokens[8]);
                double r = parseDouble(tokens[9]);

              //DataRecord<Feature> record = new DataRecord<Feature>();
                //record.setTarget((float)r);
                //record.add( new Feature(u, 0, 1));
                //record.add( new Feature(i, 0, 1));
                 
                 */
                
            }
        }

        return dataset;
    }

}
