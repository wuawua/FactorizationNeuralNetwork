
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
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.apache.commons.io.FileUtils;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.Well19937c;

import com.wuawua.research.fnn.model.Dictionary;
import com.wuawua.research.nn.data.DataRecord;
import com.wuawua.research.nn.data.Feature;
import com.wuawua.research.nn.data.SimpleDataSet;
import com.wuawua.research.nn.layer.impl.fm.FmInputLayer;
import com.wuawua.research.nn.layer.impl.fm.FmOutputLayer;
import com.wuawua.research.nn.math.Matrix;
import com.wuawua.research.nn.math.Vector;
import com.wuawua.research.nn.neuralnetwork.NeuralNetwork;
import com.wuawua.research.nn.neuralnetwork.impl.FmNeuralNetwork;


public class FmConverter {

    private static final int NUM_USERS = 943;
    private static final int NUM_ITEMS = 1682;
    public static Args args = new Args();
	public static Dictionary dict = new Dictionary(args);

    public static void main(String[] args) throws Exception {
        //FMData train = getRecommendationDataset("data/u1.base");
        //FMData test = getRecommendationDataset("data/u1.test");
    	String trainFile = "data/u1.feature.base.txt";
    	String testFile = "data/u1.feature.test.txt";
    	
    	ArrayList<Integer> skips = new ArrayList<Integer>();
		
    	//skips.add(1);
		//skips.add(2);
		//skips.add(3);
		//skips.add(4);
		//skips.add(5);
		//skips.add(6);
		//skips.add(7);
		//skips.add(8);
	    //skips.add(9);
		
    	
    	dict.readFromFile(trainFile, 0, skips);
    	dict.readFromFile(testFile, 0, skips);
    	
    	System.out.println(dict.getWords());
    	
    	getConvertRatingFile(trainFile, "data/u1.base.ffm", skips );
    	getConvertRatingFile(testFile, "data/u1.test.ffm", skips );
    }
    
    
   
    
    public static void getConvertRatingFile(String inputFile, String outputFile,  List<Integer> skips) throws IOException {
		List<String> writeLines = new ArrayList<String>();
		
		List<String> lines = FileUtils.readLines(new File(inputFile), "UTF-8");
		for(String line : lines) {
			if(line.startsWith("#")) {
				continue;
			}
			DataRecord<Feature> record = dict.getDataRecord(line, 0, skips);
			StringBuilder content = new StringBuilder();
			content.append((int)(record.getTarget())).append(" ");
			int field = 1;
			for(Feature feature : record.getFeatures()) {
				content.append(field).append(":");
				content.append(feature.getID()).append(":");
				content.append(feature.getValue());
				content.append(" ");
				field++;
			}
			writeLines.add(content.toString().trim());
		}

		Path trainFile = Paths.get(outputFile);
		if(writeLines.size() > 0) {
			Files.write(trainFile, writeLines, Charset.forName("UTF-8"));
		}
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
