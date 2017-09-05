
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

import com.wuawua.research.fnn.data.DataRecord;
import com.wuawua.research.fnn.data.Feature;
import com.wuawua.research.fnn.data.SimpleDataSet;
import com.wuawua.research.fnn.layer.impl.fm.FmInputLayer;
import com.wuawua.research.fnn.layer.impl.fm.FmOutputLayer;
import com.wuawua.research.fnn.math.Matrix;
import com.wuawua.research.fnn.math.Vector;
import com.wuawua.research.fnn.model.Dictionary;
import com.wuawua.research.fnn.neuralnetwork.NeuralNetwork;
import com.wuawua.research.fnn.neuralnetwork.impl.FmNeuralNetwork;


public class MyMediaLiteConverter {

    private static final int NUM_USERS = 943;
    private static final int NUM_ITEMS = 1682;
    public static Args args = new Args();
	public static Dictionary dict = new Dictionary(args);

    public static void main(String[] args) throws Exception {
        //FMData train = getRecommendationDataset("data/u1.base");
        //FMData test = getRecommendationDataset("data/u1.test");
    	String trainFile = "data/u1.base";
    	String testFile = "data/u1.test";
    	
    	
    	getConvertRatingFile(trainFile, "data/u1.base.mml", null );
    	getConvertRatingFile(testFile, "data/u1.test.mml", null );
    }
    
    
   
    
    public static void getConvertRatingFile(String inputFile, String outputFile,  List<Integer> skips) throws IOException {
		List<String> writeLines = new ArrayList<String>();
		
		List<String> lines = FileUtils.readLines(new File(inputFile), "UTF-8");
		for(String line : lines) {
			if(line.startsWith("#")) {
				continue;
			}
			String[] tokens = line.split("\\s|\t");
			StringBuilder content = new StringBuilder();
			int index = 0;
			content.append(tokens[0]).append("\t");
			content.append(tokens[1]).append("\t");
			content.append(tokens[2]);
			writeLines.add(content.toString().trim());
		}

		Path trainFile = Paths.get(outputFile);
		if(writeLines.size() > 0) {
			Files.write(trainFile, writeLines, Charset.forName("UTF-8"));
		}
	}
    
    
}
