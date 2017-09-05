package com.wuawua.research.fnn.model;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Vector;

import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.log4j.Logger;

import com.google.common.base.Preconditions;
import com.google.common.primitives.UnsignedInteger;
import com.wuawua.research.fnn.Args;
import com.wuawua.research.fnn.Args.model_name;
import com.wuawua.research.fnn.data.DataRecord;
import com.wuawua.research.fnn.data.Feature;
import com.wuawua.research.fnn.utils.IOUtil;

import it.unimi.dsi.fastutil.longs.Long2IntOpenHashMap;

public class Dictionary {

	private static Logger logger = Logger.getLogger(Dictionary.class);

	private static final int MAX_FEATURE_SIZE = 1000000;
	private static final int MAX_LINE_SIZE = 1024;

	public Vector<WordEntry> words;         //All words of feature
	public Vector<Float> probDiscard;       //Probability of discard
	public Long2IntOpenHashMap word2int;    // Map<Long, Integer>
	public int size;
	public int numWords;
	public int numLabels;
	public long numTokens;

	private Args args;
	
	private final int ID_FIELD_VALUE_LENGTH = 3;
	private final int ID_VALUE_LENGTH = 2;
	
	private final int ID_FIELD_VALUE_FIELD_INDEX = 1;
	private final int ID_FIELD_VALUE_VALUE_INDEX = 2;
	private final int ID_VALUE_VALUE_INDEX = 1;
	
	public Dictionary(Args args) {
		size = 0;
		numWords = 0;
		numLabels = 0;
		numTokens = 0;
		word2int = new Long2IntOpenHashMap(MAX_FEATURE_SIZE);
		((Long2IntOpenHashMap) word2int).defaultReturnValue(-1);
		words = new Vector<WordEntry>(MAX_FEATURE_SIZE);
		this.args = args;
	}

	public long find(final String w) {
		long h = hash(w) % MAX_FEATURE_SIZE;
		WordEntry e = null;
		while (word2int.get(h) != -1 && ((e = words.get(word2int.get(h))) != null && !w.equals(e.word))) {
			h = (h + 1) % MAX_FEATURE_SIZE;
		}
		return h;
	}

	public void add(final String word) {
		long h = find(word);
		numTokens++;
		if (word2int.get(h) == -1) {
			WordEntry e = new WordEntry();
			e.word = word;
			e.count = 1;
			e.type = word.contains(args.label) ? WordEntryType.label : WordEntryType.word;
			words.add(e);
			word2int.put(h, size++);
		} else {
			words.get(word2int.get(h)).count++;
		}
	}

	public int getWords() {
		return numWords;
	}

	public int getNumLabels() {
		return numLabels;
	}

	public long getNumTokens() {
		return numTokens;
	}

	public boolean discard(int id, float rand) {
		Preconditions.checkArgument(id >= 0);
		Preconditions.checkArgument(id < numWords);
		if (args.model == model_name.sup) {
			return false;
		}
		return rand > probDiscard.get(id);
	}
	
	public int getId(final String w) {
		long h = find(w);
		return word2int.get(h);
	}

	public WordEntryType getType(int id) {
		Preconditions.checkArgument(id >= 0);
		Preconditions.checkArgument(id < size);
		return words.get(id).type;
	}
	
	public String getWord(int id) {
		Preconditions.checkArgument(id >= 0);
		Preconditions.checkArgument(id < size);
		return words.get(id).word;
	}
	
	public String getLabel(int lid) {
		Preconditions.checkArgument(lid >= 0);
		Preconditions.checkArgument(lid < numLabels);
		return words.get(lid + numWords).word;
	}
	
	/**
	 * String FNV-1a Hash
	 * 
	 * @param str
	 * @return
	 */
	public static long hash(final String str) {
		int h = (int) 2166136261L;// 0xffffffc5;
		for (byte strByte : str.getBytes()) {
			h = (h ^ strByte) * 16777619; // FNV-1a
		}
		return UnsignedInteger.fromIntBits(h).longValue();
	}

	

	private transient Comparator<WordEntry> entry_comparator = new Comparator<WordEntry>() {
		@Override
		public int compare(WordEntry o1, WordEntry o2) {
			int cmp = o1.type.value > o2.type.value ? +1 : o1.type.value < o2.type.value ? -1 : 0;
			if (cmp == 0) {
				cmp = o1.count > o2.count ? +1 : o1.count < o2.count ? -1 : 0;
			}
			return cmp;
		}
	};

	public void threshold(long t) {
		Collections.sort(words, entry_comparator);
		Iterator<WordEntry> iterator = words.iterator();
		while (iterator.hasNext()) {
			WordEntry _entry = iterator.next();
			if (_entry.count < t && _entry.type == WordEntryType.word ) {
				iterator.remove();
			}
		}
		size = 0;
		numWords = 0;
		numLabels = 0;
		word2int.clear();
		for (WordEntry _entry : words) {
			long h = find(_entry.word);
			word2int.put(h, size++);
			if (_entry.type == WordEntryType.word) {
				numWords++;
			}
			if (_entry.type == WordEntryType.label) {
				numLabels++;
			}
				
		}
	}
	
	public void initTableDiscard() {
		probDiscard = new Vector<Float>(size);
		for (int i = 0; i < size; i++) {
			float f = (float) (words.get(i).count) / (float) numTokens;
			probDiscard.add((float) (Math.sqrt(args.t / f) + args.t / f));
		}
	}

	public Vector<Long> getCounts(WordEntryType type) {
		Vector<Long> counts = new Vector<Long>(words.size());
		for (WordEntry w : words) {
			if (w.type == type) {
				counts.add(w.count);
			}
		}
		return counts;
	}

	/**
	 * Read all feature information from file.
	 * The data format is the same as in SVMlite and LIBSVM, or LIBFFM
	 * example:
	 *     SVMlite format:    4 0:1.5 3:-7.9
	 *     LIBFFM format:     4 0:1:1.5 3:2:-7.9
	 * @param file
	 * @param labelIndex
	 * @param skipColumns
	 * @throws IOException
	 */
	public void readFromFile(String file, int labelIndex, List<Integer> skipColumns) throws IOException {
		FileInputStream fis = new FileInputStream(file);
		BufferedReader br = new BufferedReader(new InputStreamReader(fis, "UTF-8"));
		try {
			long minThreshold = 1;
			String line;
			while ((line = br.readLine()) != null) {
				if (line.length() == 0 || line.startsWith("#")) {
					continue;
				}
				String[] words = line.split("\\s+|\t");
				for (int ii = 0; ii < words.length; ii++) {
					if(skipColumns != null && skipColumns.contains(ii)) {
						continue;
					}
					
					String word = words[ii];
					if(ii == labelIndex) {
						word = args.label + word;
						continue;
					}
					String[] features = word.split(":|::");
					if(features.length > 0) {
						StringBuilder feature = new StringBuilder();
						feature.append("f").append(ii).append("_").append(features[0]);
						word = feature.toString();
					}
					add(word);
					if (numTokens % 1000000 == 0) {
						System.out.println("Read " + numTokens / 1000000 + "M words");
					}
					if (size > 0.75 * MAX_FEATURE_SIZE) {
						threshold(minThreshold++);
					}
				}
			}
		} finally {
			fis.close();
			br.close();
		}
		System.out.println("\rRead " + numTokens  + " words");
		threshold(args.minCount);
		initTableDiscard();
	}
	
	/**
	 * Read a rating record.
	 * The data format is the same as in SVMlite and LIBSVM, or LIBFFM
	 * example:
	 *     SVMlite format:    4 0:1.5 3:-7.9
	 *     LIBFFM format:     4 0:1:1.5 3:2:-7.9
	 * @param line
	 * @param labelIndex
	 * @param skips
	 * @return
	 * @throws IOException
	 */
	public DataRecord<Feature> getDataRecord(String line, int labelIndex, List<Integer> skips)
			throws IOException {
		DataRecord<Feature> record = new DataRecord<Feature>(); 
		if (line != null) {
			if (line.length() == 0 || line.startsWith("#")) {
				return null;
			}
			String[] words = line.split("\\s+|\t");
			float target = 0.0f;
			for(int ii = 0; ii < words.length; ii++) {
				//Skip some columns
				if(skips != null && skips.contains(ii)) {
					continue;
				}
				
				String word = words[ii];
				
				//Parse target
				if(ii == labelIndex) {
					try {
						target = Float.parseFloat(word);
					}
					catch(Exception e) {
					}
					
					record.setTarget(target);
					continue;
				}
				
				//Parse features
				String[] tokens = word.split(":|::");
				int field = 0;
				float value = 1.0f;
				if(tokens.length > 0) {
					
					//Replace to format field_value, f1_1234
					StringBuilder feature = new StringBuilder();
					feature.append("f").append(ii).append("_").append(tokens[0]);
					word = feature.toString();	
					
					//Read field or value, Id:Field:Value or Id:Value
					if(tokens.length == ID_FIELD_VALUE_LENGTH ) {
						for(int jj = 0; jj < tokens.length; jj++) {
							if(jj == ID_FIELD_VALUE_FIELD_INDEX) {
								try {
									field = Integer.parseInt(tokens[jj]);
								}
								catch(Exception e) {
								}
							}
							else if(jj == ID_FIELD_VALUE_VALUE_INDEX) {
								try {
									value = Float.parseFloat(tokens[jj]);
								}
								catch(Exception e) {
								}
							}
						}
					}
					else if(tokens.length == ID_VALUE_LENGTH){
						for(int jj = 0; jj < tokens.length; jj++) {
							if(jj == ID_VALUE_VALUE_INDEX) {
								try {
									value = Float.parseFloat(tokens[jj]);
								}
								catch(Exception e) {
								}
							}
						}
					}
				}
				
				int wid = getId(word);
				if (wid < 0) {
					continue;
				}
				
				Feature feature = new Feature(wid, field, value);
				record.add(feature);
			}
		}
		return record;
	}

	public void save(OutputStream ofs) throws IOException {
		ofs.write(IOUtil.intToByteArray(size));
		ofs.write(IOUtil.intToByteArray(numWords));
		ofs.write(IOUtil.intToByteArray(numLabels));
		ofs.write(IOUtil.longToByteArray(numTokens));
		for (int i = 0; i < size; i++) {
			WordEntry e = words.get(i);
			ofs.write(e.word.getBytes());
			ofs.write(0);
			ofs.write(IOUtil.longToByteArray(e.count));
			ofs.write(e.type.value & 0xFF);
		}
	}

	public void load(InputStream ifs) throws IOException {
		words.clear();
		word2int.clear();
		size = IOUtil.readInt(ifs);
		numWords = IOUtil.readInt(ifs);
		numLabels = IOUtil.readInt(ifs);
		numTokens = IOUtil.readLong(ifs);

		if (logger.isDebugEnabled()) {
			logger.debug("size_: " + size);
			logger.debug("nwords_: " + numWords);
			logger.debug("nlabels_: " + numLabels);
			logger.debug("ntokens_: " + numTokens);
		}

		for (int i = 0; i < size; i++) {
			WordEntry e = new WordEntry();
			e.word = IOUtil.readString((DataInputStream) ifs);
			e.count = IOUtil.readLong(ifs);
			e.type = WordEntryType.fromValue(((DataInputStream) ifs).readByte() & 0xFF);
			words.add(e);
			word2int.put(find(e.word), i);

			if (logger.isDebugEnabled()) {
				logger.debug("e.word: " + e.word);
				logger.debug("e.count: " + e.count);
				logger.debug("e.type: " + e.type);
			}
		}
		initTableDiscard();
	}

}