package com.wuawua.research.fnn.model;

import java.util.Vector;

/***
 * 
 * @author Huang Haiping (hhp05@mails.tsinghua.edu.cn)
 *
 */
public class WordEntry {
	public String word;
	public long count;
	public WordEntryType type;
	public Vector<Integer> subwords;
}
