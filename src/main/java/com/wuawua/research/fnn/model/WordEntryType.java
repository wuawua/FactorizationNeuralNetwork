package com.wuawua.research.fnn.model;

/***
 * 
 * @author Huang Haiping (hhp05@mails.tsinghua.edu.cn)
 *
 */
public enum WordEntryType {
	word(0), label(1);

	public int value;

	private WordEntryType(int value) {
		this.value = value;
	}

	public int getValue() {
		return this.value;
	}

	public static WordEntryType fromValue(int value) throws IllegalArgumentException {
		try {
			return WordEntryType.values()[value];
		} catch (ArrayIndexOutOfBoundsException e) {
			throw new IllegalArgumentException("Unknown entry_type enum value :" + value);
		}
	}

	@Override
	public String toString() {
		return value == 0 ? "word" : value == 1 ? "label" : "unknown";
	}
}