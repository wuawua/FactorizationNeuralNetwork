package com.wuawua.research.fnn.data;

public class Feature {
	private int id;
	private int field;
	private float value;
	//private float target;
	
	public Feature(int id, int field, float value) {
		this.id = id;
		this.field = field;
		this.value = value;
	}
	
	public int getID() {
		return id;
	}
	
	public void setID(int id) {
		this.id = id;
	}

	public int getField() {
		return field;
	}

	public void setField(int field) {
		this.field = field;
	}

	public float getValue() {
		return value;
	}

	public void setValue(float value) {
		this.value = value;
	}

	//public float getTarget() {
	//	return target;
	//}

	//public void setTarget(float target) {
	//	this.target = target;
	//}
	
	public String toString() {
		StringBuilder content = new StringBuilder();
		content.append(id).append(":");
		content.append(field).append(":");
		content.append(value);
		return content.toString();
	}
}
