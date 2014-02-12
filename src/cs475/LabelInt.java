package cs475;

import java.io.Serializable;

public class LabelInt implements Serializable {
	private static final long serialVersionUID = 1L;
	
	Label _label = null;
	int _value = 0;

	public LabelInt(Label label, int value) {
		this._label = label;
		this._value = value;
	}

	public Label getLabel() {
		return _label;
	}

	public void setLabel(Label label) {
		this._label = label;
	}

	public int getValue() {
		return _value;
	}

	public void setValue(int value) {
		this._value = value;
	}	
	
}
