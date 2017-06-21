#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_
#include "HyperParams.h"
#include "LSTM.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams{

public:
	Alphabet wordAlpha; // should be initialized outside
	LookupTable words; // should be initialized outside
	Alphabet featAlpha;
	UniParams hidden_linear;
	UniParams olayer_linear; // output
	LSTMParams lstm_params;
public:
	Alphabet labelAlpha; // should be initialized outside
	SoftMaxLoss loss;


public:
	bool initial(HyperParams& opts, AlignedMemoryPool* mem = NULL){

		// some model parameters should be initialized outside
		if (words.nVSize <= 0 || labelAlpha.size() <= 0){
			return false;
		}
		opts.wordDim = words.nDim;
		opts.wordWindow = opts.wordContext * 2 + 1;
		opts.windowOutput = opts.wordDim * opts.wordWindow;
		opts.labelSize = labelAlpha.size();
		hidden_linear.initial(opts.hiddenSize, opts.windowOutput, true, mem);
		opts.inputSize = opts.hiddenSize * 3;
		olayer_linear.initial(opts.labelSize, opts.inputSize, false, mem);
		lstm_params.initial(opts.hiddenSize, opts.windowOutput, mem);
		return true;
	}

	bool TestInitial(HyperParams& opts, AlignedMemoryPool* mem = NULL){

		// some model parameters should be initialized outside
		if (words.nVSize <= 0 || labelAlpha.size() <= 0){
			return false;
		}
		opts.wordDim = words.nDim;
		opts.wordWindow = opts.wordContext * 2 + 1;
		opts.windowOutput = opts.wordDim * opts.wordWindow;
		opts.labelSize = labelAlpha.size();
		opts.inputSize = opts.hiddenSize * 3;
		return true;
	}

	void exportModelParams(ModelUpdate& ada){
		words.exportAdaParams(ada);
		hidden_linear.exportAdaParams(ada);
		olayer_linear.exportAdaParams(ada);
		lstm_params.exportToAdaParams(ada);
	}


	void exportCheckGradParams(CheckGrad& checkgrad){
		checkgrad.add(&words.E, "words E");
		//checkgrad.add(&hidden_linear.W, "hidden w");
		//checkgrad.add(&hidden_linear.b, "hidden b");
		checkgrad.add(&olayer_linear.W, "output layer W");
		checkgrad.add(&lstm_params.cellParams.b(), "LSTM cell b");
		checkgrad.add(&lstm_params.cellParams.w1(), "LSTM cell w1");
		checkgrad.add(&lstm_params.cellParams.w2(), "LSTM cell w2");
		checkgrad.add(&lstm_params.forgetParams.w1(), "LSTM forget w1");
		checkgrad.add(&lstm_params.forgetParams.w2(), "LSTM forget w2");
		checkgrad.add(&lstm_params.forgetParams.w3(), "LSTM forget w3");
		checkgrad.add(&lstm_params.forgetParams.b(), "LSTM forget b");
		checkgrad.add(&lstm_params.inputParams.w1(), "LSTM input w1");
		checkgrad.add(&lstm_params.inputParams.w2(), "LSTM input w2");
		checkgrad.add(&lstm_params.inputParams.w3(), "LSTM input w3");
		checkgrad.add(&lstm_params.inputParams.b(), "LSTM input b");
		checkgrad.add(&lstm_params.outputParams.w1(), "LSTM output w1");
		checkgrad.add(&lstm_params.outputParams.w2(), "LSTM output w2");
		checkgrad.add(&lstm_params.outputParams.w3(), "LSTM output w3");
		checkgrad.add(&lstm_params.outputParams.b(), "LSTM output b");
	}

	// will add it later
	void saveModel(std::ofstream &os) const{
		wordAlpha.write(os);
		words.save(os);
		hidden_linear.save(os);
		olayer_linear.save(os);
		labelAlpha.write(os);
	}

	void loadModel(std::ifstream &is, AlignedMemoryPool* mem = NULL){
		wordAlpha.read(is);
		words.load(is, &wordAlpha, mem);
		hidden_linear.load(is, mem);
		olayer_linear.load(is, mem);
		labelAlpha.read(is);
	}

};

#endif /* SRC_ModelParams_H_ */