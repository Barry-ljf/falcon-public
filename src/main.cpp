#include <iostream>
#include <string>
#include "AESObject.h"
#include "Precompute.h"
#include "secondary.h"
#include "connect.h"
#include "NeuralNetConfig.h"
#include "NeuralNetwork.h"
#include "unitTests.h"
#include "assert.h"
#include "Functionalities.h"


int partyNum;
AESObject* aes_indep;
AESObject* aes_next;
AESObject* aes_prev;
Precompute PrecomputeObject;


int main(int argc, char** argv)
{
/****************************** PREPROCESSING ******************************/ 
	parseInputs(argc, argv);
	NeuralNetConfig* config = new NeuralNetConfig(NUM_ITERATIONS);
	string network, dataset, security;
	bool PRELOADING = false;

/****************************** SELECT NETWORK ******************************/ 
	//Network {SecureML, Sarda, MiniONN, LeNet, AlexNet,RepVgg and VGG16}
	//Dataset {MNIST, CIFAR10,DIAMOND, and ImageNet}
	//Security {Semi-honest or Malicious}
	if (argc == 9)
	{network = argv[6]; dataset = argv[7]; security = argv[8];}
	else
	{
		network = "LeNet";
		dataset = "MNIST";
		security = "Semi-honest";
	}
	selectNetwork(network, dataset, security, config);
	config->checkNetwork();
	NeuralNetwork* net = new NeuralNetwork(config);
	cout << "--------------------net config--------------------------" << endl; 
	printNetwork(net);
/****************************** AES SETUP and SYNC ******************************/ 
// if (partyNum == PARTY_A)
// 	{
// 	aes_indep = new AESObject("files/keyA");
// 	aes_next = new AESObject("files/keyAB");
// 	aes_prev = new AESObject("files/keyAC");
// 	}

// 	if (partyNum == PARTY_B)
// 	{
// 	aes_indep = new AESObject("files/keyB");
// 	aes_next = new AESObject("files/keyBC");
// 	aes_prev = new AESObject("files/keyAB");
// 	}

// 	if (partyNum == PARTY_C)
// 	{
// 	aes_indep = new AESObject("files/keyC");
// 	aes_next = new AESObject("files/keyAC");
// 	aes_prev = new AESObject("files/keyBC");
// 	}	
	// aes_indep = new AESObject(argv[3]);
	// aes_next = new AESObject(argv[4]);
	// aes_prev = new AESObject(argv[5]);
	////没有上述内容也可以运行？

	 initializeCommunication(argv[2], partyNum);
	 synchronize(2000000);

/****************************** RUN NETWORK/UNIT TESTS ******************************/ 
	//Run these if you want a preloaded network to be tested
	// assert(NUM_ITERATIONS == 1 and "check if readMiniBatch is false in test(net)");
	// //First argument {SecureML, Sarda, MiniONN, or LeNet};
	// network += " preloaded"; PRELOADING = true;
	// preload_network(PRELOADING, network, net);

	// start_m();
	//Run unit tests in two modes: 
	//	1. Debug {Mat-Mul, DotProd, PC, Wrap, ReLUPrime, ReLU,FasterDivision, Division, BN, SSBits, SS, AdaptAvgpool and Maxpool}
	//	2. Test {Mat-Mul1, Mat-Mul2, Mat-Mul3 (and similarly) Conv*, ReLU*, ReLUPrime*, and Maxpool*} where * = {1,2,3}
	runTest("Debug", "FasterDivision", network);
	//runTest("Test", "Maxpool1", network);

	// Run forward/backward for single layers
	//  1. what {F, D, U}
	// 	2. l {0,1,....NUM_LAYERS-1}
	// size_t l = 0;
	// string what = "F";
	// runOnly(net, l, what, network);

	//Run training
	// network += " train";
	// train(net);

	//Run inference (possibly with preloading a network)
	// network += " test";
	// test(PRELOADING, network, net);

		// end_m(network);
		// cout << "----------------------------------------------" << endl;  	
		// cout << "Run details: " << NUM_OF_PARTIES << "PC (P" << partyNum 
		// 	 << "), " << NUM_ITERATIONS << " iterations, batch size " << MINI_BATCH_SIZE << endl 
		// 	 << "Running " << security << " " << network << " on " << dataset << " dataset" << endl;
		// cout << "----------------------------------------------" << endl << endl;  

		//printNetwork(net);

/****************************** CLEAN-UP ******************************/ 
	delete aes_indep;
	delete aes_next;
	delete aes_prev;
	delete net;
	delete config;
	
	deleteObjects();

	return 0;
}




